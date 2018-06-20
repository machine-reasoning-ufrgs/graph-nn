import sys, os, time
import tensorflow as tf
import numpy as np
import random
from itertools import islice
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import model builder
from graphnn import GraphNN
from mlp import Mlp
from util import timestamp, memory_usage, dense_to_sparse, load_weights, save_weights
from tsp_utils import InstanceLoader, create_dataset_metric, create_dataset_random

def build_network(d):
	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define GNN dictionary
	GNN = {}

	# Define Graph neural network
	gnn = GraphNN(
		{
			# V is the set of vertices
			"V": d,
			# E is the set of edges
			"E": d
		},
		{
			# M is a E×V adjacency matrix connecting each edge to the vertices it's connected to
			"M": ("E","V"),
			# W is a column matrix of shape |E|×1 where W[i,1] is the weight of the i-th edge
			"W": ("E",1)
		},
		{
			# Vmsg is a MLP which computes messages from vertex embeddings to edge embeddings
			"Vmsg": ("V","E"),
			# Emsg is a MLP which computes messages from edge embeddings to vertex embeddings
			"Emsg": ("E","V")
		},
		{
			# V(t+1) ← Vu( Mᵀ × Emsg(E(t)) )
			"V": [
				{
					"mat": "M",
					"msg": "Emsg",
					"transpose?": True,
					"var": "E"
				}
			],
			# E(t+1) ← Eu( M × Vmsg(V(t)), W )
			"E": [
				{
					"mat": "M",
					"msg": "Vmsg",
					"var": "V"
				},
				{
					"mat": "W"
				}
			]
		},
		name="TSP"
	)

	# Define E_vote, which will compute one logit for each edge
	E_vote_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "E_vote",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
		)

	# Define placeholder for routes' edges (a mask of edges per problem)
	route_edges = tf.placeholder( tf.float32, [ None ], name = "route_edges" )

	# Define placeholder for routes' costs (one per problem)
	cost = tf.placeholder( tf.float32, [ None ], name = "cost" )

	# Placeholders for the list of number of vertices and edges per instance
	n_vertices 	= tf.placeholder( tf.int32, shape = (None,), name = "n_vertices" )
	n_edges 	= tf.placeholder( tf.int32, shape = (None,), name = "edges" )

	# Compute the number of vertices
	n = tf.shape( gnn.matrix_placeholders["M"] )[1]
	# Compute number of problems
	p = tf.shape( cost )[0]

	# Get the last embeddings
	E_n = gnn.last_states["E"].h
	E_vote = E_vote_MLP( E_n )

	# Compute a probability pᵢ ∈ [0,1] that each edge belongs to the TSP optimal route
	E_prob = tf.sigmoid(E_vote)

	"""
		Compute a 'fuzzy' cost for each edge by multiplying each edge weight
		with the corresponding edge probability
	"""
	cost_per_edge_fuzzy = tf.multiply(gnn.matrix_placeholders["W"], E_prob)

	"""
		Compute a 'binary' cost for each edge. Edges whose probabilities fall
		below 50% have their weights zeroed while edges whose probabilities
		fall above 50% have their weights unaltered
	"""
	cost_per_edge_binary = tf.multiply(gnn.matrix_placeholders["W"], tf.round(E_prob))

	# Reorganize votes' result to obtain a prediction for each problem instance
	def _vote_while_cond(i, n_acc, cost_predictions_fuzzy, cost_predictions_binary):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, n_acc, cost_predictions_fuzzy, cost_predictions_binary):
		
		# Gather the set of edge costs relative to the i-th problem
		costs_fuzzy_i 	= tf.gather(cost_per_edge_fuzzy,	tf.range(n_acc, tf.add(n_acc, n_edges[i])))
		costs_binary_i 	= tf.gather(cost_per_edge_binary,	tf.range(n_acc, tf.add(n_acc, n_edges[i])))

		# The total TSP cost for this problem is the sum of all its costs
		problem_cost_prediction_fuzzy	= tf.reduce_sum(costs_fuzzy_i)
		problem_cost_prediction_binary	= tf.reduce_sum(costs_binary_i)

		# Update TensorArray
		cost_predictions_fuzzy	= cost_predictions_fuzzy.write( i, problem_cost_prediction_fuzzy )
		cost_predictions_binary	= cost_predictions_binary.write( i, problem_cost_prediction_binary )
		return tf.add(i, tf.constant(1)), tf.add(n_acc, n_edges[i]), cost_predictions_fuzzy, cost_predictions_binary
	#end _vote_while_body
	
	# Obtain a list of predictions, one per problem
	cost_predictions_fuzzy 	= tf.TensorArray( size = p, dtype = tf.float32 )
	cost_predictions_binary = tf.TensorArray( size = p, dtype = tf.float32 )
	_, _, cost_predictions_fuzzy, cost_predictions_binary = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant(0), tf.constant(0), cost_predictions_fuzzy, cost_predictions_binary ]
	)
	cost_predictions_fuzzy 	= cost_predictions_fuzzy.stack()
	cost_predictions_binary = cost_predictions_binary.stack()

	# Define losses, accuracies, optimizer, train step
	
	# Define cost loss, which is the mean squared error between the 'fuzzy'
	# route cost computed from edge probabilities and the cost label
	#cost_loss = tf.losses.mean_squared_error(cost,cost_predictions_fuzzy)
	cost_loss = tf.losses.mean_squared_error(tf.reduce_sum(cost), tf.reduce_sum(cost_per_edge_binary))

	# Define edges loss, which is the binary cross entropy between the
	# computed edge probabilities and the edge (binary) labels, reduced among
	# all instances
	solution_edges 		= tf.cast(tf.reduce_sum(route_edges), tf.float32)
	total_edges 		= tf.cast(tf.shape(route_edges)[0], tf.float32)
	not_solution_edges	= tf.subtract(total_edges,solution_edges)
	edges_loss = tf.losses.sigmoid_cross_entropy(
			multi_class_labels = route_edges,
			logits = tf.reshape(E_vote, [-1]),
			weights = tf.add(
				tf.scalar_mul(tf.divide(solution_edges,total_edges), 		route_edges),
				tf.scalar_mul(tf.divide(not_solution_edges,total_edges), 	tf.subtract(tf.ones((tf.cast(total_edges, tf.int32),)), route_edges))
				)
			)

	# Define cost accuracy, which is the deviation between the 'binary' route
	# cost computed from the binarized edge probabilities and the cost label
	cost_acc = tf.reduce_mean(
		tf.div(
			tf.subtract(cost_predictions_binary, cost),
			tf.add(cost, tf.constant(10**(-5)))
			)
		)

	# Define edges accuracy, which is the proportion of correctly guessed edges
	edges_acc = tf.reduce_mean(
		tf.div(
			tf.reduce_sum(
				tf.multiply(
					route_edges,
					tf.cast(
						tf.equal(
							route_edges,
							tf.reshape(E_prob, [-1])
							),
						tf.float32
						)
					)
				),
				tf.maximum(tf.reduce_sum(route_edges), tf.constant(1.0))
			)
		)
	
	vars_cost 		= tf.zeros([])
	tvars 			= tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	
	loss 		= tf.add( cost_loss, tf.multiply( vars_cost, parameter_l2norm_scaling ) )
	
	optimizer 	= tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ 	= tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
	train_step 	= optimizer.apply_gradients( zip( grads, tvars ) )
	

	GNN["gnn"] 						= gnn
	GNN["n_vertices"]				= n_vertices
	GNN["n_edges"]					= n_edges
	GNN["cost"] 					= cost
	GNN["route_edges"]				= route_edges
	GNN["cost_predictions_fuzzy"] 	= cost_predictions_fuzzy
	GNN["cost_predictions_binary"] 	= cost_predictions_binary
	GNN["avg_cost_fuzzy"]			= tf.reduce_mean(cost_predictions_fuzzy)
	GNN["avg_cost_binary"]			= tf.reduce_mean(cost_predictions_binary)
	GNN["cost_loss"]				= cost_loss
	GNN["edges_loss"]				= edges_loss
	GNN["cost_acc"]					= cost_acc
	GNN["edges_acc"]				= edges_acc
	GNN["loss"] 					= loss
	GNN["train_step"] 				= train_step
	GNN["E_prob"]					= E_prob
	return GNN
#end

if __name__ == '__main__':
	
	create_datasets 	= True
	load_checkpoints	= False
	save_checkpoints	= True

	d 					= 128
	epochs 				= 1000
	batch_size			= 32
	batches_per_epoch 	= 128
	time_steps 			= 32

	if create_datasets:
		n = 20
		samples = batch_size*batches_per_epoch
		print("Creating {} train instances...".format(samples))
		create_dataset_metric(n, path="TSP-train", samples=samples)
		print("Creating {} test instances...".format(samples))
		create_dataset_metric(n, path="TSP-test", samples=samples)
	#end

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(d)

	# Create train, test loaders
	train_loader 	= InstanceLoader("TSP-train")
	test_loader		= InstanceLoader("TSP-test")

	# Disallow GPU use
	config = tf.ConfigProto( device_count = {"GPU":0})
	with tf.Session(config=config) as sess:
		
		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )

		# Restore saved weights
		if load_checkpoints: load_weights(sess,"./TSP-checkpoints");

		with open("log-TSP.dat","w") as logfile:
			# Run for a number of epochs
			print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
			for epoch in range( epochs ):

				# Reset train loader
				train_loader.reset()
				e_loss_train, e_acc_train, e_pred_train = 0, 0, 0
				for (batch_i, batch) in islice(enumerate(train_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels
					Ma_all, W_all, n_vertices, n_edges, route_edges, cost = batch

					total_vertices 	= sum(n_vertices)
					total_edges		= sum(n_edges)

					M 	= np.zeros((total_edges,total_vertices))
					W 	= np.zeros((total_edges,1))

					for (e,(i,j)) in enumerate(zip(list(np.nonzero(Ma_all)[0]), list(np.nonzero(Ma_all)[1]))):
						M[e,i] = 1
						M[e,j] = 1
						W[e,0] = W_all[i,j]
					#end

					# Run one SGD iteration
					_, loss, acc, pred, E_prob = sess.run(
						[ GNN["train_step"], GNN["loss"], GNN["cost_acc"], GNN["avg_cost_binary"], GNN["E_prob"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["M"]:	M,
							GNN["gnn"].matrix_placeholders["W"]:	W,
							GNN["n_vertices"]:						n_vertices,
							GNN["n_edges"]:							n_edges,
							GNN["gnn"].time_steps: 					time_steps,
							GNN["route_edges"]:						route_edges,
							GNN["cost"]: 							cost,
						}
					)

					e_loss_train 	+= loss
					e_acc_train 	+= acc
					e_pred_train 	+= pred

					# Print batch summary
					print(
						"{timestamp}\t{memory}\tTrain Epoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| (Loss,Acc,Avg.Pred): ({loss:.3f},{acc:.3f},{pred:.3f})".format(
							timestamp = timestamp(),
							memory = memory_usage(),
							epoch = epoch,
							batch = batch_i,
							loss = loss,
							acc = acc,
							pred = pred,
							n = total_vertices,
							m = total_edges,
							i = batch_size
						),
						flush = True
					)
				#end
				e_loss_train 	/= batches_per_epoch
				e_acc_train 	/= batches_per_epoch
				e_pred_train 	/= batches_per_epoch
				# Print train epoch summary
				print(
					"{timestamp}\t{memory}\tTrain Epoch {epoch}\tMain (Loss,Acc,Avg.Pred): ({loss:.3f},{acc:.3f},{pred:.3f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						loss = e_loss_train,
						acc = e_acc_train,
						pred = e_pred_train
					),
					flush = True
				)

				if save_checkpoints: save_weights(sess,"./TSP-checkpoints");

				# Reset test loader
				print("{timestamp}\t{memory}\tTesting...".format(timestamp=timestamp(), memory=memory_usage()))
				test_loader.reset()
				e_loss_test, e_acc_test, e_pred_test = 0, 0, 0
				for (batch_i, batch) in islice(enumerate(test_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels
					Ma_all, W_all, n_vertices, n_edges, route_edges, cost = batch

					total_vertices 	= sum(n_vertices)
					total_edges		= sum(n_edges)

					M 	= np.zeros((total_edges,total_vertices))
					W 		= np.zeros((total_edges,1))

					for (e,(i,j)) in enumerate(zip(list(np.nonzero(Ma_all)[0]), list(np.nonzero(Ma_all)[1]))):
						M[e,i] = 1
						M[e,j] = 1
						W[e,0] = W_all[i,j]
					#end

					loss, acc, pred = sess.run(
						[ GNN["loss"], GNN["cost_acc"], GNN["avg_cost_binary"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["M"]:	M,
							GNN["gnn"].matrix_placeholders["W"]:	W,
							GNN["n_vertices"]:						n_vertices,
							GNN["n_edges"]:							n_edges,
							GNN["gnn"].time_steps: 					time_steps,
							GNN["route_edges"]:						route_edges,
							GNN["cost"]: 							cost,
						}
					)

					e_loss_test 	+= loss
					e_acc_test 		+= acc
					e_pred_test 	+= pred
				#end
				e_loss_test 	/= batches_per_epoch
				e_acc_test 		/= batches_per_epoch
				e_pred_test 	/= batches_per_epoch
				# Print test epoch summary
				print(
					"{timestamp}\t{memory}\tTest Epoch {epoch}\tMain (Loss,Acc,Avg.Pred): ({loss:.3f},\t{acc:.3f},\t{pred:.3f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						loss = e_loss_test,
						acc = e_acc_test,
						pred = e_pred_test
					),
					flush = True
				)

				# Write train and test results into log file
				logfile.write("{epoch} {loss_train} {acc_train} {loss_test} {acc_test}\n".format(
					epoch = epoch,
					loss_train = e_loss_train,
					acc_train = e_acc_train,
					loss_test = e_loss_test,
					acc_test = e_acc_test
					)
				)
				logfile.flush()
			#end
		#end
	#end
#end
