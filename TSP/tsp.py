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
from tsp_utils import InstanceLoader, create_dataset_metric, to_quiver

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

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
			# M is a E×V adjacency matrix connecting each edge to the vertices it is connected to
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
	route_cost = tf.placeholder( tf.float32, [ None ], name = "route_cost" )

	# Placeholders for the list of number of vertices and edges per instance
	n_vertices 	= tf.placeholder( tf.int32, shape = (None,), name = "n_vertices" )
	n_edges 	= tf.placeholder( tf.int32, shape = (None,), name = "edges" )

	# Compute the number of vertices
	n = tf.shape( gnn.matrix_placeholders["M"] )[1]
	# Compute number of problems
	p = tf.shape( route_cost )[0]

	# Get the last embeddings
	E_n = gnn.last_states["E"].h
	E_vote = tf.reshape(E_vote_MLP(E_n), [-1])

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
	cost_loss = tf.losses.mean_squared_error(tf.reduce_sum(route_cost), tf.reduce_sum(cost_predictions_fuzzy))

	# Define cost accuracy, which is the deviation between the 'binary' route
	# cost computed from the binarized edge probabilities and the cost label
	cost_acc = tf.reduce_mean(
		tf.div(
			tf.subtract(cost_predictions_binary, route_cost),
			tf.add(route_cost, tf.constant(10**(-5)))
			)
		)

	
	# Count the number of edges that appear in the solution
	pos_edges_n = tf.reduce_sum(route_edges)
	# Count the number of edges that do not appear in the solution
	neg_edges_n = tf.reduce_sum(tf.subtract(tf.ones_like(route_edges), route_edges))
	# Compute edges loss
	edges_loss = tf.losses.sigmoid_cross_entropy(
		multi_class_labels	= route_edges,
		logits 				= E_vote,
		weights 			= tf.add(
			tf.scalar_mul(
				tf.divide(tf.add(pos_edges_n,neg_edges_n),pos_edges_n),
				route_edges),
			tf.scalar_mul(
				tf.divide(tf.add(pos_edges_n,neg_edges_n),neg_edges_n),
				tf.subtract(tf.ones_like(route_edges), route_edges)
				)
			)
		)

	# Compute precision and recall in terms of selected edges
	# Precision = |{relevant items} ∩ {retrieved items}| / |{retrieved items}|
	# Recall 	= |{relevant items} ∩ {retrieved items}| / |{relevant items}|

	true_positives = tf.reduce_sum(
			tf.multiply(
				route_edges,
				tf.cast(
					tf.equal(
						route_edges,
						tf.round(E_prob)
						),
					tf.float32
					)
			)
		)

	true_negatives = tf.reduce_sum(
			tf.multiply(
				tf.subtract(tf.ones_like(route_edges),route_edges),
				tf.cast(
					tf.equal(
						route_edges,
						tf.round(E_prob)
						),
					tf.float32
					)
			)
		)

	false_positives = tf.reduce_sum(
			tf.multiply(
				tf.subtract(tf.ones_like(route_edges),route_edges),
				tf.cast(
					tf.not_equal(
						route_edges,
						tf.round(E_prob)
						),
					tf.float32
					)
			)
		)

	false_negatives = tf.reduce_sum(
			tf.multiply(
				route_edges,
				tf.cast(
					tf.not_equal(
						route_edges,
						tf.round(E_prob)
						),
					tf.float32
					)
			)
		)

	precision = tf.divide(
		true_positives,
		tf.add(true_positives,false_positives)
	)

	recall = tf.divide(
		true_positives,
		tf.add(true_positives,false_negatives)
	)

	true_negative_rate = tf.divide(
		true_negatives,
		tf.add(true_negatives,false_positives)
	)

	accuracy = tf.divide(
		tf.add(true_positives,true_negatives),
		tf.reduce_sum([true_positives,true_negatives,false_positives,false_negatives])
	)

	# top_k accuracy
	top_edges_acc = tf.reduce_mean(
		tf.cast(
			tf.equal(
				# Sum one-hot representations to obtain edges mask
				tf.reduce_sum(
					# Convert array of indices to array of one-hot representations
					tf.one_hot(
						# Get the indices of the 'n' edges with the higher probabilities (as given by E_prob)
						tf.nn.top_k(E_prob, k=n)[1],
						depth = tf.shape(route_edges)[0]
						)
					),
					route_edges
				),
			tf.float32
			)
		)
	
	vars_cost 		= tf.zeros([])
	tvars 			= tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for


	# Define train step for cost loss	
	optimizer 			= tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ 			= tf.clip_by_global_norm( tf.gradients( tf.add( cost_loss, tf.multiply( vars_cost, parameter_l2norm_scaling ) ), tvars ), global_norm_gradient_clipping_ratio )
	cost_train_step 	= optimizer.apply_gradients( zip( grads, tvars ) )

	# Define train step for edges loss	
	optimizer 			= tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ 			= tf.clip_by_global_norm( tf.gradients( tf.add( edges_loss, tf.multiply( vars_cost, parameter_l2norm_scaling ) ), tvars ), global_norm_gradient_clipping_ratio )
	edges_train_step 	= optimizer.apply_gradients( zip( grads, tvars ) )

	GNN["gnn"] 						= gnn
	GNN["n_vertices"]				= n_vertices
	GNN["n_edges"]					= n_edges
	GNN["route_cost"] 				= route_cost
	GNN["route_edges"]				= route_edges
	
	GNN["cost_predictions_fuzzy"] 	= cost_predictions_fuzzy
	GNN["cost_predictions_binary"] 	= cost_predictions_binary
	GNN["avg_cost_fuzzy"]			= tf.reduce_mean(cost_predictions_fuzzy)
	GNN["avg_cost_binary"]			= tf.reduce_mean(cost_predictions_binary)
	GNN["cost_loss"]				= cost_loss
	GNN["edges_loss"]				= edges_loss
	
	GNN["precision"]				= precision
	GNN["recall"]					= recall
	GNN["true_negative_rate"]		= true_negative_rate
	GNN["accuracy"]					= accuracy

	GNN["top_edges_acc"]			= top_edges_acc
	
	GNN["cost_train_step"] 			= cost_train_step
	GNN["edges_train_step"] 		= edges_train_step
	
	GNN["E_prob"]					= E_prob
	
	return GNN
#end

if __name__ == '__main__':
	
	create_datasets 	= False
	load_checkpoints	= False
	save_checkpoints	= True

	d 					= 128
	epochs 				= 100
	batch_size			= 32
	batches_per_epoch 	= 128
	time_steps 			= 32
	loss_type			= "edges"

	print('\n\n')

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
		print("Initializing global variables ... ")
		sess.run( tf.global_variables_initializer() )

		# Restore saved weights
		if load_checkpoints: load_weights(sess,"./TSP-checkpoints-{}".format(loss_type));

		with open("log-TSP-{}.dat".format(loss_type),"w") as logfile:
			# Run for a number of epochs
			print("Running for {} epochs\n".format(epochs))
			for epoch in range( epochs ):

				# Reset train loader because we are starting a new epoch
				train_loader.reset()

				train_loss 					= np.zeros(batches_per_epoch)
				train_precision             = np.zeros(batches_per_epoch)
				train_recall                = np.zeros(batches_per_epoch)
				train_true_negative_rate    = np.zeros(batches_per_epoch)
				train_accuracy              = np.zeros(batches_per_epoch)
				train_tacc 					= np.zeros(batches_per_epoch)

				# Run batches
				for (batch_i, batch) in islice(enumerate(train_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels for this batch
					Ma_all, Mw_all, n_vertices, n_edges, route_edges, route_cost = batch

					# Convert to quiver format
					M, W = to_quiver(Ma_all, Mw_all)

					# Run one SGD iteration
					_, train_loss[batch_i], train_precision[batch_i], train_recall[batch_i], train_true_negative_rate[batch_i], train_accuracy[batch_i], train_tacc[batch_i], e_prob = sess.run(
						[ GNN["{}_train_step".format(loss_type)], GNN["{}_loss".format(loss_type)], GNN["precision"], GNN["recall"], GNN["true_negative_rate"], GNN["accuracy"], GNN["top_edges_acc"], GNN["E_prob"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["M"]:	M,
							GNN["gnn"].matrix_placeholders["W"]:	W,
							GNN["n_vertices"]:						n_vertices,
							GNN["n_edges"]:							n_edges,
							GNN["gnn"].time_steps: 					time_steps,
							GNN["route_edges"]:						route_edges,
							GNN["route_cost"]: 						route_cost,
						}
					)

					print('Train Epoch {epoch}\tBatch {batch}\t(n,m,batch size):\t\t({n},{m},{batch_size})'.format(
						epoch = epoch,
						batch = batch_i,
						n = np.sum(n_vertices),
						m = np.sum(n_edges),
						batch_size = batch_size
						))
					print('Loss:\t\t\t\t\t\t\t{loss:.3f}'.format(
						loss = train_loss[batch_i]
						))
					print('(Precision, Recall, True Negative Rate, Accuracy):\t({precision:.3f}, {recall:.3f}, {true_negative_rate:.3f}, {accuracy:.3f})'.format(
						precision 			= train_precision[batch_i],
						recall 				= train_recall[batch_i],
						true_negative_rate	= train_true_negative_rate[batch_i],
						accuracy 			= train_accuracy[batch_i],
						))
					print('Top edges accuracy:\t\t\t\t\t{top_edges_acc:.3f}'.format(
						top_edges_acc = train_tacc[batch_i]
						))
					predicted_edges = np.nonzero(np.round(e_prob))[0]
					discarded_edges = np.nonzero(np.round(1-e_prob))[0]
					#print([ (w, np.round(len([e for e in discarded_edges if np.round(10*W[e]) == w])/len([e for e in range(e_prob.shape[0]) if np.round(10*W[e]) == w]),2) if len([e for e in range(e_prob.shape[0]) if np.round(10*W[e]) == w]) > 0 else 0 ) for w in range(10) ])
					print('')

					

				#end

				# Print train epoch summary
				print('Train Epoch {epoch} Averages')
				print('Loss:\t\t\t\t\t\t\t{loss}'.format(
					loss = np.mean(train_loss)
					))
				print('(Precision, Recall, True Negative Rate, Accuracy):\t({precision:.3f}, {recall:.3f}, {true_negative_rate:.3f}, {accuracy:.3f})'.format(
					precision 			= np.mean(train_precision),
					recall 				= np.mean(train_recall),
					true_negative_rate	= np.mean(train_true_negative_rate),
					accuracy 			= np.mean(train_accuracy),
					))
				print('Top edges accuracy:\t\t\t\t\t{top_edges_acc:.3f}'.format(
					top_edges_acc = np.mean(train_tacc)
					))
				print('')

				print("Testing...")
				
				# Reset test loader as we are starting a new epoch
				test_loader.reset()
				
				test_loss				= np.zeros(batches_per_epoch)
				test_precision			= np.zeros(batches_per_epoch)
				test_recall   			= np.zeros(batches_per_epoch)
				test_true_negative_rate	= np.zeros(batches_per_epoch)
				test_accuracy          	= np.zeros(batches_per_epoch)
				test_tacc 				= np.zeros(batches_per_epoch)

				# Run batches
				for (batch_i, batch) in islice(enumerate(test_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels for this batch
					Ma_all, Mw_all, n_vertices, n_edges, route_edges, route_cost = batch

					# Convert to quiver format
					M, W = to_quiver(Ma_all, Mw_all)

					test_loss[batch_i], test_precision[batch_i], test_recall[batch_i], test_true_negative_rate[batch_i], test_accuracy[batch_i], test_tacc[batch_i], e_prob = sess.run(
						[ GNN["{}_loss".format(loss_type)], GNN["precision"], GNN["recall"], GNN["true_negative_rate"], GNN["accuracy"], GNN["top_edges_acc"], GNN["E_prob"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["M"]:	M,
							GNN["gnn"].matrix_placeholders["W"]:	W,
							GNN["n_vertices"]:						n_vertices,
							GNN["n_edges"]:							n_edges,
							GNN["gnn"].time_steps: 					time_steps,
							GNN["route_edges"]:						route_edges,
							GNN["route_cost"]: 						route_cost,
						}
					)
				#end
				
				# Print test epoch summary
				print('Test Epoch {epoch} Averages'.format(
					epoch = epoch
					))
				print('Loss:\t\t\t\t\t\t\t{loss:.3f}'.format(
					loss = np.mean(test_loss)
					))
				print('(Precision, Recall, True Negative Rate, Accuracy):\t({precision:.3f}, {recall:.3f}, {true_negative_rate:.3f}, {accuracy:.3f})'.format(
					precision 			= np.mean(test_precision),
					recall 				= np.mean(test_recall),
					true_negative_rate	= np.mean(test_true_negative_rate),
					accuracy 			= np.mean(test_accuracy),
					))
				print('Top edges accuracy:\t\t\t\t\t{top_edges_acc:.3f}'.format(
					top_edges_acc = np.mean(test_tacc)
					))
				print('')

				# Save weights
				if save_checkpoints: save_weights(sess,"./TSP-checkpoints-{}".format(loss_type));

				print('--------------------------------------------------------------------\n')

				# Write train and test results into log file
				logfile.write("{epoch} {tr_loss} {tr_precision} {tr_recall} {tr_true_negative_rate} {tr_acc} {tr_tacc} {te_loss} {te_precision} {te_recall} {te_true_negative_rate} {te_acc} {te_tacc}\n".format(
					epoch 		= epoch,
					tr_loss 				= np.mean(train_loss),
					tr_precision 			= np.mean(train_precision),
					tr_recall 				= np.mean(train_recall),
					tr_true_negative_rate 	= np.mean(train_true_negative_rate),
					tr_acc 					= np.mean(train_accuracy),
					tr_tacc 				= np.mean(train_tacc),

					te_loss 				= np.mean(test_loss),
					te_precision 			= np.mean(test_precision),
					te_recall 				= np.mean(test_recall),
					te_true_negative_rate 	= np.mean(test_true_negative_rate),
					te_acc 					= np.mean(test_accuracy),
					te_tacc 				= np.mean(test_tacc)
					)
				)
				logfile.flush()
			#end
		#end
	#end
#end
