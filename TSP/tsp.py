import sys, os, time
import tensorflow as tf
import numpy as np
import random
from itertools import islice
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import model builder
from graphnn import GraphNN
from mlp import Mlp
from util import timestamp, memory_usage, dense_to_sparse, load_weights, save_weights
from tsp_utils import InstanceLoader, write_graph, read_graph

def solve(Ma, W):
	"""
		Find the optimal TSP tour given vertex adjacencies given by the binary
		matrix Ma and edge weights given by the real-valued matrix W
	"""

	n = Ma.shape[0]

	# Create a routing model
	routing = pywrapcp.RoutingModel(n, 1, 0)

	def dist(i,j):
		return W[i,j]
	#end

	# Define edge weights
	routing.SetArcCostEvaluatorOfAllVehicles(dist)

	# Remove connections where Ma[i,j] = 0
	for i in range(n):
		for j in range(n):
			if Ma[i,j] == 0:
				routing.NextVar(i).RemoveValue(j)
			#end
		#end
	#end

	assignment = routing.Solve()

	if assignment is None:
		return None
	else:
		return assignment.ObjectiveValue()
	#end
#end

def create(n,d,max_dist):	
	Ma = (np.random.rand(n,n) < d).astype(int)
	W = (np.random.randint(1,max_dist,(n,n)))

	solution = solve(Ma,W)

	return Ma, W, 0 if solution is None else solution
#end

def create_dataset(n, path, max_dist=100, min_density=0.5, max_density=0.75, samples=1000):

	if not os.path.exists(path):
		os.makedirs(path)
	#end if

	for i,d in enumerate(np.linspace(min_density,max_density,samples)):
		Ma,W,solution = create(n,d,max_dist)
		print("Writing graph file n,m=({},{})".format(Ma.shape[0], len(np.nonzero(Ma)[0])))
		write_graph(Ma,W,solution,"{}/{}.graph".format(path,i))
	#end
#end

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
			# Msrc is a E×v adjacency matrix connecting each edge to its source vertex
			"Msrc": ("E","V"),
			# Mtgt is a E×v adjacency matrix connecting each edge to its target vertex
			"Mtgt": ("E","V"),
			# W is a diagonal matrix of shape |E|×|E| where W[i,i] is the weight of the i-th edge
			"W": ("E","E")
		},
		{
			# Vmsg computes messages from vertices to edges
			"Vmsg": ("V","E"),
			# Emsg computes messages from edges to vertices
			"Emsg": ("E","V")
		},
		{
			# V(t+1) ← Vu( Msrcᵀ × Emsg(E(t)), Mtgtᵀ × Emsg(E(t)) )
			"V": [
				{
					"mat": "Msrc",
					"transpose?": True,
					"var": "E"
				},
				{
					"mat": "Mtgt",
					"transpose?": True,
					"var": "E"
				}
			],
			# C(t+1) ← Cu( Msrc × Vmsg(V(t)), Mtgt × Vmsg(V(t)), W × ones(|E|))
			"E": [
				{
					"mat": "Msrc",
					"msg": "Vmsg",
					"var": "V"
				},
				{
					"mat": "Mtgt",
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

	# Define E_vote
	E_vote_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "E_vote",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
		)

	# Define placeholder for result values (one per problem)
	labels = tf.placeholder( tf.float32, [ None ], name = "labels" )

	# Placeholders for the list of number of vertices and edges per instance
	n_vertices 	= tf.placeholder( tf.int32, shape = (None,), name = "n_vertices" )
	n_edges 	= tf.placeholder( tf.int32, shape = (None,), name = "edges" )

	# Compute the number of variables
	n = tf.shape( gnn.matrix_placeholders["Msrc"] )[1]
	# Compute number of problems
	p = tf.shape( labels )[0]

	# Get the last embeddings
	E_n = gnn.last_states["E"].h
	E_vote = E_vote_MLP( E_n )

	# Compute a probability pᵢ ∈ [0,1] that each edge belongs to the TSP optimal route
	E_prob = tf.sigmoid(E_vote)

	# Compute a cost for each edge by multiplying each weight with the corresponding edge probability
	cost_per_edge = tf.sparse_tensor_dense_matmul(gnn.matrix_placeholders["W"],E_prob)

	# Reorganize votes' result to obtain a prediction for each problem instance
	def _vote_while_cond(i, n_acc, predictions):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, n_acc, predictions):
		
		# Gather the set of edge costs relative to the i-th problem
		costs_i = tf.gather(cost_per_edge, tf.range(n_acc, tf.add(n_acc, n_edges[i])))

		# The total TSP cost for this problem is the sum of all its costs
		problem_prediction = tf.reduce_sum(costs_i)

		# Update TensorArray
		predictions = predictions.write( i, problem_prediction )
		return tf.add(i, tf.constant(1)), tf.add(n_acc, n_edges[i]), predictions
	#end _vote_while_body
	
	# Obtain a list of predictions, one per problem
	predictions = tf.TensorArray( size = p, dtype = tf.float32 )
	_, _, predictions = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant(0), tf.constant(0), predictions ]
	)
	predictions = predictions.stack()

	# Define loss, optimizer, train step
	predict_costs 	= tf.losses.mean_squared_error(labels,predictions)
	predict_cost 	= tf.reduce_mean(predict_costs)
	vars_cost 		= tf.zeros([])
	tvars 			= tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	loss 		= tf.add( predict_cost, tf.multiply( vars_cost, parameter_l2norm_scaling ) )
	optimizer 	= tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ 	= tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
	train_step 	= optimizer.apply_gradients( zip( grads, tvars ) )

	# Define accuracy
	acc = tf.reduce_mean(tf.div(tf.subtract(predictions, labels), labels))
	modacc = tf.reduce_mean(tf.div(tf.abs(tf.subtract(predictions, labels)), labels))

	GNN["gnn"] 						= gnn
	GNN["n_vertices"]				= n_vertices
	GNN["n_edges"]					= n_edges
	GNN["labels"] 					= labels
	GNN["predictions"] 				= predictions
	GNN["avg_pred"]					= tf.reduce_mean(predictions)
	GNN["loss"] 					= loss
	GNN["acc"]						= acc
	GNN["modacc"] = modacc
	GNN["train_step"] 				= train_step
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
	time_steps 			= 40

	if create_datasets:
		samples = batch_size*batches_per_epoch
		print("Creating {} train instances...".format(samples))
		create_dataset(20, path="TSP-train", samples=samples)
		print("Creating {} test instances...".format(samples))
		create_dataset(20, path="TSP-test", samples=samples)
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
				e_loss_train, e_acc_train, e_modacc_train, e_pred_train = 0, 0, 0, 0
				for (batch_i, batch) in islice(enumerate(train_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels
					Ma_all, W_all, n_vertices, n_edges, solution = batch

					total_vertices 	= sum(n_vertices)
					total_edges		= sum(n_edges)

					Msrc 	= np.zeros((total_edges,total_vertices))
					Mtgt 	= np.zeros((total_edges,total_vertices))
					W 		= np.zeros((total_edges,total_edges))

					for (e,(i,j)) in enumerate(zip(list(np.nonzero(Ma_all)[0]), list(np.nonzero(Ma_all)[1]))):
						Msrc[e] = i
						Mtgt[e] = j
						W[e,e] = W_all[i,j]
					#end

					# Run one SGD iteration
					_, loss, acc, modacc, pred = sess.run(
						[ GNN["train_step"], GNN["loss"], GNN["acc"], GNN["modacc"], GNN["avg_pred"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["Msrc"]:	dense_to_sparse(Msrc),
							GNN["gnn"].matrix_placeholders["Mtgt"]:	dense_to_sparse(Mtgt),
							GNN["gnn"].matrix_placeholders["W"]:	dense_to_sparse(W),
							GNN["n_vertices"]:						n_vertices,
							GNN["n_edges"]:							n_edges,
							GNN["gnn"].time_steps: 					time_steps,
							GNN["labels"]: 							solution
						}
					)

					e_loss_train += loss
					e_acc_train += acc
					e_modacc_train += modacc
					e_pred_train += pred

					# Print batch summary
					print(
						"{timestamp}\t{memory}\tTrain Epoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| (Loss,%Err|%Err|,Avg.Pred): ({loss:.5f},{acc:.5f}|{modacc:.5f}|,{pred:.3f})".format(
							timestamp = timestamp(),
							memory = memory_usage(),
							epoch = epoch,
							batch = batch_i,
							loss = loss,
							acc = acc,
							modacc = modacc,
							pred = pred,
							n = total_vertices,
							m = total_edges,
							i = batch_size
						),
						flush = True
					)
				#end
				e_loss_train /= batches_per_epoch
				e_acc_train /= batches_per_epoch
				e_modacc_train /= batches_per_epoch
				e_pred_train /= batches_per_epoch
				# Print train epoch summary
				print(
					"{timestamp}\t{memory}\tTrain Epoch {epoch}\tMain (Loss,%Err|%Err|,Avg.Pred): ({loss:.5f},{acc:.5f}|{modacc:.5f}|,{pred:.3f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						loss = e_loss_train,
						acc = e_acc_train,
						modacc = e_modacc_train,
						pred = e_pred_train
					),
					flush = True
				)

				if save_checkpoints: save_weights(sess,"./TSP-checkpoints");

				# Reset test loader
				print("{timestamp}\t{memory}\tTesting...".format(timestamp=timestamp(), memory=memory_usage()))
				test_loader.reset()
				e_loss_test, e_acc_test, e_modacc_test, e_pred_test = 0, 0, 0, 0
				for (batch_i, batch) in islice(enumerate(test_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels
					Ma_all, W_all, n_vertices, n_edges, solution = batch

					total_vertices 	= sum(n_vertices)
					total_edges		= sum(n_edges)

					Msrc 	= np.zeros((total_edges,total_vertices))
					Mtgt 	= np.zeros((total_edges,total_vertices))
					W 		= np.zeros((total_edges,total_edges))

					for (e,(i,j)) in enumerate(zip(list(np.nonzero(Ma_all)[0]), list(np.nonzero(Ma_all)[1]))):
						Msrc[e] = i
						Mtgt[e] = j
						W[e,e] = W_all[i,j]
					#end

					# Run one SGD iteration
					loss, acc, modacc, pred = sess.run(
						[ GNN["loss"], GNN["acc"], GNN["modacc"], GNN["avg_pred"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["Msrc"]:	dense_to_sparse(Msrc),
							GNN["gnn"].matrix_placeholders["Mtgt"]:	dense_to_sparse(Mtgt),
							GNN["gnn"].matrix_placeholders["W"]:	dense_to_sparse(W),
							GNN["n_vertices"]: n_vertices,
							GNN["n_edges"]: n_edges,
							GNN["gnn"].time_steps: time_steps,
							GNN["labels"]: solution
						}
					)

					e_loss_test += loss
					e_acc_test += acc
					e_modacc_test += modacc
					e_pred_test += pred
				#end
				e_loss_test /= batches_per_epoch
				e_acc_test /= batches_per_epoch
				e_modacc_test /= batches_per_epoch
				e_pred_test /= batches_per_epoch
				# Print test epoch summary
				print(
					"{timestamp}\t{memory}\tTest Epoch {epoch}\tMain (Loss,%Err|%Err|,Avg.Pred): ({loss:.5f},{acc:.5f}|{modacc:.5f}|,{pred:.3f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						loss = e_loss_test,
						acc = e_acc_test,
						modacc = e_modacc_test,
						pred = e_pred_test
					),
					flush = True
				)

				# Write train and test results into log file
				logfile.write("{epoch} {loss_train} {acc_train} {modacc_train} {loss_test} {acc_test} {modacc_test}\n".format(
					epoch = epoch,
					loss_train = e_loss_train,
					acc_train = e_acc_train,
					modacc_train = e_modacc_train,
					loss_test = e_loss_test,
					acc_test = e_acc_test,
					modacc_test = e_modacc_test
					)
				)
				logfile.flush()
			#end
		#end
	#end
#end
