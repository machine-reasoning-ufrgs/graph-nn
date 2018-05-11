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

def solve(M):
	"""
		Solve a Hamiltonian path instance given a binary adjacency matrix M
	"""

	n = M.shape[0]

	# Create a routing model
	routing = pywrapcp.RoutingModel(n, 1, 0)

	# Remove connections where M[i,j] = 0
	for i in range(n):
		for j in range(n):
			if M[i,j] == 0:
				routing.NextVar(i).RemoveValue(j)
			#end
		#end
	#end

	return routing.Solve() is not None
#end

def create_graph_pair(n):
	"""
		Starts with a disjoint set of n vertices and iteratively adds edges
		while the resulting graph does not admit a Hamiltonian cycle. The
		penultimate and last of such graphs differ by only one edge, but admit
		and do not admit a Hamiltonian path respectively.
	"""
	M1 = np.zeros((n,n), dtype=int)
	M2 = np.zeros((n,n), dtype=int)

	hamiltonian_path = False
	while not hamiltonian_path:
		# Choose a random edge and add it to M2
		i,j = random.choice([ (i,j) for i in range(n) for j in range(n) if i != j and M2[i,j]==0 ])
		M2[i,j] = M2[j,i] = 1

		# Check if there is a Hamiltonian path in M2
		hamiltonian_path = solve(M2)

		if not hamiltonian_path:
			M1[i,j] = M1[j,i] = 1
		#end
	#end

	return M1,M2
#end

def create_dataset(n, path, samples=1000):

	if not os.path.exists(path):
		os.makedirs(path)
	#end if

	for i in range(samples//2):
		M1,M2 = create_graph_pair(n)
		print("Writing graph file n,m=({},{})".format(M1.shape[0], len(np.nonzero(M1)[0])))
		print("Writing graph file n,m=({},{})".format(M2.shape[0], len(np.nonzero(M2)[0])))
		write_graph(M1,np.zeros(M1.shape),0,"{}/{}.graph".format(path,2*i))
		write_graph(M2,np.zeros(M2.shape),1,"{}/{}.graph".format(path,2*i+1))
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
			"V": d
		},
		{
			# M is a V×V adjacency matrix between vertices
			"M": ("V","V"),
		},
		{
			# Don't need to send messages because there is only one type of
			# embedding
		},
		{
			# V(t+1) ← Vu( M × V(t) )
			"V": [
				{
					"mat": "M",
					"var": "V"
				}
			]
		},
		name="TSP"
	)

	# Define V_vote
	V_vote_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "N_vote",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
		)

	# Define placeholder for result values (one per problem)
	labels = tf.placeholder( tf.float32, [ None ], name = "labels" )

	# Placeholder for the list of number of vertices per instance
	n_vertices = tf.placeholder( tf.int32, shape = (None,), name = "n_vertices" )

	# Placeholder for the list of number of vertices per instance
	n_edges = tf.placeholder( tf.int32, shape = (None,), name = "n_edges" )

	# Compute the number of variables
	n = tf.shape( gnn.matrix_placeholders["M"] )[0]
	# Compute number of problems
	p = tf.shape( labels )[0]

	# Get the last embeddings
	V_n = gnn.last_states["V"].h
	V_vote = V_vote_MLP( V_n )

	# Reorganize votes' result to obtain a prediction for each problem instance
	def _vote_while_cond(i, n_acc, predictions):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, n_acc, predictions):
		# Gather the set of vertex votes relative to the i-th problem
		votes_i = tf.gather(V_vote, tf.range(n_acc, tf.add(n_acc, n_vertices[i])))
		problem_prediction = tf.reduce_mean(votes_i)
		# Update TensorArray
		predictions = predictions.write( i, problem_prediction )
		return tf.add(i, tf.constant(1)), tf.add(n_acc, n_vertices[i]), predictions
	#end _vote_while_body
			
	predictions = tf.TensorArray( size = p, dtype = tf.float32 )
	_, _, predictions = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant(0), tf.constant(0), predictions ]
	)
	predictions = predictions.stack()

	# Define loss, optimizer, train step
	predict_costs 	= tf.nn.sigmoid_cross_entropy_with_logits( labels = labels, logits = predictions )
	predict_cost 	= tf.reduce_mean( predict_costs )
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
	acc = tf.reduce_mean(
			tf.cast(
				tf.equal(
					tf.cast(labels, tf.bool),
					tf.cast(tf.round(tf.nn.sigmoid(predictions)), tf.bool)
				)
				, tf.float32
			)
		)

	GNN["gnn"] 						= gnn
	GNN["n_vertices"]				= n_vertices
	GNN["n_edges"]					= n_edges
	GNN["labels"] 					= labels
	GNN["predictions"] 				= predictions
	GNN["avg_pred"]					= tf.reduce_mean(tf.round(tf.sigmoid(predictions)))
	GNN["loss"] 					= loss
	GNN["acc"]						= acc
	GNN["train_step"] 				= train_step
	return GNN
#end

if __name__ == '__main__':
	
	create_datasets 	= True
	load_checkpoints	= False
	save_checkpoints	= True

	d 					= 256
	epochs 				= 1000
	batch_size			= 32
	batches_per_epoch 	= 128
	time_steps 			= 40

	if create_datasets:
		samples = batch_size*batches_per_epoch
		print("Creating {} train instances...".format(samples))
		create_dataset(20, path="Hamiltonian-train", samples=samples)
		print("Creating {} test instances...".format(samples))
		create_dataset(20, path="Hamiltonian-test", samples=samples)
	#end

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(d)

	# Create train, test loaders
	train_loader 	= InstanceLoader("Hamiltonian-train")
	test_loader		= InstanceLoader("Hamiltonian-test")

	# Disallow GPU use
	config = tf.ConfigProto( device_count = {"GPU":0})
	with tf.Session(config=config) as sess:
		
		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )

		# Restore saved weights
		if load_checkpoints: load_weights(sess,"./hamiltonian-checkpoints");

		with open("log-Hamiltonian.dat","w") as logfile:
			# Run for a number of epochs
			print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
			for epoch in range( epochs ):

				# Reset train loader
				train_loader.reset()
				e_loss_train, e_acc_train, e_pred_train = 0, 0, 0
				for (batch_i, batch) in islice(enumerate(train_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels
					Ma_all, Mw_all, n_vertices, n_edges, solution = batch

					# Run one SGD iteration
					_, loss, acc, pred = sess.run(
						[ GNN["train_step"], GNN["loss"], GNN["acc"], GNN["avg_pred"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["M"]:	dense_to_sparse(Ma_all),
							GNN["n_vertices"]:						n_vertices,
							GNN["n_edges"]:							n_edges,
							GNN["gnn"].time_steps: 					time_steps,
							GNN["labels"]: 							solution
						}
					)

					e_loss_train += loss
					e_acc_train += acc
					e_pred_train += pred

					# Print batch summary
					print(
						"{timestamp}\t{memory}\tTrain Epoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| (Loss,Acc,Avg.Pred): ({loss:.5f},{acc:.5f},{pred:.3f})".format(
							timestamp = timestamp(),
							memory = memory_usage(),
							epoch = epoch,
							loss = loss,
							batch_i = batch_i,
							acc = acc,
							pred = pred,
							n = Ma_all.shape[0],
							m = Ma_all.shape[1],
							i = batch_size
						),
						flush = True
					)
				#end
				e_loss_train /= batches_per_epoch
				e_acc_train /= batches_per_epoch
				e_pred_train /= batches_per_epoch
				# Print train epoch summary
				print(
					"{timestamp}\t{memory}\tTrain Epoch {epoch}\tMain (Loss,Acc,Avg.Pred): ({loss:.5f},{acc:.5f},{pred:.3f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						loss = e_loss_train,
						acc = e_acc_train,
						pred = e_pred_train
					),
					flush = True
				)

				if save_checkpoints: save_weights(sess,"./hamiltonian-checkpoints");

				# Reset test loader
				print("{timestamp}\t{memory}\tTesting...".format(timestamp=timestamp(), memory=memory_usage()))
				test_loader.reset()
				e_loss_test, e_acc_test, e_pred_test = 0, 0, 0
				for (batch_i, batch) in islice(enumerate(test_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels
					Ma_all, Mw_all, n_vertices, n_edges, solution = batch

					# Run one SGD iteration
					loss, acc, pred = sess.run(
						[ GNN["loss"], GNN["acc"], GNN["avg_pred"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["M"]:	dense_to_sparse(Ma_all),
							GNN["n_vertices"]:						n_vertices,
							GNN["n_edges"]:							n_edges,
							GNN["gnn"].time_steps: 					time_steps,
							GNN["labels"]: 							solution
						}
					)

					e_loss_test += loss
					e_acc_test += acc
					e_pred_test += pred
				#end
				e_loss_test /= batches_per_epoch
				e_acc_test /= batches_per_epoch
				e_pred_test /= batches_per_epoch
				# Print test epoch summary
				print(
					"{timestamp}\t{memory}\tTest Epoch {epoch}\tMain (Loss,Acc): ({loss:.5f},{acc:.5f},{pred:.3f})".format(
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
					acc_test = e_acc_test))
				logfile.flush()
			#end
		#end
	#end
#end
