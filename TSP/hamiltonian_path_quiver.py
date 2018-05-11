import sys, os, time
import tensorflow as tf
import numpy as np
import random
from itertools import islice
#from joblib import Parallel, delayed
import multiprocessing
from ortools.constraint_solver import pywrapcp
from ortools.constraint_solver import routing_enums_pb2
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import model builder
from graphnn import GraphNN
from mlp import Mlp
from util import timestamp, memory_usage, dense_to_sparse

class InstanceLoader(object):

	def __init__(self,path):
		self.path = path

		self.filenames = [ path + '/' + x for x in os.listdir(path) ]
		self.reset()
	#end

	def get_instances(self, n_instances):
		for i in range(n_instances):
			yield read_graph(self.filenames[self.index])
			self.index += 1
		#end
	#end

	def create_batch(self,instances):
		n_vertices 	= np.array([ x[0].shape[0] for x in instances ])
		n_edges		= np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
		solution	= np.array([ x[2] for x in instances ])
		total_n 	= sum(n_vertices)

		Ma_all, Mw_all = np.zeros((total_n,total_n)), np.zeros((total_n,total_n))
		for (i,Ma,Mw) in [ (i,x[0],x[1]) for (i,x) in enumerate(instances) ]:
			n_acc = sum(n_vertices[0:i])
			Ma_all[n_acc:n_acc+n_vertices[i], n_acc:n_acc+n_vertices[i]] = Ma
			Mw_all[n_acc:n_acc+n_vertices[i], n_acc:n_acc+n_vertices[i]] = Mw
		#end

		return Ma_all, Mw_all, n_vertices, n_edges, solution
	#end

	def get_batches(self, batch_size):
		for i in range( len(self.filenames) // batch_size ):
			yield self.create_batch(list(self.get_instances(batch_size)))
		#end
	#end

	def reset(self):
		random.shuffle( self.filenames )
		self.index = 0
	#end
#end

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

def write_graph(Ma, Mw, solution, filepath):
	with open(filepath,"w") as out:
		# Write header 'p |V| |E| solution'
		out.write("p {} {} {}\n".format(Ma.shape[0], len(np.nonzero(Ma)[0]), solution))

		# Write edges
		for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
			out.write("{} {} {}\n".format(i,j,Mw[i,j]))
		#end
	#end
#end

def read_graph(filepath):
	with open(filepath,"r") as f:
		n, m, solution = [ int(x) for x in f.readline().split()[1:]]
		Ma = np.zeros((n,n),dtype=int)
		Mw = np.zeros((n,n),dtype=int)
		for edge in range(m):
			i,j,w = [ int(float(x)) for x in f.readline().split() ]
			Ma[i,j] = 1
			Mw[i,j] = w
		#end
	#end
	return Ma,Mw,solution
#end

def create_and_write(path,n,i):
	M1,M2 = create_graph_pair(n)
	print("Writing graph file n,m=({},{})".format(M1.shape[0], len(np.nonzero(M1)[0])))
	print("Writing graph file n,m=({},{})".format(M2.shape[0], len(np.nonzero(M2)[0])))
	write_graph(M1,np.zeros(M1.shape),0,"{}/{}.graph".format(path,2*i))
	write_graph(M2,np.zeros(M2.shape),1,"{}/{}.graph".format(path,2*i+1))
#end

def create_dataset(n, path, samples=1000):

	if not os.path.exists(path):
		os.makedirs(path)
	#end if

	num_cores = multiprocessing.cpu_count()

	for i in range(samples//2):
		create_and_write(path,n,i)
	#end
#end

def build_network(d):
	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define GNN dictionary
	GNN = {}

	gnn = GraphNN(
		{
			"N": d, # Nodes
			"E": d  # Edges
		},
		{
			"Ms": ("N","E"), # Matrix pointing from nodes to the edges they are sources
			"Mt": ("N","E"), # Matrix pointing from nodes to the edges they are targets 
			"Mw": ("E","E") # Matrix indicating an Edge weight
		},
		{
			"NsmsgE": ("N","E"), # Message cast to convert messages from node sources to edges
			"NtmsgE": ("N","E"), # Message cast to convert messages from node targets to edges
			"EmsgNs": ("N","E"), # Message cast to convert messages from edges to node sources
			"EmsgNt": ("N","E")  # Message cast to convert messages from edges to node targets
		},
		{
			"N": [
				{
					"mat": "Ms",
					"msg": "EmsgNs",
					"var": "E"
				},
				{
					"mat": "Mt",
					"msg": "EmsgNt",
					"var": "E"
				}
			],
			"E": [
				{
					"mat": "Ms",
					"transpose?": True,
					"msg": "NsmsgE",
					"var": "N"
				},
				{
					"mat": "Mt",
					"transpose?": True,
					"msg": "NtmsgE",
					"var": "N"
				},
				{
					"mat": "Mw"
				}
			]
		},
		name="Dijkstra_Quiver",
		float_dtype = tf.float32
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
	n = tf.shape( gnn.matrix_placeholders["Ms"] )[0]
	# Compute number of problems
	p = tf.shape( labels )[0]

	# Get the last embeddings
	V_n = gnn.last_states["N"].h
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

def load_weights(sess,path,scope=None):
	if os.path.exists(path):
		# Restore saved weights
		print( "{timestamp}\t{memory}\tRestoring saved model ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		# Create model saver
		if scope is None:
			saver = tf.train.Saver()
		else:
			saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
		#end
		saver.restore(sess, "%s/model.ckpt" % path)
	#end if
#end

def save_weights(sess,path,scope=None):
	# Create /tmp/ directory to save weights
	if not os.path.exists(path):
		os.makedirs(path)
	#end if
	# Create model saver
	if scope is None:
		saver = tf.train.Saver()
	else:
		saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope))
	#end
	saver.save(sess, "%s/model.ckpt" % path)
	print( "{timestamp}\t{memory}\tMODEL SAVED IN PATH: {path}".format( timestamp = timestamp(), memory = memory_usage(), path=path ) )
#end

def dense_to_quiver( Ma, Mw=None, undirected=True ):
	Ms_index = []
	Mt_index = []
	Mw_index = []
	Mw_value = []
	e = 0
	for i in range( Ma.shape[0] ):
		for j in range( Ma.shape[1] ):
			if Ma[i,j] != 0:
				Ms_index.append( (i, e) )
				Mt_index.append( (j, e) )
				if undirected:
					Ms_index.append( (j, e) )
					Mt_index.append( (i, e) )
				#end if
				Mw_index.append( (e, e) )
				Mw_value.append( Mw[i,j] if Mw is not None else 1 )
				e += 1
			#end if
		#end for
	#end for
	Ms = [ Ms_index, [1 for _ in Ms_index], (Ma.shape[0],e) ]
	Mt = [ Mt_index, [1 for _ in Mt_index], (Ma.shape[0],e) ]
	Mw = [ Mw_index, Mw_value, (e,e) ]
	return Ms, Mt, Mw
#end dense_to_quiver

if __name__ == '__main__':
	
	create_datasets 	= False
	load_checkpoints	= False
	save_checkpoints	= False

	d 					= 128
	epochs 				= 100
	batch_size			= 3#32
	batches_per_epoch 	= 2#128
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
		if load_checkpoints: load_weights(sess,"./hamiltonian-quiver-checkpoints");

		with open("log-Hamiltonian-quiver.dat","w") as logfile:
			# Run for a number of epochs
			print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
			for epoch in range( epochs ):

				# Reset train loader
				train_loader.reset()
				e_loss_train, e_acc_train, e_pred_train = 0, 0, 0
				for (batch_i, batch) in islice(enumerate(train_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels
					Ma_all, Mw_all, n_vertices, n_edges, solution = batch
					Ms, Mt, Mw = dense_to_quiver( Ma_all )

					# Run one SGD iteration
					_, loss, acc, pred = sess.run(
						[ GNN["train_step"], GNN["loss"], GNN["acc"], GNN["avg_pred"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["Ms"]:	Ms,
							GNN["gnn"].matrix_placeholders["Mt"]:	Mt,
							GNN["gnn"].matrix_placeholders["Mw"]:	Mw,
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
							batch = batch_i,
							loss = loss,
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

				if save_checkpoints: save_weights(sess,"./hamiltonian-quiver-checkpoints");

				# Reset test loader
				print("{timestamp}\t{memory}\tTesting...".format(timestamp=timestamp(), memory=memory_usage()))
				test_loader.reset()
				e_loss_test, e_acc_test, e_pred_test = 0, 0, 0
				for (batch_i, batch) in islice(enumerate(test_loader.get_batches(32)), batches_per_epoch):

					# Get features, problem sizes, labels
					Ma_all, Mw_all, n_vertices, n_edges, solution = batch
					Ms, Mt, Mw = dense_to_quiver( Ma_all )

					# Run one SGD iteration
					loss, acc, pred = sess.run(
						[ GNN["loss"], GNN["acc"], GNN["avg_pred"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["Ms"]:	Ms,
							GNN["gnn"].matrix_placeholders["Mt"]:	Mt,
							GNN["gnn"].matrix_placeholders["Mw"]:	Mw,
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
