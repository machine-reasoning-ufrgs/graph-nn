import sys, os, time
import tensorflow as tf
import numpy as np
import networkx as nx
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import model builder
from graphnn_refactored import GraphNN
from mlp import Mlp
# Import tools
#import itertools
from util import timestamp, memory_usage, sparse_to_dense, save_weights#, percent_error

# 32-bit floating point number mantissa limit for a number that is close to zero, but still can be added up to 1. (Not the real value, but a decimal approximation )
FLOAT32_MANTISSA_LIMIT = 0.00000001
# Alternatives:
# ALT 1
#FLOAT32_MANTISSA_LENGTH = 20 # Actually 24, but set to 20 to allow ignoring some small numbers.
# ALT 2
#FLOAT32_MANTISSA_LIMIT = 1
#for _ in range( MANTISSA_SIZE ): FLOAT32_MANTISSA_LIMIT /= 2;
def crop_to_float32_mantissa_limit(n):
	# Crops a float32 number under the mantissa limit to zero.
	return n if abs(n) > FLOAT32_MANTISSA_LIMIT else 0
#end crop_to_float32_mantissa_limit

def percent_error(labels,predictions,dtype=tf.float32):
	# Calculates percent error, avoiding division-by-zero
	# Replace any zeroes with ones
	labels_div = tf.where( tf.equal( labels, 0 ), tf.ones_like( labels, dtype=dtype ), labels )
	# Get % error
	return tf.reduce_mean( tf.divide( tf.abs( tf.subtract( labels, predictions ) ), labels_div ) )
#end percent_error

def percent_error_prob(labels,predictions,dtype=tf.float32):
	# Calculates percent error, avoiding division-by-zero
	# Replace any zeroes with ones
	labels_div = tf.where( tf.equal( labels, 0 ), tf.ones_like( labels, dtype=dtype ), labels )
	
	# if prediction > 0.5, it is 1. 0 otherwise
	predictions_mod = tf.where( tf.greater(predictions, 0.5), tf.ones_like( predictions, dtype=dtype ),tf.zeros_like (predictions, dtype = dtype) ) 
	# Get % error
	return tf.reduce_mean( tf.divide( tf.abs( tf.subtract( labels, predictions_mod ) ), labels_div ) )

def calc_degree( G, T, g_n ):
	# Calculates the degree centrality, non-normalized
	degree = crop_to_float32_mantissa_limit( nx.degree_centrality( G )[T] )
	degree = degree * ( g_n - 1 )
	return degree
#end calc_degree

def calc_degree_dict( G, g_n):
	# Calculates the degree centrality for all nodes (returns dict node->degree), non-normalized
	degree = nx.degree_centrality( G )
	for k,v in degree.items():
		degree[k] = degree[k] * ( g_n - 1 )
	return degree
#end calc_degree_dict

def build_network(d):
	# Builds the model
	
	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define GNN dictionary
	GNN = {}

	# Define placeholder for result values (one per problem)
	#instance_prob_matrix_degree = tf.placeholder( tf.float32, [ None, None ], name = "instance_prob_matrix_degree" )
	labels = tf.placeholder( tf.float32, [ None, None ], name = "labels" )
	nodes_n = tf.placeholder( tf.int32, [ None ], name = "nodes_n" )

	# Define Graph neural network
	gnn = GraphNN(
		{
			"N": d
		},
		{
			"M": ("N","N")
		},
		{
			"Nsource": ("N","N"),
			"Ntarget": ("N","N")
		},
		{
			"N": [
				{
					"mat": "M",
					"var": "N",
					"msg": "Nsource"
				},
				{
					"mat": "M",
					"transpose?": True,
					"var": "N",
					"msg": "Ntarget"
				}
			]
		},
		#Cell_activation = tf.nn.sigmoid,
		#Msg_last_activation = tf.nn.sigmoid,
		#name="Centrality",
	)

	# Define votes
#	prep_MLP = Mlp(
#		layer_sizes = [ d for _ in range(2) ],
#		activations = [ tf.nn.relu for _ in range(2) ],
#		output_size = d,
#		name = "prep_MLP",
#		name_internal_layers = True,
#		kernel_initializer = tf.contrib.layers.xavier_initializer(),
#		bias_initializer = tf.zeros_initializer()
#	)
	
	comp_MLP = Mlp(
		layer_sizes = [ 2*d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "comp_MLP",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
	)
	

	# Compute the number of variables
	n = tf.shape( gnn.matrix_placeholders["M"] )[0]
	# Compute number of problems
	p = tf.shape( nodes_n )[0]

	# Get the last embeddings
	N_n = gnn.last_states["N"].h
	#print("N_n shape:" +str(N_n.shape))
	
#	M_prob_exp = tf.expand_dims(N_n, 0)
#	M_1 = tf.tile( M_prob_exp, (n,1,1))
#	
#	M_prob_exp_trans = tf.transpose( M_prob_exp, (1,0,2))
#	M_2 = tf.tile( M_prob_exp_trans, (1, n, 1))
#	
#	M1M2 = tf.concat([M_1, M_2], 2)
#	
#	predicted_matrix = comp_MLP( M1M2 )
#	predicted_matrix = tf.squeeze( predicted_matrix )  

	# Reorganize votes' result to obtain a prediction for each problem instance
	def _vote_while_cond(i, acc_arr, cost_arr, n_acc):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, acc_arr, cost_arr, n_acc):
		# Gather the embeddings for that problem
		#p_embeddings = tf.gather(N_n, tf.range( n_acc, tf.add(n_acc, nodes_n[i]) ))
		p_embeddings = tf.slice( N_n, [n_acc, 0], [nodes_n[i], d]) 
		
		N_expanded = tf.expand_dims(p_embeddings, 0)
		N1 = tf.tile(N_expanded,(nodes_n[i],1,1))
		
		N_transposed = tf.transpose(N_expanded, (1,0,2))
		N2 = tf.tile(N_transposed, (1,nodes_n[i],1))
		
		N1N2 = tf.concat([N1,N2], 2)
		 
		prob_matrix = comp_MLP( N1N2 )
		
		problem_predicted_matrix = tf.squeeze( prob_matrix )
		
		
		#Gather matrix containing all the labels for a given problem
#		p_labels = tf.gather(
#			tf.gather(
#				labels, tf.range(  n_acc, tf.add(n_acc, nodes_n[i])),
#				axis=1
#			),
#			tf.range(  n_acc, tf.add(n_acc, nodes_n[i]) ),
#			axis=0
#		)

		p_labels = tf.slice( labels, [n_acc, n_acc], [nodes_n[i], nodes_n[i]])
		
		#Compare labels to predicted values
		#p_error = p_labels[n_acc:n_acc+nodes_n[i],n_acc:n_acc+nodes_n[i]]#
		p_acc = tf.reduce_mean(
			tf.cast(
				tf.equal(
					tf.round(tf.sigmoid(problem_predicted_matrix)),
																																																																																																																																																																																																																																																																																																																																																																																																			
				),
				tf.float32
			)
		)
		
		#Calculate cost for this problem
		p_cost = tf.losses.sigmoid_cross_entropy( multi_class_labels = p_labels, logits = problem_predicted_matrix)
			
		# Update TensorArray
		acc_arr = acc_arr.write( i, p_acc )
		cost_arr = cost_arr.write(i, p_cost)
		
		return tf.add( i, tf.constant( 1 ) ), acc_arr, cost_arr,tf.add( n_acc, nodes_n[i] )
	#end _vote_while_body
			
	acc_arr = tf.TensorArray( size = p, dtype = tf.float32 )
	cost_arr = tf.TensorArray( size = p, dtype = tf.float32 )
	
	_, acc_arr, cost_arr, _ = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), acc_arr, cost_arr, tf.constant( 0, dtype = tf.int32 ) ]
	)
	acc_arr = acc_arr.stack()
	cost_arr = cost_arr.stack()
	
	

	# Define loss, %error
	prob_degree_predict_cost = tf.reduce_mean( cost_arr ) 
	
	
	vars_cost = tf.zeros([])
	tvars = tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	loss = tf.add_n( [ prob_degree_predict_cost, tf.multiply( vars_cost, parameter_l2norm_scaling ) ] )
	optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
	train_step = optimizer.apply_gradients( zip( grads, tvars ) ) #optimizer.minimize(loss) #
	
	
	prob_degree_predict_acc = tf.reduce_mean( acc_arr ) #percent_error_prob( labels = instance_prob_degree, predictions = predicted_prob )
	
	GNN["gnn"] = gnn
	GNN["labels"] = labels
	#GNN["predicted_matrix"] = predicted_matrix
	GNN["prob_degree_predict_cost"] = prob_degree_predict_cost
	GNN["prob_degree_predict_acc"] = prob_degree_predict_acc
	GNN["loss"] = loss
	GNN["nodes_n"] = nodes_n
	GNN["train_step"] = train_step
	return GNN
#end build_network

def create_graph( g_n, erdos_renyi_p = 0.25, powerlaw_gamma = 3, smallworld_k = 4, smallworld_p = 0.25, powerlaw_cluster_m = 3, powerlaw_cluster_p = 0.1 ):
	# Tries to create a graph until all of its centralities can be computed
	Gs = None
	while Gs is None:
		Gs = _create_graph( g_n, erdos_renyi_p, powerlaw_gamma, smallworld_k, smallworld_p, powerlaw_cluster_m, powerlaw_cluster_p )
	#end while
	return Gs
#end create_graph

def build_M_from_graph( G, undirected = True, key = None ):
	# Build the sparse tensor matrices from a graph
	getval = lambda a: 1 if key is None else a[key]
	M_index = []
	M_values = []
	for s, t in G.edges:
		M_index.append( ( s, t ) )
		M_values.append( getval( G[s][t] ) )
		if undirected:
			M_index.append( ( t, s ) )
			M_values.append( getval( G[s][t] ) )
		#end if
	#end for
	return M_index, M_values
#end build_M_from_graph

def build_Mprobs_from_graph( matrix_probs, g_n ):
	Mprobs_index = []
	Mprobs_values = []
	for row in range(0, g_n-1):
		for col in range(row+1, g_n):
			if matrix_probs[row][col] == 1:
				Mprobs_index.append( (row, col) )
			else:
				Mprobs_index.append( (col, row) )
			Mprobs_values.append(1)
			#end if
		#end for
	#end for
	return Mprobs_index, Mprobs_values
#end build_Mprobs_from_graph

def _create_graph( g_n, erdos_renyi_p = 0.25, powerlaw_gamma = 3, smallworld_k = 4, smallworld_p = 0.25, powerlaw_cluster_m = 3, powerlaw_cluster_p = 0.1 ):
	# Tries to create a graph and returns None if it fails at any point.
	# Create a graph from a random distribution
	graph_type = np.random.randint( 0, 4 )
	if graph_type == 0:
		G = nx.fast_gnp_random_graph( g_n, erdos_renyi_p )
	elif graph_type == 1:
		try:
			G = nx.random_graphs.random_powerlaw_tree( g_n, powerlaw_gamma )
		except nx.NetworkXError as e:
			print( e, file = sys.stderr, flush = True )
			return None
		#end try
	elif graph_type == 2:
		try:
			G = nx.random_graphs.connected_watts_strogatz_graph( g_n, smallworld_k, smallworld_p )
		except nx.NetworkXError as e:
			print( e, file = sys.stderr, flush = True )
			return None
		#end try
	elif graph_type == 3:
		try:
			G = nx.random_graphs.powerlaw_cluster_graph( g_n, powerlaw_cluster_m, powerlaw_cluster_p )
		except nx.NetworkXError as e:
			print( e, file = sys.stderr, flush = True )
			return None
		#end try
	#end if
	for s, t in G.edges:
		G[s][t]["weight"] = 1#np.random.rand()
	#end for
	
	#Compute the matrix representation of the ranking
	degree_dict = calc_degree_dict( G, g_n)
	matrix_probs = np.zeros((g_n,g_n))
	for row in range(0, g_n-1):
		for col in range(row+1, g_n):
			matrix_probs[row][col] = 1 if( degree_dict[row] > degree_dict[col]) else 0
			matrix_probs[col][row] = 1 if( degree_dict[col] > degree_dict[row]) else 0
	
	
	
	# Build sparse matrices
	# TODO: Get values from the edges
	M_index, M_values = build_M_from_graph( G, key = "weight" )
	M = [M_index,M_values,(g_n,g_n)]
	Mprobs_index, Mprobs_values = build_Mprobs_from_graph( matrix_probs, g_n )
	Mprobs = [Mprobs_index, Mprobs_values, (g_n,g_n)]
	#M = sparse_to_dense( M )
	return M,Mprobs
#end _create_graph

def reindex_matrix( n, m, M ):
	# Reindex a sparse matrix
	new_index = []
	new_value = []
	for i, v in zip( M[0], M[1] ):
		s, t = i
		new_index.append( (n + s, m + t) )
		new_value.append( v )
	#end for
	return zip( new_index, new_value )

def create_batch(problems):
	# Create a problem-batch from the problem list passed
	nodesPerProblem = np.zeros(len(problems))
	count = 0
	n = 0
	m = 0
	batch_Madj_index = []
	batch_Madj_value = []
	j = 0
	k = 0
	batch_Mprobs_index =[]
	batch_Mprobs_value = []
	for p in problems:
		if p is None:
			continue
		#end if
		M_adj, M_probs = p
		nodesPerProblem[count] = (M_adj[2])[0]
		# Reindex the matrix to the new indexes
		for i, v in reindex_matrix( n, n, M_adj ):
			batch_Madj_index.append( i )
			batch_Madj_value.append( v )
		#end for
		n += M_adj[2][0]
		m += len(M_adj[0])
		for i, v in reindex_matrix( j, j, M_probs ):
			batch_Mprobs_index.append( i )
			batch_Mprobs_value.append( v )
		#end for
		j += M_probs[2][0]
		k += len(M_probs[0])
		count += 1
	#end for
	Madj = [batch_Madj_index,batch_Madj_value,(n,n)]
	Mprobs = [batch_Mprobs_index,batch_Mprobs_value,(j,j)]
	#print( "shape targets: {shape}".format( shape = np.shape(targets) ) )
	return Madj,Mprobs,nodesPerProblem #, targets_matrix
#end create_batch


#def create_batch(problems):
#	n = np.zeros(len(problems))
#	n_total = np.sum([ M.shape[0] for M,_ in problems ])
#	batch_M = np.zeros((n_total,n_total))
#	batch_labels = np.zeros((n_total,n_total))
#	for (i,problem) in enumerate(problems):
#		M, labels = problem
#		n[i] = M.shape[0]
#		#print(sum(n[0:i]))
#		#print(sum(n[0:i]) + n[i])
#		batch_M[ int(sum(n[0:i])) : int(sum(n[0:i]) + n[i]), int(sum(n[0:i])) : int(sum(n[0:i]) + n[i]) ] = M.copy()
#		batch_labels[ int(sum(n[0:i])) : int(sum(n[0:i]) + n[i]), int(sum(n[0:i])) : int(sum(n[0:i]) + n[i]) ] = labels.copy()
#	#end
#	return batch_M, batch_labels, n
##end

if __name__ == '__main__':
	embedding_size = 64
	epochs = 100
	batch_n_max = 4096
	batches_per_epoch = 32
	n_size_min = 16
	n_size_max = 512
	edge_probability = 0.25
	time_steps = 32

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(embedding_size)

	# Disallow GPU use
	config = tf.ConfigProto(
		device_count = {"GPU":0},
		inter_op_parallelism_threads=1,
		intra_op_parallelism_threads=1
	)
	with tf.Session(config=config) as sess:
		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )

		# Run for a number of epochs
		print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
		for epoch in range( epochs ):
			# Run batches
			#instance_generator.reset()
			epoch_loss = 0.0
			epoch_degc = 0
			epoch_degacc = 0
			epoch_n = 0
			epoch_m = 0
			for batch_i in range( batches_per_epoch ):
				# Create random graphs
				batch_n_size = np.random.randint( n_size_min, n_size_max+1 )
				n_acc = 0
				max_n = 0
				instances = 0
				batch = []
				#degree_vals = []
				#prob_matrixdegree_vals = []
				# Generate graphs until the batch can't fit another one
				while True:
					g_n = np.random.randint( batch_n_size//2, batch_n_size*2 )
					if n_acc + g_n < batch_n_max:
						n_acc += g_n
						instances += 1
						max_n = max( max_n, g_n )
						M_adj,M_probs = create_graph( g_n )
						batch.append( (M_adj, M_probs) )
						#prob_matrix_degree_vals.append( matrix_probs )
					else:
						break
					#end if
				#end while
				# Create the reindexed batch matrix and target-list
				M, labels, nodesPerProblem = create_batch( batch )
				n = M[2][0]
				m = len( M[0] )
				#M_sparse = sparse_to_dense(M)
				#labels_sparse = sparse_to_dense(labels)
#				print(M_sparse.shape)
#				print(labels_sparse.shape)
#				print(nodesPerProblem)
				_, loss, degc, degacc = sess.run(
					[ GNN["train_step"], GNN["loss"], GNN["prob_degree_predict_cost"],  GNN["prob_degree_predict_acc"] ],
					feed_dict = {
						GNN["nodes_n"]: nodesPerProblem,
						GNN["labels"]: sparse_to_dense(labels),
						GNN["gnn"].time_steps: time_steps,
						GNN["gnn"].matrix_placeholders["M"]: sparse_to_dense(M)
					}
				)
				
				epoch_loss += loss
				epoch_degc += degc
				epoch_degacc += degacc
				epoch_n += n
				epoch_m += m
				
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,i): ({n},{m},{i})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f}) Acc(D:{degree_acc:.5f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = batch_i,
						loss = loss,
						degree_cost = degc,
						degree_acc = degacc,
						n = n,
						m = m,
						i = instances
					),
					flush = True
				)
			#end for
			# Summarize Epoch
			epoch_loss /= batches_per_epoch
			epoch_degc /= batches_per_epoch
			epoch_degacc /= batches_per_epoch
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f}) Error(D:{degree_acc:.5f}))".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "all",
					loss = epoch_loss,
					degree_cost = epoch_degc,
					degree_acc = epoch_degacc,
					n = epoch_n,
					m = epoch_m,
				),
				flush = True
			)
			# TEST TIME
			# Create random graphs
			n_acc = 0
			max_n = 0
			instances = 0
			batch = []
			#degree_vals = []
			while True:
				g_n = np.random.randint( n_size_max//2, n_size_max )
				if n_acc + g_n < batch_n_max:
					n_acc += g_n
					instances += 1
					max_n = max( max_n, g_n )
					M_adj,M_probs = create_graph( g_n )
					batch.append( (M_adj, M_probs) )
				else:
					break
				#end if
			#end for
			M, labels, nodesPerProblem = create_batch( batch )
			test_n = M[2][0]
			test_m = len( M[0] )
			
			test_loss, test_degc, test_degacc = sess.run(
					[ GNN["loss"], GNN["prob_degree_predict_cost"], GNN["prob_degree_predict_acc"]],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["M"]: sparse_to_dense( M ) ,
					GNN["gnn"].time_steps: time_steps,
					GNN["labels"]: sparse_to_dense( labels ),
					GNN["nodes_n"]: nodesPerProblem
				}
			)
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,i): ({n},{m},{i})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f}) Error(D:{degree_acc:.5f}))".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "rnd",
					loss = test_loss,
					degree_cost = test_degc,
					degree_acc = test_degacc,
					n = test_n,
					m = test_m,
					i = instances
				),
				flush = True
			)
			
			#end if
			save_weights(sess,"degreecentrality-checkpoints")
		#end for(epochs)
	#end Session
