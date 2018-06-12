import sys, os, time
import tensorflow as tf
import numpy as np
import networkx as nx
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import model builder
from graphnn import GraphNN
from mlp import Mlp
# Import tools
#import itertools
from util import timestamp, memory_usage, sparse_to_dense#, percent_error

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

def calc_degree( G, T, g_n ):
	# Calculates the degree centrality, non-normalized
	degree = crop_to_float32_mantissa_limit( nx.degree_centrality( G )[T] )
	degree = degree * ( g_n - 1 )
	return degree
#end calc_degree

def calc_betweenness( G, T, g_n ):
	# Calculates the betweenness centrality, non-normalized
	return crop_to_float32_mantissa_limit( nx.betweenness_centrality( G, normalized = False )[T] )
#end calc_betweenness

def calc_closeness( G, T, g_n ):
	# Calculates the closeness centrality, non-normalized
	closeness = crop_to_float32_mantissa_limit( nx.closeness_centrality( G, T ) )
	closeness_n = len( nx.node_connected_component(G, T) )
	closeness = closeness * ( g_n - 1 ) / ( closeness_n - 1 )
	return closeness
#end calc_closeness

def calc_eigenvector( G, T, g_n ):
	# Calculates the eigenvector centrality
	return crop_to_float32_mantissa_limit( nx.eigenvector_centrality( G )[T] )
#end calc_eigenvector

def build_network(d):
	# Builds the model
	
	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define GNN dictionary
	GNN = {}

	# Define placeholder for result values (one per problem)
	instance_degree = tf.placeholder( tf.float32, [ None ], name = "instance_degree" )
	instance_betweenness = tf.placeholder( tf.float32, [ None ], name = "instance_betweenness" )
	instance_closeness = tf.placeholder( tf.float32, [ None ], name = "instance_closeness" )
	instance_eigenvector = tf.placeholder( tf.float32, [ None ], name = "instance_eigenvector" )
	instance_target = tf.placeholder( tf.int32, [ None ], name = "instance_target" )

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
		Cell_activation = tf.nn.sigmoid,
		Msg_last_activation = tf.nn.sigmoid,
		name="Centrality",
	)

	# Define votes
	degree_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "degree_MLP",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
	)
	betweenness_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "betweenness_MLP",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
	)
	closeness_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "closeness_MLP",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
	)
	eigenvector_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "eigenvector_MLP",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
	)

	# Compute the number of variables
	n = tf.shape( gnn.matrix_placeholders["M"] )[0]
	# Compute number of problems
	p = tf.shape( instance_target )[0]

	# Get the last embeddings
	N_n = gnn.last_states["N"].h
	degree_vote = degree_MLP( N_n )
	betweenness_vote = betweenness_MLP( N_n )
	closeness_vote = closeness_MLP( N_n )
	eigenvector_vote = eigenvector_MLP( N_n )

	# Reorganize votes' result to obtain a prediction for each problem instance
	def _vote_while_cond(i, predicted_degree, predicted_betweenness, predicted_closeness, predicted_eigenvector):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, predicted_degree, predicted_betweenness, predicted_closeness, predicted_eigenvector):
		# Gather the target nodes for that problem
		final_node_degree = degree_vote[ instance_target[i] ]
		problem_predicted_degree = tf.reshape( final_node_degree, shape = [] )
		final_node_betweenness = betweenness_vote[ instance_target[i] ]
		problem_predicted_betweenness = tf.reshape( final_node_betweenness, shape = [] )
		final_node_closeness = closeness_vote[ instance_target[i] ]
		problem_predicted_closeness = tf.reshape( final_node_closeness, shape = [] )
		final_node_eigenvector = eigenvector_vote[ instance_target[i] ]
		problem_predicted_eigenvector = tf.reshape( final_node_eigenvector, shape = [] )
		# Update TensorArray
		predicted_degree = predicted_degree.write( i, problem_predicted_degree )
		predicted_betweenness = predicted_betweenness.write( i, problem_predicted_betweenness )
		predicted_closeness = predicted_closeness.write( i, problem_predicted_closeness )
		predicted_eigenvector = predicted_eigenvector.write( i, problem_predicted_eigenvector )
		return tf.add( i, tf.constant( 1 ) ), predicted_degree, predicted_betweenness, predicted_closeness, predicted_eigenvector
	#end _vote_while_body
			
	predicted_degree = tf.TensorArray( size = p, dtype = tf.float32 )
	predicted_betweenness = tf.TensorArray( size = p, dtype = tf.float32 )
	predicted_closeness = tf.TensorArray( size = p, dtype = tf.float32 )
	predicted_eigenvector = tf.TensorArray( size = p, dtype = tf.float32 )
	
	_, predicted_degree, predicted_betweenness, predicted_closeness, predicted_eigenvector = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), predicted_degree, predicted_betweenness,predicted_closeness, predicted_eigenvector ]
	)
	predicted_degree = predicted_degree.stack()
	predicted_betweenness = predicted_betweenness.stack()
	predicted_closeness = predicted_closeness.stack()
	predicted_eigenvector = predicted_eigenvector.stack()

	# Define loss, %error
	betweenness_predict_costs = tf.losses.mean_squared_error( labels = instance_betweenness, predictions = predicted_betweenness )
	degree_predict_costs = tf.losses.mean_squared_error( labels = instance_degree, predictions = predicted_degree )
	closeness_predict_costs = tf.losses.mean_squared_error( labels = instance_closeness, predictions = predicted_closeness )
	eigenvector_predict_costs = tf.losses.mean_squared_error( labels = instance_eigenvector, predictions = predicted_eigenvector )
	betweenness_predict_cost = tf.reduce_mean( betweenness_predict_costs )
	degree_predict_cost = tf.reduce_mean( degree_predict_costs )
	closeness_predict_cost = tf.reduce_mean( closeness_predict_costs )
	eigenvector_predict_cost = tf.reduce_mean( eigenvector_predict_costs )
	vars_cost = tf.zeros([])
	tvars = tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	loss = tf.add_n( [ degree_predict_cost, betweenness_predict_cost, closeness_predict_cost, eigenvector_predict_cost, tf.multiply( vars_cost, parameter_l2norm_scaling ) ] )
	optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
	train_step = optimizer.apply_gradients( zip( grads, tvars ) )
	
	
	degree_predict_error = percent_error( labels = instance_degree, predictions = predicted_degree )
	betweenness_predict_error = percent_error( labels = instance_betweenness, predictions = predicted_betweenness )
	closeness_predict_error = percent_error( labels = instance_closeness, predictions = predicted_closeness )
	eigenvector_predict_error = percent_error( labels = instance_eigenvector, predictions = predicted_eigenvector )
	error = tf.reduce_mean( [ betweenness_predict_error, closeness_predict_error, eigenvector_predict_error ] )
	
	GNN["gnn"] = gnn
	GNN["instance_degree"] = instance_degree
	GNN["instance_betweenness"] = instance_betweenness
	GNN["instance_closeness"] = instance_closeness
	GNN["instance_eigenvector"] = instance_eigenvector
	GNN["instance_target"] = instance_target
	GNN["predicted_degree"] = predicted_degree
	GNN["predicted_betweenness"] = predicted_betweenness
	GNN["predicted_closeness"] = predicted_closeness
	GNN["predicted_eigenvector"] = predicted_eigenvector
	GNN["degree_predict_cost"] = degree_predict_cost
	GNN["betweenness_predict_cost"] = betweenness_predict_cost
	GNN["closeness_predict_cost"] = closeness_predict_cost
	GNN["eigenvector_predict_cost"] = eigenvector_predict_cost
	GNN["error"] = error
	GNN["degree_predict_error"] = degree_predict_error
	GNN["betweenness_predict_error"] = betweenness_predict_error
	GNN["closeness_predict_error"] = closeness_predict_error
	GNN["eigenvector_predict_error"] = eigenvector_predict_error
	GNN["loss"] = loss
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
		G[s][t]["weight"] = np.random.rand()
	#end for
	# Define a random target
	T = np.random.randint( 0, g_n )
	# Calculate centrality measures
	try:
		eigenvector = calc_eigenvector( G, T, g_n )
	except nx.exception.PowerIterationFailedConvergence as e:
		print( e, file = sys.stderr, flush = True )
		return None
	#end try
	degree = calc_degree( G, T, g_n )
	betweenness = calc_betweenness( G, T, g_n )
	closeness = calc_closeness( G, T, g_n )
	# Build matrices
	# TODO: Get values from the edges
	M_index, M_values = build_M_from_graph( G, key = "weight" )
	M = [M_index,M_values,(g_n,g_n)]
	return M,T,degree,betweenness,closeness,eigenvector
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
	n = 0
	m = 0
	batch_M_index = []
	batch_M_value = []
	targets = []
	for p in problems:
		if p is None:
			continue
		#end if
		M, t = p
		# Reindex the matrix to the new indexes
		for i, v in reindex_matrix( n, n, M ):
			batch_M_index.append( i )
			batch_M_value.append( v )
		#end for
		targets.append( n + t )
		# Update n and m
		n += M[2][0]
		m += len(M[0])
	#end for
	M = [batch_M_index,batch_M_value,(n,n)]
	return M, targets
#end create_batch

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
	config = tf.ConfigProto( device_count = {"GPU":0})
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
			epoch_betc = 0
			epoch_cloc = 0
			epoch_eigc = 0
			epoch_err = 0.0
			epoch_dege = 0
			epoch_bete = 0
			epoch_cloe = 0
			epoch_eige = 0
			epoch_n = 0
			epoch_m = 0
			for batch_i in range( batches_per_epoch ):
				# Create random graphs
				batch_n_size = np.random.randint( n_size_min, n_size_max+1 )
				n_acc = 0
				max_n = 0
				instances = 0
				batch = []
				degree_vals = []
				betweenness_vals = []
				closeness_vals = []
				eigenvector_vals = []
				targets = []
				# Generate graphs until the batch can't fit another one
				while True:
					g_n = np.random.randint( batch_n_size//2, batch_n_size*2 )
					if n_acc + g_n < batch_n_max:
						n_acc += g_n
						instances += 1
						max_n = max( max_n, g_n )
						g,t,d,b,c,e = create_graph( g_n )
						batch.append( (g,t) )
						degree_vals.append( d )
						betweenness_vals.append( b )
						closeness_vals.append( c )
						eigenvector_vals.append( e )
					else:
						break
					#end if
				#end for
				# Create the reindexed batch matrix and target-list
				M, targets = create_batch( batch )
				n = M[2][0]
				m = len( M[0] )
				
				_, loss, degc, betc, cloc, eigc, err, dege, bete, cloe, eige = sess.run(
					[ GNN["train_step"], GNN["loss"], GNN["degree_predict_cost"], GNN["betweenness_predict_cost"], GNN["closeness_predict_cost"], GNN["eigenvector_predict_cost"], GNN["error"], GNN["betweenness_predict_error"], GNN["degree_predict_error"], GNN["closeness_predict_error"], GNN["eigenvector_predict_error"] ],
					feed_dict = {
						GNN["gnn"].matrix_placeholders["M"]: sparse_to_dense( M ),
						GNN["gnn"].time_steps: time_steps,
						GNN["instance_degree"]: degree_vals,
						GNN["instance_betweenness"]: betweenness_vals,
						GNN["instance_closeness"]: closeness_vals,
						GNN["instance_eigenvector"]: eigenvector_vals,
						GNN["instance_target"]: targets
					}
				)
				
				epoch_loss += loss
				epoch_degc += degc
				epoch_betc += betc
				epoch_cloc += cloc
				epoch_eigc += eigc
				epoch_err += err
				epoch_dege += dege
				epoch_bete += bete
				epoch_cloe += cloe
				epoch_eige += eige
				epoch_n += n
				epoch_m += m
				
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,i): ({n},{m},{i})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f},B:{betweenness_cost:.5f},C:{closeness_cost:.5f},E:{eigenvector_cost:.5f}) Error(T:{error:.5f},D:{degree_error:.5f},B:{betweenness_error:.5f},C:{closeness_error:.5f},E:{eigenvector_error:.5f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = batch_i,
						loss = loss,
						degree_cost = degc,
						betweenness_cost = betc,
						closeness_cost = cloc,
						eigenvector_cost = eigc,
						error = err,
						degree_error = dege,
						betweenness_error = bete,
						closeness_error = cloe,
						eigenvector_error = eige,
						n = n,
						m = m,
						i = instances
					),
					flush = True
				)
			#end for
			# Summarize Epoch
			epoch_loss /= batches_per_epoch
			epoch_betc /= batches_per_epoch
			epoch_betc /= batches_per_epoch
			epoch_cloc /= batches_per_epoch
			epoch_eigc /= batches_per_epoch
			epoch_err /= batches_per_epoch
			epoch_dege /= batches_per_epoch
			epoch_bete /= batches_per_epoch
			epoch_cloe /= batches_per_epoch
			epoch_eige /= batches_per_epoch
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f},B:{betweenness_cost:.5f},C:{closeness_cost:.5f},E:{eigenvector_cost:.5f}) Error(T:{error:.5f},D:{degree_error:.5f},B:{betweenness_error:.5f},C:{closeness_error:.5f},E:{eigenvector_error:.5f}))".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "all",
					loss = epoch_loss,
					degree_cost = epoch_degc,
					betweenness_cost = epoch_betc,
					closeness_cost = epoch_cloc,
					eigenvector_cost = epoch_eigc,
					error = epoch_err,
					degree_error = epoch_dege,
					betweenness_error = epoch_bete,
					closeness_error = epoch_cloe,
					eigenvector_error = epoch_eige,
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
			degree_vals = []
			betweenness_vals = []
			closeness_vals = []
			eigenvector_vals = []
			targets = []
			while True:
				g_n = np.random.randint( n_size_max//2, n_size_max )
				if n_acc + g_n < batch_n_max:
					n_acc += g_n
					instances += 1
					max_n = max( max_n, g_n )
					g,t,d,b,c,e = create_graph( g_n )
					batch.append( (g,t) )
					degree_vals.append( d )
					betweenness_vals.append( b )
					closeness_vals.append( c )
					eigenvector_vals.append( e )
				else:
					break
				#end if
			#end for
			M, targets = create_batch( batch )
			test_n = M[2][0]
			test_m = len( M[0] )
			
			test_loss, test_degc, test_betc, test_cloc, test_eigc, test_err, test_dege, test_bete, test_cloe, test_eige = sess.run(
					[ GNN["loss"], GNN["degree_predict_cost"], GNN["betweenness_predict_cost"], GNN["closeness_predict_cost"], GNN["eigenvector_predict_cost"], GNN["error"], GNN["degree_predict_error"], GNN["betweenness_predict_error"], GNN["closeness_predict_error"], GNN["eigenvector_predict_error"] ],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["M"]: sparse_to_dense( M ),
					GNN["gnn"].time_steps: time_steps,
					GNN["instance_degree"]: degree_vals,
					GNN["instance_betweenness"]: betweenness_vals,
					GNN["instance_closeness"]: closeness_vals,
					GNN["instance_eigenvector"]: eigenvector_vals,
					GNN["instance_target"]: targets
				}
			)
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,i): ({n},{m},{i})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f},B:{betweenness_cost:.5f},C:{closeness_cost:.5f},E:{eigenvector_cost:.5f}) Error(T:{error:.5f},D:{degree_error:.5f},B:{betweenness_error:.5f},C:{closeness_error:.5f},E:{eigenvector_error:.5f}))".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "rnd",
					loss = test_loss,
					degree_cost = test_degc,
					betweenness_cost = test_betc,
					closeness_cost = test_cloc,
					eigenvector_cost = test_eigc,
					error = test_err,
					degree_error = test_dege,
					betweenness_error = test_bete,
					closeness_error = test_cloe,
					eigenvector_error = test_eige,
					n = test_n,
					m = test_m,
					i = instances
				),
				flush = True
			)
			# Test with real graphs
			test_loss = 0.0
			test_degc = 0.0
			test_betc = 0.0
			test_cloc = 0.0
			test_eigc = 0.0
			test_err = 0.0
			test_dege = 0.0
			test_bete = 0.0
			test_cloe = 0.0
			test_eige = 0.0
			instances = 0
			test_n = 0
			test_m = 0
			for f in os.listdir("./centrality/test"):
				if f.endswith(".g"):
					G = nx.read_edgelist( os.path.join("./centrality/test", f) )
					G = nx.convert_node_labels_to_integers(G)
					g_n = len( G.nodes )
					T = np.random.randint( 0, g_n )
					degree = calc_degree( G, T, g_n )
					betweenness = calc_betweenness( G, T, g_n )
					closeness = calc_closeness( G, T, g_n )
					try:
						eigenvector = calc_eigenvector( G, T, g_n )
					except nx.exception.PowerIterationFailedConvergence as e:
						print( e, file = sys.stderr, flush = True )
						continue
					#end try
					M_i, M_v = build_M_from_graph( G )
					M = [M_i,M_v,(g_n,g_n)]
					loss, degc, betc, cloc, eigc, err, dege, bete, cloe, eige = sess.run(
						[ GNN["loss"], GNN["degree_predict_cost"], GNN["betweenness_predict_cost"], GNN["closeness_predict_cost"], GNN["eigenvector_predict_cost"], GNN["error"], GNN["degree_predict_error"], GNN["betweenness_predict_error"], GNN["closeness_predict_error"], GNN["eigenvector_predict_error"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["M"]: sparse_to_dense( M ),
							GNN["gnn"].time_steps: time_steps,
							GNN["instance_degree"]: [ degree ],
							GNN["instance_betweenness"]: [ betweenness ],
							GNN["instance_closeness"]: [ closeness ],
							GNN["instance_eigenvector"]: [ eigenvector ],
							GNN["instance_target"]: [ T ]
						}
					)
					print( f, loss, degc, betc, cloc, eigc, err, dege, bete, cloe, eige, file=sys.stderr )
					test_n += M[2][0]
					test_m += len( M[0] )
					test_loss += loss
					test_degc += degc
					test_betc += betc
					test_cloc += cloc
					test_eigc += eigc
					test_err += err
					test_dege += dege
					test_bete += bete
					test_cloe += cloe
					test_eige += eige
					instances += 1
				#end if
			#end for
			if instances > 0:
				test_loss /= instances
				test_degc /= instances
				test_betc /= instances
				test_cloc /= instances
				test_eigc /= instances
				test_err /= instances
				test_dege /= instances
				test_bete /= instances
				test_cloe /= instances
				test_eige /= instances
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,i): ({n},{m},{i})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f},B:{betweenness_cost:.5f},C:{closeness_cost:.5f},E:{eigenvector_cost:.5f}) Error(T:{error:.5f},D:{degree_error:.5f},B:{betweenness_error:.5f},C:{closeness_error:.5f},E:{eigenvector_error:.5f}))".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = "tst",
						loss = test_loss,
						degree_cost = test_degc,
						betweenness_cost = test_betc,
						closeness_cost = test_cloc,
						eigenvector_cost = test_eigc,
						error = test_err,
						degree_error = test_dege,
						betweenness_error = test_bete,
						closeness_error = test_cloe,
						eigenvector_error = test_eige,
						n = test_n,
						m = test_m,
						i = instances
					),
					flush = True
				)
			#end if
		#end for(epochs)
	#end Session
