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

def timestamp():
	return time.strftime( "%Y%m%d%H%M%S", time.gmtime() )
#end timestamp

def memory_usage():
	pid=os.getpid()
	s = next( line for line in open( '/proc/{}/status'.format( pid ) ).read().splitlines() if line.startswith( 'VmSize' ) ).split()
	return "{} {}".format( s[-2], s[-1] )
#end memory_usage

def dense_to_sparse( M ):
	n, m = M.shape
	M_i = []
	M_v = []
	M_shape = (n,m)
	for i in range( n ):
		for j in range( m ):
			if M[i,j] > 0.5:
				M_i.append( (i,j ) )
				M_v.append( 1 )
			#end if
		#end for
	#end for
	return (M_i,M_v,M_shape)
#end dense_to_sparse

def percent_error(labels,predictions):
	return tf.reduce_mean( tf.divide( tf.abs( tf.subtract( labels, predictions ) ), labels ) )
#end percent_error

def build_network(d):

	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define GNN dictionary
	GNN = {}

	# Define placeholder for result values (one per problem)
	instance_betweenness = tf.placeholder( tf.float32, [ None ], name = "instance_betweenness" )
	instance_closeness = tf.placeholder( tf.float32, [ None ], name = "instance_closeness" )
	instance_eigenvector = tf.placeholder( tf.float32, [ None ], name = "instance_eigenvector" )
	instance_target = tf.placeholder( tf.int32, [ None ], name = "instance_target" )

	# Define INV, a tf function to exchange positive and negative literal embeddings
	def INV(Lh):
		l = tf.shape(Lh)[0]
		n = tf.div(l,tf.constant(2))
		# Send messages from negated literals to positive ones, and vice-versa
		Lh_pos = tf.gather( Lh, tf.range( tf.constant( 0 ), n ) )
		Lh_neg = tf.gather( Lh, tf.range( n, l ) )
		Lh_inverted = tf.concat( [ Lh_neg, Lh_pos ], axis = 0 )
		return Lh_inverted
	#end

	# Define Graph neural network
	gnn = GraphNN(
		{
			"N": d
		},
		{
			"M": ("N","N")
		},
		{
			"Nmsg": ("N","N")
		},
		{
			"N": [
				{
					"mat": "M",
					"var": "N"
				},
				{
					"mat": "M",
					"transpose?": True,
					"var": "N"
				}
			]
		},
		name="Centrality",
	)

	# Define L_vote
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
	betweenness_vote = betweenness_MLP( N_n )
	closeness_vote = closeness_MLP( N_n )
	eigenvector_vote = eigenvector_MLP( N_n )

	# Reorganize votes' result to obtain a prediction for each problem instance
	def _vote_while_cond(i, predicted_betweenness, predicted_closeness, predicted_eigenvector):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, predicted_betweenness, predicted_closeness, predicted_eigenvector):
		# Gather the target nodes for that problem
		final_node_betweenness = betweenness_vote[ instance_target[i] ]
		problem_predicted_betweenness = tf.reshape( final_node_betweenness, shape = [] )
		final_node_closeness = closeness_vote[ instance_target[i] ]
		problem_predicted_closeness = tf.reshape( final_node_closeness, shape = [] )
		final_node_eigenvector = eigenvector_vote[ instance_target[i] ]
		problem_predicted_eigenvector = tf.reshape( final_node_eigenvector, shape = [] )
		# Update TensorArray
		predicted_betweenness = predicted_betweenness.write( i, problem_predicted_betweenness )
		predicted_closeness = predicted_closeness.write( i, problem_predicted_closeness )
		predicted_eigenvector = predicted_eigenvector.write( i, problem_predicted_eigenvector )
		return tf.add( i, tf.constant( 1 ) ), predicted_betweenness, predicted_closeness, predicted_eigenvector
	#end _vote_while_body
			
	predicted_betweenness = tf.TensorArray( size = p, dtype = tf.float32 )
	predicted_closeness = tf.TensorArray( size = p, dtype = tf.float32 )
	predicted_eigenvector = tf.TensorArray( size = p, dtype = tf.float32 )
	
	_, predicted_betweenness, predicted_closeness, predicted_eigenvector = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), predicted_betweenness, predicted_closeness, predicted_eigenvector ]
	)
	predicted_betweenness = predicted_betweenness.stack()
	predicted_closeness = predicted_closeness.stack()
	predicted_eigenvector = predicted_eigenvector.stack()

	# Define loss, %error
	betweenness_predict_costs = tf.losses.mean_squared_error( labels = instance_betweenness, predictions = predicted_betweenness )
	closeness_predict_costs = tf.losses.mean_squared_error( labels = instance_closeness, predictions = predicted_closeness )
	eigenvector_predict_costs = tf.losses.mean_squared_error( labels = instance_eigenvector, predictions = predicted_eigenvector )
	betweenness_predict_cost = tf.reduce_mean( betweenness_predict_costs )
	closeness_predict_cost = tf.reduce_mean( closeness_predict_costs )
	eigenvector_predict_cost = tf.reduce_mean( eigenvector_predict_costs )
	vars_cost = tf.zeros([])
	tvars = tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	loss = tf.add_n( [ betweenness_predict_cost, closeness_predict_cost, eigenvector_predict_cost, tf.multiply( vars_cost, parameter_l2norm_scaling ) ] )
	optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
	train_step = optimizer.apply_gradients( zip( grads, tvars ) )
	
	
	betweenness_predict_error = percent_error( labels = instance_betweenness, predictions = predicted_betweenness )
	closeness_predict_error = percent_error( labels = instance_closeness, predictions = predicted_closeness )
	eigenvector_predict_error = percent_error( labels = instance_eigenvector, predictions = predicted_eigenvector )
	error = tf.reduce_mean( [ betweenness_predict_error, closeness_predict_error, eigenvector_predict_error ] )
	
	GNN["gnn"] = gnn
	GNN["instance_betweenness"] = instance_betweenness
	GNN["instance_closeness"] = instance_closeness
	GNN["instance_eigenvector"] = instance_eigenvector
	GNN["instance_target"] = instance_target
	GNN["predicted_betweenness"] = predicted_betweenness
	GNN["predicted_closeness"] = predicted_closeness
	GNN["predicted_eigenvector"] = predicted_eigenvector
	GNN["betweenness_predict_cost"] = betweenness_predict_cost
	GNN["closeness_predict_cost"] = closeness_predict_cost
	GNN["eigenvector_predict_cost"] = eigenvector_predict_cost
	GNN["error"] = error
	GNN["betweenness_predict_error"] = betweenness_predict_error
	GNN["closeness_predict_error"] = closeness_predict_error
	GNN["eigenvector_predict_error"] = eigenvector_predict_error
	GNN["loss"] = loss
	GNN["train_step"] = train_step
	return GNN
#end build_network

def create_graph( g_n, erdos_renyi_p = 0.25, powerlaw_gamma = 3, smallworld_k = 4, smallworld_p = 0.25, powerlaw_cluster_m = 3, powerlaw_cluster_p = 0.1  ):
	Gs = None
	while Gs is None:
		Gs = _create_graph( g_n, erdos_renyi_p, powerlaw_gamma, smallworld_k, smallworld_p, powerlaw_cluster_m, powerlaw_cluster_p )
	#end while
	return Gs
#end create_graph

def _create_graph( g_n, erdos_renyi_p = 0.25, powerlaw_gamma = 3, smallworld_k = 4, smallworld_p = 0.25, powerlaw_cluster_m = 3, powerlaw_cluster_p = 0.1 ):
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
	# TODO: Add weights to the edges and fill the sparse matrices
	M_index = []
	M_values = []
	for s, t in G.edges:
		G[s][t]["weight"] = np.random.rand()
		M_index.append( ( s, t ) )
		M_values.append( 1 )
		M_index.append( ( t, s ) )
		M_values.append( 1 )
	#end for
	T = np.random.randint( 0, g_n )
	betweenness = nx.betweenness_centrality( G )[T]
	closeness = nx.closeness_centrality( G, T )
	try:
		eigenvector = nx.eigenvector_centrality( G )[T]
	except nx.exception.PowerIterationFailedConvergence as e:
		print( e, file = sys.stderr, flush = True )
		return None
	#end try
	M = [M_index,M_values,(g_n,g_n)]
	return M,T,betweenness,closeness,eigenvector
#end _create_graph

def reindex_matrix( n, m, M ):
	new_index = []
	new_value = []
	for i, v in zip( M[0], M[1] ):
		s, t = i
		new_index.append( (n + s, m + t) )
		new_value.append( v )
	#end for
	return zip( new_index, new_value )

def create_batch(problems):
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
		for i, v in reindex_matrix( n, n, M ):
			batch_M_index.append( i )
			batch_M_value.append( v )
		#end for
		targets.append( n + t )
		n += M[2][0]
		m += len(M[0])
	#end for
	M = [batch_M_index,batch_M_value,(n,n)]
	return M, targets
#end create_batch

if __name__ == '__main__':
	d = 64
	epochs = 100
	batch_n_max = 4096
	batches_per_epoch = 32
	n_size_min = 16
	n_size_max = 512
	edge_probability = 0.25
	time_steps = 32

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(d)

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
			epoch_betc = 0
			epoch_cloc = 0
			epoch_eigc = 0
			epoch_err = 0.0
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
				betweenness_vals = []
				closeness_vals = []
				eigenvector_vals = []
				targets = []
				while True:
					g_n = np.random.randint( batch_n_size//2, batch_n_size*2 )
					if n_acc + g_n < batch_n_max:
						n_acc += g_n
						instances += 1
						max_n = max( max_n, g_n )
						g,t,b,c,e = create_graph( g_n )
						batch.append( (g,t) )
						betweenness_vals.append( b )
						closeness_vals.append( c )
						eigenvector_vals.append( e )
					else:
						break
					#end if
				#end for
				M, targets = create_batch( batch )
				n = M[2][0]
				m = len( M[0] )
				print( targets )

				_, loss, betc, cloc, eigc, err, bete, cloe, eige = sess.run(
					[ GNN["train_step"], GNN["loss"], GNN["betweenness_predict_cost"], GNN["closeness_predict_cost"], GNN["eigenvector_predict_cost"], GNN["error"], GNN["betweenness_predict_error"], GNN["closeness_predict_error"], GNN["eigenvector_predict_error"] ],
					feed_dict = {
						GNN["gnn"].matrix_placeholders["M"]: M,
						GNN["gnn"].time_steps: time_steps,
						GNN["instance_betweenness"]: betweenness_vals,
						GNN["instance_closeness"]: closeness_vals,
						GNN["instance_eigenvector"]: eigenvector_vals,
						GNN["instance_target"]: targets
					}
				)
				print( err, bete, cloe, eige )
				
				epoch_loss += loss
				epoch_betc += betc
				epoch_cloc += cloc
				epoch_eigc += eigc
				epoch_err += err
				epoch_betc += bete
				epoch_cloc += cloe
				epoch_eigc += eige
				epoch_n += n
				epoch_m += m
				
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,i): ({n},{m},{i})\t| Loss(T:{loss:.5f},B:{betweenness_cost:.5f},C:{closeness_cost:.5f},E:{eigenvector_cost:.5f}) Error(T:{error:.5f},B:{betweenness_error:.5f},C:{closeness_error:.5f},E:{eigenvector_error:.5f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = batch_i,
						loss = loss,
						betweenness_cost = betc,
						closeness_cost = cloc,
						eigenvector_cost = eigc,
						error = err,
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
			epoch_loss = epoch_loss / batches_per_epoch
			epoch_betc = epoch_betc / batches_per_epoch
			epoch_cloc = epoch_cloc / batches_per_epoch
			epoch_eigc = epoch_eigc / batches_per_epoch
			epoch_err = epoch_err / batches_per_epoch
			epoch_bete = epoch_bete / batches_per_epoch
			epoch_cloe = epoch_cloe / batches_per_epoch
			epoch_eige = epoch_eige / batches_per_epoch
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Loss(T:{loss:.5f},B:{betweenness_cost:.5f},C:{closeness_cost:.5f},E:{eigenvector_cost:.5f}) Error(T:{error:.5f},B:{betweenness_error:.5f},C:{closeness_error:.5f},E:{eigenvector_error:.5f}))".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "all",
					loss = epoch_loss,
					betweenness_cost = epoch_betc,
					closeness_cost = epoch_cloc,
					eigenvector_cost = epoch_eigc,
					error = epoch_err,
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
					g,t,b,c,e = create_graph( g_n )
					batch.append( (g,t) )
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
			
			test_loss, test_betc, test_cloc, test_eigc, test_err, test_bete, test_cloe, test_eige = sess.run(
					[ GNN["loss"], GNN["betweenness_predict_cost"], GNN["closeness_predict_cost"], GNN["eigenvector_predict_cost"], GNN["error"], GNN["betweenness_predict_error"], GNN["closeness_predict_error"], GNN["eigenvector_predict_error"] ],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["M"]: M,
					GNN["gnn"].time_steps: time_steps,
					GNN["instance_betweenness"]: betweenness_vals,
					GNN["instance_closeness"]: closeness_vals,
					GNN["instance_eigenvector"]: eigenvector_vals,
					GNN["instance_target"]: targets
				}
			)
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,i): ({n},{m},{i})\t| Loss(T:{loss:.5f},B:{betweenness_cost:.5f},C:{closeness_cost:.5f},E:{eigenvector_cost:.5f}) Error(T:{error:.5f},B:{betweenness_error:.5f},C:{closeness_error:.5f},E:{eigenvector_error:.5f}))".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "tst",
					loss = test_loss,
					betweenness_cost = test_betc,
					closeness_cost = test_cloc,
					eigenvector_cost = test_eigc,
					error = test_err,
					betweenness_error = test_bete,
					closeness_error = test_cloe,
					eigenvector_error = test_eige,
					n = test_n,
					m = test_m,
					i = instances
				),
				flush = True
			)
			
	#end Session
