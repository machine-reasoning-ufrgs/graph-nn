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

def build_network(d):

	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define GNN dictionary
	GNN = {}

	# Define placeholder for result values (one per problem)
	instance_val = tf.placeholder( tf.float32, [ None ], name = "instance_val" )
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
			"M": ("N","N"),
			"S": ("N","N")
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
				},
				{
					"mat": "S"
				}
			]
		},
		name="Dijkstra",
		float_dype = tf.float32
	)

	# Define L_vote
	N_vote_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "N_vote",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
		)

	# Compute the number of variables
	n = tf.shape( gnn.matrix_placeholders["M"] )[0]
	# Compute number of problems
	p = tf.shape( instance_val )[0]

	# Get the last embeddings
	N_n = gnn.last_states["N"].h
	N_vote = N_vote_MLP( N_n )

	# Reorganize votes' result to obtain a prediction for each problem instance
	def _vote_while_cond(i, predicted_val):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, predicted_val):
		# Gather the target node for that problem
		final_node = N_vote[ instance_target[i] ]
		# Concatenate positive and negative literals and average their vote values
		problem_predicted_val = tf.reshape( final_node, shape = [] )
		# Update TensorArray
		predicted_val = predicted_val.write( i, problem_predicted_val )
		return tf.add( i, tf.constant( 1 ) ), predicted_val
	#end _vote_while_body
			
	predicted_val = tf.TensorArray( size = p, dtype = tf.float32 )
	_, predicted_val = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), predicted_val ]
	)
	predicted_val = predicted_val.stack()

	# Define loss, %error
	predict_costs = tf.losses.mean_squared_error( labels = instance_val, predictions = predicted_val )
	predict_cost = tf.reduce_mean( predict_costs )
	# "%Error"
	error = tf.reduce_mean( tf.divide( tf.subtract( instance_val, predicted_val ), predicted_val ) )
	vars_cost = tf.zeros([])
	tvars = tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	loss = tf.add( predict_cost, tf.multiply( vars_cost, parameter_l2norm_scaling ) )
	optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
	train_step = optimizer.apply_gradients( zip( grads, tvars ) )
	
	GNN["gnn"] = gnn
	GNN["instance_val"] = instance_val
	GNN["instance_target"] = instance_target
	GNN["predicted_val"] = predicted_val
	GNN["loss"] = loss
	GNN["%error"] = error
	GNN["train_step"] = train_step
	return GNN
#end build_network

def create_graph( g_n, edge_probability, max_multiplier = 2 ):
	Gs = None
	while Gs is None:
		Gs = _create_graph( g_n, edge_probability, max_multiplier )
	#end while
	return Gs
#end create_graph

def _create_graph( g_n, edge_probability, max_multiplier = 2 ):
	k = np.random.randint( 2, max_multiplier + 1 )
	# Create the main graph along with a scaled version of it
	M1_index = []
	M1_values = []
	M2_index = []
	M2_values = []
	M3_index = []
	M3_values = []
	G1 = nx.fast_gnp_random_graph( g_n, edge_probability )
	G2 = G1.copy()
	f1_norm = 0
	f2_norm = 0
	for s, t in G1.edges:
		G1[s][t]["distance"] = np.random.rand()
		M1_index.append( ( s, t ) )
		M1_values.append( 1 / ( 1 + G1[s][t]["distance"] ) )
		M1_index.append( ( t, s ) )
		M1_values.append( 1 / ( 1 + G1[s][t]["distance"] ) )
		f1_norm += 2 * G1[s][t]["distance"]
		G2[s][t]["distance"] = G1[s][t]["distance"] * k
		M2_index.append( ( s, t ) )
		M2_values.append( 1 / ( 1 + G2[s][t]["distance"] ) )
		M2_index.append( ( t, s ) )
		M2_values.append( 1 / ( 1 + G2[s][t]["distance"] ) )
		f2_norm += 2 * G2[s][t]["distance"]
	#end for
	S = np.random.randint( 0, g_n )
	T = S
	while T == S:
		T = np.random.randint( 0, g_n )
	#end while
	if not nx.has_path( G1, S, T ):
		return None
	#end if
	path = nx.shortest_path( G1, S, T, "distance" )
	f1 = nx.shortest_path_length( G1, S, T, "distance" )
	f2 = nx.shortest_path_length( G2, S, T, "distance" )
	# Then create a complementary graph with the shortest path removed
	G3 = G1.copy()
	for node in path:
		if node not in [S,T]:
			G3.remove_node( node )
		#end if
	#end for
	if not nx.has_path( G3, S, T ):
		return None
	#end if
	f3_norm = 0
	for s, t in G3.edges:
		M3_index.append( ( s, t ) )
		M3_values.append( 1 / ( 1 + G3[s][t]["distance"] ) )
		M3_index.append( ( t, s ) )
		M3_values.append( 1 / ( 1 + G3[s][t]["distance"] ) )
		f3_norm += 2 * G3[s][t]["distance"]
	#end for
	f3 = nx.shortest_path_length( G3, S, T, "distance" )
	M1 = [M1_index,M1_values,(g_n,g_n)]
	M2 = [M2_index,M2_values,(g_n,g_n)]
	M3 = [M3_index,M3_values,(g_n,g_n)]
	S_mat = [[(S,S)],[1],(g_n,g_n)]
	T_mat = [[(T,T)],[1],(g_n,g_n)]
	f1_norm = f1_norm if f1_norm > 0 else 1
	f2_norm = f2_norm if f2_norm > 0 else 1
	f3_norm = f3_norm if f3_norm > 0 else 1
	return ((M1,S_mat,T_mat),f1/f1_norm), ((M2,S_mat,T_mat),f2/f2_norm), ((M3,S_mat,T_mat),f3/f3_norm)
#end _create_graph


def create_graph2( g_n, edge_probability, max_multiplier = 2 ):
	Gs = None
	while Gs is None:
		Gs = _create_graph2( g_n, edge_probability, max_multiplier )
	#end while
	return Gs
#end create_graph2

def _create_graph2( g_n, edge_probability, max_multiplier = 2, factor = 2 ):
	nb = g_n//2
	na = g_n - nb
	
	Ga = nx.fast_gnp_random_graph( na, edge_probability )
	for s, t in Ga.edges:
		Ga[s][t]["distance"] = np.random.rand()
	#end
	Gb = nx.fast_gnp_random_graph( nb, edge_probability )
	for s, t in Gb.edges:
		Gb[s][t]["distance"] = np.random.rand()
	#end
	
	Gb = nx.relabel.convert_node_labels_to_integers(Gb, first_label=na)
	
	sa = np.random.randint( 0, na )
	distances = [ ( ta, nx.shortest_path_length( Ga, sa, ta, weight="distance" ) ) for ta in range(0,na) if ta != sa and nx.has_path( Ga, sa, ta ) ]
	distances.sort( key = lambda ta_d: ta_d[1] )
	if len( distances ) == 0:
		return None
	#end if
	ta,da = distances[-1]
	sb = np.random.randint( na, na+nb )
	distances = [ ( tb, nx.shortest_path_length( Gb, sb, tb, weight="distance" ) ) for tb in range(na, na+nb) if tb != sb and nx.has_path( Gb, sb, tb ) ]
	distances = [ (tb,db) for tb,db in distances if db < 2 + da/factor ]
	if len( distances ) == 0:
		return None
	#end if
	tb,db = distances[ np.random.randint( 0, len( distances ) ) ]
	
	G1 = nx.union( Ga, Gb )
	G1.add_edge( sa, sb )
	G1[sa][sb]["distance"] = np.random.rand()
	G2 = G1.copy()
	G2.add_edge( ta, tb )
	G2[ta][tb]["distance"] = np.random.rand()
	
	d1 = nx.shortest_path_length( G1, sa, ta, weight="distance" )
	p1 = nx.shortest_path( G1, sa, ta, weight="distance" )
	d2 = nx.shortest_path_length( G2, sa, ta, weight="distance" )
	p2 = nx.shortest_path( G2, sa, ta, weight="distance" )
	if d1 < d2 or p1 == p2:
		return None
	#end if
	M1_index = []
	M1_values = []
	M2_index = []
	M2_values = []
	d1_norm = 0
	d2_norm = 0
	for s, t in G1.edges:
		M1_index.append( ( s, t ) )
		M1_values.append( 1 / ( 1 + G1[s][t]["distance"] ) )
		M1_index.append( ( t, s ) )
		M1_values.append( 1 / ( 1 + G1[s][t]["distance"] ) )
		d1_norm += 2 * G1[s][t]["distance"]
	#end for
	for s, t in G2.edges:
		M2_index.append( ( s, t ) )
		M2_values.append( 1 / ( 1 + G2[s][t]["distance"] ) )
		M2_index.append( ( t, s ) )
		M2_values.append( 1 / ( 1 + G2[s][t]["distance"] ) )
		d2_norm += 2 * G2[s][t]["distance"]
	#end for
	M1 = [M1_index,M1_values,(g_n,g_n)]
	M2 = [M2_index,M2_values,(g_n,g_n)]
	S_mat = [[(sa,sa)],[1],(g_n,g_n)]
	T_mat = [[(ta,ta)],[1],(g_n,g_n)]
	d1_norm = d1_norm if d1_norm > 0 else 1
	d2_norm = d2_norm if d2_norm > 0 else 1
	return ((M1,S_mat,T_mat),d1/d1_norm), ((M2,S_mat,T_mat),d2/d2_norm), None
#end _create_graph2

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
	batch_S_index = []
	batch_S_value = []
	batch_T_index = []
	batch_T_value = []
	for p in problems:
		if p is None:
			continue
		#end if
		M, S, T = p
		for i, v in reindex_matrix( n, n, M ):
			batch_M_index.append( i )
			batch_M_value.append( v )
		#end for
		for i, v in reindex_matrix( n, n, S ):
			batch_S_index.append( i )
			batch_S_value.append( v )
		#end for
		for i, v in reindex_matrix( n, n, T ):
			batch_T_index.append( i )
			batch_T_value.append( v )
		#end for
		n += M[2][0]
		m += len(M[0])
	#end for
	M = [batch_M_index,batch_M_value,(n,n)]
	S = [batch_S_index,batch_S_value,(n,n)]
	T = [batch_T_index,batch_T_value,(n,n)]
	return (M,S,T)
#end create_batch

if __name__ == '__main__':
	d = 64
	epochs = 100
	batch_n_max = 4096
	batches_per_epoch = 32
	n_size_min = 16
	n_loss_increase_threshold = 0.01
	n_size_max = 512
	edge_probability = 0.25

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
		n_size = n_size_min
		for epoch in range( epochs ):
			# Run batches
			#instance_generator.reset()
			epoch_loss = 0.0
			epoch_err = 0
			epoch_n = 0
			epoch_m = 0
			for batch_i in range( batches_per_epoch ):
				# Create random graphs
				batch_n_size = np.random.randint( n_size_min, n_size+1 )
				n_acc = 0
				max_n = 0
				instances = 0
				Gs = []
				distances = []
				while True:
					g_n = np.random.randint( batch_n_size//2, batch_n_size*2 )
					if n_acc + g_n * 3 < batch_n_max:
						n_acc += g_n * 3
						instances += 3
						max_n = max( max_n, g_n )
						(g1,f1),(g2,f2),_ = create_graph2( g_n, edge_probability )
						Gs = Gs + [g1,g2]
						distances = distances + [f1,f2]
					else:
						break
					#end if
				#end for
				M, S, T = create_batch( Gs )
				targets = [ t for (t,_) in T[0] ]
				time_steps = max_n
				batch_allowed_error = sum( distances ) * n_loss_increase_threshold
				n = M[2][0]
				m = len( M[0] )

				_, loss, err = sess.run(
					[ GNN["train_step"], GNN["loss"], GNN["%error"] ],
					feed_dict = {
						GNN["gnn"].matrix_placeholders["M"]: M,
						GNN["gnn"].matrix_placeholders["S"]: S,
						GNN["gnn"].time_steps: time_steps,
						GNN["instance_val"]: distances,
						GNN["instance_target"]: targets
					}
				)
				
				epoch_loss += loss
				epoch_err += err
				epoch_n += n
				epoch_m += m
				
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| (Loss,``%Error''): ({loss:.5f},{error:.5f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = batch_i,
						loss = loss,
						error = err,
						n = n,
						m = m,
						i = instances
					),
					flush = True
				)
			#end for
			# Summarize Epoch
			epoch_loss = epoch_loss / batches_per_epoch
			epoch_err = epoch_err / batches_per_epoch
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Mean (Loss,``%Error''): ({loss:.5f},{error:.5f})".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "all",
					loss = epoch_loss,
					error = epoch_err,
					n = epoch_n,
					m = epoch_m,
				),
				flush = True
			)
			if abs(epoch_err) < n_loss_increase_threshold and n_size < n_size_max:
				n_size = 2 * n_size
				n_size = n_size_max if n_size > n_size_max else n_size
			#end if
			
			# TEST TIME
			# Create random graphs
			n_acc = 0
			max_n = 0
			instances = 0
			Gs = []
			distances = []
			while True:
				g_n = np.random.randint( n_size, n_size*2 )
				if n_acc + g_n < batch_n_max:
					n_acc += g_n
					instances += 1
					max_n = max( max_n, g_n )
					(g1,f1),_,_ = create_graph2( g_n, edge_probability )
					Gs.append( g1 )
					distances.append( f1 )
				else:
					break
				#end if
			#end for
			M, S, T = create_batch( Gs )
			targets = [ t for (t,_) in T[0] ]
			time_steps = max_n
			test_allowed_error = sum( distances ) * n_loss_increase_threshold
			test_n = M[2][0]
			test_m = len( M[0] )
			
			test_loss, test_err = sess.run(
				[GNN["loss"],GNN["%error"]],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["M"]: M,
					GNN["gnn"].matrix_placeholders["S"]: S,
					GNN["gnn"].time_steps: time_steps,
					GNN["instance_val"]: distances,
					GNN["instance_target"]: targets
				}
			)
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| Test (Loss,``%Error''): ({loss:.5f},{error:.5f})".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "tst",
					loss = test_loss,
					error = test_err,
					n = test_n,
					m = test_m,
					i = instances
				),
				flush = True
			)
			
	#end Session