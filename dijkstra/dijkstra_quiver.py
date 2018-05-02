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
			"N": d, # Nodes
			"E": d  # Edges
		},
		{
			"Ms": ("N","E"), # Matrix pointing from nodes to the edges they are sources
			"Mt": ("N","E"), # Matrix pointing from nodes to the edges they are targets 
			"Mw": ("E","E"), # Matrix indicating an Edge weight
			"S": ("N","N"), # Matrix indicating whether a node is the source
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
				},
				{
					"mat": "S"
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
		name="Flow",
		
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
	n = tf.shape( gnn.matrix_placeholders["S"] )[0]
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

	# Define loss, accuracy
	predict_costs = tf.losses.mean_squared_error( labels = instance_val, predictions = predicted_val )
	predict_cost = tf.reduce_mean( predict_costs )
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
	Ms_1_index = []
	Mt_1_index = []
	Mw_1_index = []
	Mw_1_values = []
	Ms_2_index = []
	Mt_2_index = []
	Mw_2_index = []
	Mw_2_values = []
	Ms_3_index = []
	Mt_3_index = []
	Mw_3_index = []
	Mw_3_values = []
	G1 = nx.fast_gnp_random_graph( g_n, edge_probability )
	G2 = G1.copy()
	f1_norm = 0
	f2_norm = 0
	for e, (s, t) in enumerate( G1.edges ):
		G1[s][t]["distance"] = np.random.rand()
		Ms_1_index.append( (s, e) )
		Ms_1_index.append( (t, e) )
		Mt_1_index.append( (s, e) )
		Mt_1_index.append( (t, e) )
		Mw_1_index.append( (e, e) )
		Mw_1_values.append( G1[s][t]["distance"] )
		f1_norm += 2 * G1[s][t]["distance"]
		G2[s][t]["distance"] = G1[s][t]["distance"] * k 
		Ms_2_index.append( (s, e) )
		Ms_2_index.append( (t, e) )
		Mt_2_index.append( (s, e) )
		Mt_2_index.append( (t, e) )
		Mw_2_index.append( (e, e) )
		Mw_2_values.append( G2[s][t]["distance"] )
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
	for e, (s, t) in enumerate( G3.edges ):
		Ms_3_index.append( (s, e) )
		Ms_3_index.append( (t, e) )
		Mt_3_index.append( (s, e) )
		Mt_3_index.append( (t, e) )
		Mw_3_index.append( (e, e) )
		Mw_3_values.append( G3[s][t]["distance"] )
		f3_norm += 2 * G3[s][t]["distance"]
	#end for
	f3 = nx.shortest_path_length( G3, S, T, "distance" )
	g_m = len( G1.edges )
	l1= lambda l: [1 for _ in l]
	Ms_1 = [Ms_1_index,l1(Ms_1_index),(g_n,g_m)]
	Mt_1 = [Mt_1_index,l1(Mt_1_index),(g_n,g_m)]
	Mw_1 = [Mw_1_index,Mw_1_values,(g_m,g_m)]
	Ms_2 = [Ms_2_index,l1(Ms_2_index),(g_n,g_m)]
	Mt_2 = [Mt_2_index,l1(Mt_2_index),(g_n,g_m)]
	Mw_2 = [Mw_2_index,Mw_2_values,(g_m,g_m)]
	Ms_3 = [Ms_3_index,l1(Ms_3_index),(g_n,g_m)]
	Mt_3 = [Mt_3_index,l1(Mt_3_index),(g_n,g_m)]
	Mw_3 = [Mw_3_index,Mw_3_values,(g_m,g_m)]
	S_mat = [[(S,S)],[1],(g_n,g_n)]
	T_mat = [[(T,T)],[1],(g_n,g_n)]
	f1_norm = f1_norm if f1_norm > 0 else 1
	f2_norm = f2_norm if f2_norm > 0 else 1
	f3_norm = f3_norm if f3_norm > 0 else 1
	return ((Ms_1,Mt_1,Mw_1,S_mat,T_mat),f1/f1_norm), ((Ms_2,Mt_2,Mw_2,S_mat,T_mat),f2/f2_norm), ((Ms_3,Mt_3,Mw_3,S_mat,T_mat),f3/f3_norm)
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
	batch_Ms_index = []
	batch_Ms_value = []
	batch_Mt_index = []
	batch_Mt_value = []
	batch_Mw_index = []
	batch_Mw_value = []
	batch_S_index = []
	batch_S_value = []
	batch_T_index = []
	batch_T_value = []
	for p in problems:
		Ms, Mt, Mw, S, T = p
		for i, v in reindex_matrix( n, m, Ms ):
			batch_Ms_index.append( i )
			batch_Ms_value.append( v )
		#end for
		for i, v in reindex_matrix( n, m, Mt ):
			batch_Mt_index.append( i )
			batch_Mt_value.append( v )
		#end for
		for i, v in reindex_matrix( m, m, Mw ):
			batch_Mw_index.append( i )
			batch_Mw_value.append( v )
		#end for
		for i, v in reindex_matrix( n, n, S ):
			batch_S_index.append( i )
			batch_S_value.append( v )
		#end for
		for i, v in reindex_matrix( n, n, T ):
			batch_T_index.append( i )
			batch_T_value.append( v )
		#end for
		n += Ms[2][0]
		m += Ms[2][1]
	#end for
	Ms = [batch_Ms_index,batch_Ms_value,(n,m)]
	Mt = [batch_Mt_index,batch_Mt_value,(n,m)]
	Mw = [batch_Mw_index,batch_Mw_value,(m,m)]
	S = [batch_S_index,batch_S_value,(n,n)]
	T = [batch_T_index,batch_T_value,(n,n)]
	return (Ms,Mt,Mw,S,T)
#end create_batch

if __name__ == '__main__':
	d = 64
	epochs = 100
	batch_n_max = 4096
	batches_per_epoch = 32
	n_size_min = 8
	n_loss_increase_threshold = 0.01
	n_size_max = 64
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
			epoch_allowed_flow_error = 0
			epoch_n = 0
			epoch_m = 0
			#for b, batch in itertools.islice( enumerate( instance_generator.get_batches( batch_size ) ), batches_per_epoch ):
			for batch_i in range( batches_per_epoch ):
				batch_n_size = np.random.randint( n_size_min, n_size+1 )
				n_acc = 0
				max_n = 0
				instances = 0
				Gs = []
				flows = []
				while True:
					g_n = np.random.randint( batch_n_size//2, batch_n_size*2 )
					if n_acc + g_n * 3 < batch_n_max:
						n_acc += g_n * 3
						instances += 3
						max_n = max( max_n, g_n )
						(g1,f1),(g2,f2),(g3,f3) = create_graph( g_n, edge_probability )
						Gs = Gs + [g1,g2,g3]
						flows = flows + [f1,f2,f3]
					else:
						break
					#end if
				#end for
				Ms, Mt, Mw, S, T = create_batch( Gs )
				targets = [ t for (t,_) in T[0] ]
				time_steps = max_n
				batch_allowed_flow_error = sum( flows ) * n_loss_increase_threshold
				n = Ms[2][0]
				m = Ms[2][1]
				

				_, loss = sess.run(
					[ GNN["train_step"], GNN["loss"] ],
					feed_dict = {
						GNN["gnn"].matrix_placeholders["Ms"]: Ms,
						GNN["gnn"].matrix_placeholders["Mt"]: Mt,
						GNN["gnn"].matrix_placeholders["Mw"]: Mw,
						GNN["gnn"].matrix_placeholders["S"]: S,
						GNN["gnn"].time_steps: time_steps,
						GNN["instance_val"]: flows,
						GNN["instance_target"]: targets
					}
				)
				
				epoch_loss += loss
				epoch_allowed_flow_error += batch_allowed_flow_error
				epoch_n += n
				epoch_m += m
				
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| (Loss,Allowed,Loss/Allowed): ({loss:.5f},{allowed:.5f},{good:.5f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = batch_i,
						loss = loss,
						allowed = batch_allowed_flow_error,
						good = loss / batch_allowed_flow_error,
						n = n,
						m = m,
						i = instances
					),
					flush = True
				)
			#end for
			# Summarize Epoch
			epoch_loss = epoch_loss / batches_per_epoch
			epoch_allowed_flow_error = epoch_allowed_flow_error / batches_per_epoch
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Mean (Loss,Allowed,Loss/Allowed): ({loss:.5f},{allowed:.5f},{good:.5f})".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "all",
					loss = epoch_loss,
					allowed = epoch_allowed_flow_error,
					good = epoch_loss / epoch_allowed_flow_error,
					n = epoch_n,
					m = epoch_m,
				),
				flush = True
			)
			if epoch_loss < epoch_allowed_flow_error and n_size < n_size_max:
				n_size = 2 * n_size
				n_size = n_size_max if n_size > n_size_max else n_size
			#end if
			
			# TEST TIME
			# Create random graphs
			n_acc = 0
			max_n = 0
			instances = 0
			Gs = []
			flows = []
			while True:
				g_n = np.random.randint( n_size, n_size*2 )
				if n_acc + g_n < batch_n_max:
					n_acc += g_n
					instances += 1
					max_n = max( max_n, g_n )
					(g1,f1),_,_ = create_graph( g_n, edge_probability )
					Gs.append( g1 )
					flows.append( f1 )
				else:
					break
				#end if
			#end for
			Ms, Mt, Mw, S, T = create_batch( Gs )
			targets = [ t for (t,_) in T[0] ]
			time_steps = max_n
			test_allowed_flow_error = sum( flows ) * n_loss_increase_threshold
			test_n = Ms[2][0]
			test_m = Ms[2][1]

			test_loss = sess.run(
				GNN["loss"],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["Ms"]: Ms,
					GNN["gnn"].matrix_placeholders["Mt"]: Mt,
					GNN["gnn"].matrix_placeholders["Mw"]: Mw,
					GNN["gnn"].matrix_placeholders["S"]: S,
					GNN["gnn"].time_steps: time_steps,
					GNN["instance_val"]: flows,
					GNN["instance_target"]: targets
				}
			)
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| Test (Loss,Allowed,Loss/Allowed): ({loss:.5f},{allowed:.5f},{good:.5f})".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "tst",
					loss = test_loss,
					allowed = test_allowed_flow_error,
					good = test_loss / test_allowed_flow_error,
					n = test_n,
					m = test_m,
					i = instances
				),
				flush = True
			)
			
	#end Session
