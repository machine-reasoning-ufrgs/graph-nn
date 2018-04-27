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
			"S": ("N","N"),
			"T": ("N","N")
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
				},
				{
					"mat": "T"
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
	n = tf.floordiv( tf.shape( gnn.matrix_placeholders["M"] )[0], tf.constant( 2 ) )
	# Compute number of problems
	p = tf.shape( instance_val )[0]
	# Compute number of variables per instance
	num_vars_on_instance = tf.placeholder( tf.int32, [ None ], name = "instance_n" )

	# Get the last embeddings
	N_n = gnn.last_states["N"].h
	N_vote = N_vote_MLP( N_n )

	# Reorganize votes' result to obtain a prediction for each problem instance

	def _vote_while_cond(i, p, n_acc, n, n_var_list, predicted_val, N_vote):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, p, n_acc, n, n_var_list, predicted_val, N_vote):
		# Helper for the amount of variables in this problem
		i_n = n_var_list[i]
		# Gather the positive and negative literals for that problem
		final_node = N_vote[ tf.subtract( tf.add( n_acc, i_n ), 1 ) ]
		# Concatenate positive and negative literals and average their vote values
		problem_predicted_val = tf.reshape( final_node, shape = [] )
		# Update TensorArray
		predicted_val = predicted_val.write( i, problem_predicted_val )
		return tf.add( i, tf.constant( 1 ) ), p, tf.add( n_acc, i_n ), n, n_var_list, predicted_val, N_vote
	#end _vote_while_body
			
	predicted_val = tf.TensorArray( size = p, dtype = tf.float32 )
	_, _, _, _, _, predicted_val, _ = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), p, tf.constant( 0, dtype = tf.int32 ), n, num_vars_on_instance, predicted_val, N_vote ]
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
	GNN["predicted_val"] = predicted_val
	GNN["num_vars_on_instance"] = num_vars_on_instance
	GNN["loss"] = loss
	GNN["train_step"] = train_step

	return GNN
#end build_network

if __name__ == '__main__':

	d = 64
	epochs = 100
	batch_size = 32
	batches_per_epoch = 256
	n_size_min = 16
	n_loss_increase_threshold = 0.01
	n_size_max = 128
	test_n = 512
	edge_probability = 0.25

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(d)

	with tf.Session() as sess:
		
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
			epoch_n = 0
			epoch_m = 0
			epoch_allowed_flow_error = 0
			#for b, batch in itertools.islice( enumerate( instance_generator.get_batches( batch_size ) ), batches_per_epoch ):
			for b in range( batches_per_epoch ):
				batch_n_size = np.random.randint( n_size_min, n_size+1 )
				max_n = 0
				m = 0
				n = 0
				batch_allowed_flow_error = 0
				S_index = []
				T_index = []
				M_index = []
				M_values = []
				flows = []
				n_vars = []
				g_i = 0
				while g_i < batch_size:
					# Create a random graph
					g_n = np.random.randint( batch_n_size//2, batch_n_size )
					max_n = max( max_n, g_n )
					G = nx.fast_gnp_random_graph( g_n, edge_probability )
					for s, t in G.edges:
						G[s][t]["capacity"] = np.random.rand()
						M_index.append( ( n + s, n + t ) )
						M_values.append( G[s][t]["capacity"] )
						M_index.append( ( n + t, n + s ) )
						M_values.append( G[t][s]["capacity"] )
					#end for
					S_index.append( (n, n) )
					T_index.append( (n + g_n - 1, n + g_n - 1) )
					flow = nx.maximum_flow_value( G, 0, g_n-1 )
					batch_allowed_flow_error += flow * n_loss_increase_threshold
					flows.append( flow )
					n_vars.append( g_n )
					n += g_n
					m += len( G.edges )
					g_i += 1
					# Then create a complementary graph
					G.add_edge( 0, g_n-1, capacity = 1.0 )
					for s, t in G.edges:
						M_index.append( ( n + s, n + t ) )
						M_values.append( G[s][t]["capacity"] )
						M_index.append( ( n + t, n + s ) )
						M_values.append( G[t][s]["capacity"] )
					#end for
					S_index.append( (n, n) )
					T_index.append( (n + g_n - 1, n + g_n - 1) )
					flow = nx.maximum_flow_value( G, 0, g_n-1 )
					batch_allowed_flow_error += flow * n_loss_increase_threshold
					flows.append( flow )
					n_vars.append( g_n )
					n += g_n
					m += len( G.edges )
					g_i += 1
				#end for
				M_shape = (n,n)
				M = (M_index, M_values, M_shape)
				S = (S_index, [1 for _ in S_index], M_shape)
				T = (T_index, [1 for _ in T_index], M_shape)
				time_steps = max_n

				_, loss = sess.run(
					[ GNN["train_step"], GNN["loss"] ],
					feed_dict = {
						GNN["gnn"].matrix_placeholders["M"]: M,
						GNN["gnn"].matrix_placeholders["S"]: S,
						GNN["gnn"].matrix_placeholders["T"]: T,
						GNN["gnn"].time_steps: time_steps,
						GNN["instance_val"]: flows,
						GNN["num_vars_on_instance"]: n_vars
					}
				)
				
				epoch_loss += loss
				epoch_allowed_flow_error += batch_allowed_flow_error
				epoch_n += n
				epoch_m += m
				
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| (Loss,Allowed,Loss/Allowed): ({loss:.5f},{allowed:.5f},{good:.5f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = b,
						loss = loss,
						allowed = batch_allowed_flow_error,
						good = loss / batch_allowed_flow_error,
						n = n,
						m = m,
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
			# Create a random graph
			G = nx.fast_gnp_random_graph( test_n, edge_probability )
			flows = []
			n_vars = []
			S_index = []
			T_index = []
			M_index = []
			M_values = []
			for s, t in G.edges:
				G[s][t]["capacity"] = np.random.rand()
				M_index.append( ( s, t ) )
				M_values.append( G[s][t]["capacity"] )
				M_index.append( ( t, s ) )
				M_values.append( G[t][s]["capacity"] )
			#end for
			S_index.append( (0, 0) )
			T_index.append( (test_n - 1, test_n - 1) )
			flow = nx.maximum_flow_value( G, 0, test_n-1 )
			test_allowed_flow_error = flow * n_loss_increase_threshold
			flows.append( flow )
			n_vars.append( test_n )
			test_m = len( G.edges )
			M_shape = (test_n,test_n)
			M = (M_index, M_values, M_shape)
			S = (S_index, [1 for _ in S_index], M_shape)
			T = (T_index, [1 for _ in T_index], M_shape)
			time_steps = test_n

			test_loss = sess.run(
				[ GNN["loss"] ],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["M"]: M,
					GNN["gnn"].matrix_placeholders["S"]: S,
					GNN["gnn"].matrix_placeholders["T"]: T,
					GNN["gnn"].time_steps: time_steps,
					GNN["instance_val"]: flows,
					GNN["num_vars_on_instance"]: n_vars
				}
			)
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Test (Loss,Allowed,Loss/Allowed): ({loss:.5f},{allowed:.5f},{good:.5f})".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "tst",
					loss = test_loss,
					allowed = test_allowed_flow_error,
					good = test_loss / test_allowed_flow_error,
					n = test_n,
					m = test_m,
				),
				flush = True
			)
			
	#end Session
