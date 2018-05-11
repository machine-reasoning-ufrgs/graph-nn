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
from util import timestamp, memory_usage
from dijkstra_util import create_graph_nonquiver as create_graph, create_batch_nonquiver as create_batch

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
		float_dtype = tf.float32
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
	# %Error
	abserror = tf.reduce_mean( tf.divide( tf.abs( tf.subtract( instance_val, predicted_val ) ), predicted_val ) )
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
	GNN["%abserror"] = abserror
	GNN["train_step"] = train_step
	GNN["nop"] = tf.no_op()
	return GNN
#end build_network

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
			epoch_abserr = 0
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
					if n_acc + g_n * 2 < batch_n_max:
						n_acc += g_n * 2
						instances += 2
						max_n = max( max_n, g_n )
						(g1,f1),(g2,f2),_ = create_graph( g_n, edge_probability, normalize = True )
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

				_, loss, err, abserr = sess.run(
					[ GNN["train_step"], GNN["loss"], GNN["%error"], GNN["%abserror"] ],
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
				epoch_abserr += err
				epoch_n += n
				epoch_m += m
				
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| (Loss,%Error|%Error|): ({loss:.5f},{error:.5f}|{abserror:.5f}|)".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = batch_i,
						loss = loss,
						error = err,
						abserror = abserr,
						n = n,
						m = m,
						i = instances
					),
					flush = True
				)
			#end for
			# Summarize Epoch
			epoch_loss /= batches_per_epoch
			epoch_err /= batches_per_epoch
			epoch_abserr /= batches_per_epoch
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Mean (Loss,%Error|%Error|): ({loss:.5f},{error:.5f}|{abserror:.5f}|)".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "all",
					loss = epoch_loss,
					error = epoch_err,
					abserror = epoch_abserr,
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
					(g1,f1),(g2,f2),_ = create_graph( g_n, edge_probability )
					g,f = np.random.choice( [(g1,f1),(g2,f2)] )
					Gs.append( g )
					distances.append( f )
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
			
			test_loss, test_err, test_abserr = sess.run(
				[GNN["loss"],GNN["%error"],GNN["%abserror"]],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["M"]: M,
					GNN["gnn"].matrix_placeholders["S"]: S,
					GNN["gnn"].time_steps: time_steps,
					GNN["instance_val"]: distances,
					GNN["instance_target"]: targets
				}
			)
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| Test (Loss,%Error|%Error|): ({loss:.5f},{error:.5f}|{abserror:.5f}|)".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "tst",
					loss = test_loss,
					abserror = test_abserr,
					error = test_err,
					n = test_n,
					m = test_m,
					i = instances
				),
				flush = True
			)
			
	#end Session
