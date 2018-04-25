
import sys, os, time
import tensorflow as tf
import numpy as np

from graphnn import GraphNN, Mlp
from cnf import CNF
import instance_loader
import itertools
from cnf import create_batchCNF

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

def build_neurosat(d):

	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define neurosat dictionary
	neurosat = {}

	# Define placeholder for satisfiability statuses (one per problem)
	instance_SAT = tf.placeholder( tf.float32, [ None ], name = "instance_SAT" )

	# Define Graph neural network
	gnn = GraphNN(
		{
			"L": d,
			"C": d
		},
		{
			"M": ("L","C"),
			"Inv": ("L","L")
		},
		{
			"Lmsg": ("L","C"),
			"Cmsg": ("C","L")
		},
		{
			"L": [("Inv",None,"L"),("M","Cmsg","C")],
			"C": [("M","Lmsg","L",True)]
		},
		name="NeuroSAT"
		)

	# Define L_vote
	L_vote_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "L_vote",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
		)

	# Compute the number of variables
	n = tf.floordiv( tf.shape( gnn.matrix_placeholders["M"] )[0], tf.constant( 2 ) )
	# Compute number of problems
	p = tf.shape( instance_SAT )[0]
	# Compute number of variables per instance
	num_vars_on_instance = tf.placeholder( tf.int32, [ None ], name = "instance_n" )

	# Get the last embeddings
	L_n = gnn.last_states["L"].h
	L_vote = L_vote_MLP( L_n )

	# Reorganize votes' result to obtain a prediction for each problem instance

	def _vote_while_cond(i, p, n_acc, n, n_var_list, predicted_sat, L_vote):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, p, n_acc, n, n_var_list, predicted_SAT, L_vote):
		# Helper for the amount of variables in this problem
		i_n = n_var_list[i]
		# Gather the positive and negative literals for that problem
		pos_lits = tf.gather( L_vote, tf.range( n_acc, tf.add( n_acc, i_n ) ) )
		neg_lits = tf.gather( L_vote, tf.range( tf.add( n, n_acc ), tf.add( n, tf.add( n_acc, i_n ) ) ) )
		# Concatenate positive and negative literals and average their vote values
		problem_predicted_SAT = tf.reduce_mean( tf.concat( [pos_lits, neg_lits], axis = 1 ) )
		# Update TensorArray
		predicted_SAT = predicted_SAT.write( i, problem_predicted_SAT )
		return tf.add( i, tf.constant( 1 ) ), p, tf.add( n_acc, i_n ), n, n_var_list, predicted_SAT, L_vote
	#end _vote_while_body
			
	predicted_SAT = tf.TensorArray( size = p, dtype = tf.float32 )
	_, _, _, _, _, predicted_SAT, _ = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), p, tf.constant( 0, dtype = tf.int32 ), n, num_vars_on_instance, predicted_SAT, L_vote ]
	)
	predicted_SAT = predicted_SAT.stack()

	# Define loss, accuracy
	predict_costs = tf.nn.sigmoid_cross_entropy_with_logits( labels = instance_SAT, logits = predicted_SAT )
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
	
	accuracy = tf.reduce_mean(
		tf.cast(
			tf.equal(
				tf.cast( instance_SAT, tf.bool ),
				tf.cast( tf.round( tf.nn.sigmoid( predicted_SAT ) ), tf.bool )
			)
			, tf.float32
		)
	)

	neurosat["gnn"] 					= gnn
	neurosat["instance_SAT"]			= instance_SAT
	neurosat["predicted_SAT"] 			= predicted_SAT
	neurosat["num_vars_on_instance"] 	= num_vars_on_instance
	neurosat["loss"] 					= loss
	neurosat["accuracy"] 				= accuracy

	return neurosat
#end build_neurosat

if __name__ == '__main__':

	d 					= 128
	epochs 				= 20
	batch_size 			= 64
	batches_per_epoch 	= 128
	timesteps 			= 26

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format	( timestamp = timestamp(), memory = memory_usage() ) )
	neurosat = build_neurosat(d)

	# Create batch loader
	print( "{timestamp}\t{memory}\tLoading instances ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	instance_generator = instance_loader.InstanceLoader( "./instances" )
	test_instance_generator = instance_loader.InstanceLoader( "./test_instances" )

	with tf.Session() as sess:
		
		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )

		# Run for a number of epochs
		print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
		for epoch in range( epochs ):
			# Run batches
			instance_generator.reset()
			epoch_loss = 0.0
			epoch_accuracy = 0.0
			epoch_n = 0
			epoch_m = 0
			for b, I in itertools.islice( enumerate( instance_generator.get_batches( batch_size ) ), batches_per_epoch ):

				I_sat = np.array( list( 1 if sat else 0 for sat in I.sat ) )
				I_n = np.array(I.n)
				M = I.get_sparse_matrix()

				n, m =  M[2][0]//2, M[2][1]

				Inv = (
					[ (i,n+i) for i in range(n) ] + [ (n+i,i) for i in range(n) ],
					[ 1 for i in range(2*n)],
					(2*n,2*n)
					)

				loss, accuracy = sess.run( [neurosat["loss"], neurosat["accuracy"]], feed_dict={
					neurosat["gnn"].matrix_placeholders["M"]:	M,
					neurosat["gnn"].matrix_placeholders["Inv"]: Inv,
					neurosat["gnn"].time_steps: 				timesteps,
					neurosat["instance_SAT"]:					I_sat,
					neurosat["num_vars_on_instance"]: 			I_n
					} )
					
				epoch_loss += loss
				epoch_accuracy += accuracy
				epoch_n += n
				epoch_m += m
				
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m}) | Solver (Loss, Acc): ({loss:.5f}, {accuracy:.5f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = b,
						loss = loss,
						accuracy = accuracy,
						n = n,
						m = m,
					),
					flush = True
				)
			#end for
			# Summarize Epoch
			epoch_loss = epoch_loss / batches_per_epoch
			epoch_accuracy = epoch_accuracy / batches_per_epoch
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m}) | Solver (Mean Loss, Mean Acc): ({loss:.5f}, {accuracy:.5f})".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "all",
					loss = epoch_loss,
					accuracy = epoch_accuracy,
					n = epoch_n,
					m = epoch_m,
				),
				flush = True
			)
		#end for
	#end Session
