import sys, os, time
import tensorflow as tf
import numpy as np
import random
from constraint import *
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import model builder
from graphnn import GraphNN
from mlp import Mlp
from instance_loader import InstanceLoader
# Import tools
import itertools

def timestamp():
	return time.strftime( "%Y%m%d%H%M%S", time.gmtime() )
#end timestamp

def memory_usage():
	pid=os.getpid()
	s = next( line for line in open( '/proc/{}/status'.format( pid ) ).read().splitlines() if line.startswith( 'VmSize' ) ).split()
	return "{} {}".format( s[-2], s[-1] )
#end memory_usage

def build_network(d):

	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define GNN dictionary
	GNN = {}

	# Define placeholder for result values (one per problem)
	instance_colorability = tf.placeholder( tf.float32, [ None ], name = "instance_colorability" )

	# Define Graph neural network
	gnn = GraphNN(
		{
			"V": d#,
			#"C": d
		},
		{
			"M_VV": ("V","V")#,
			#"M_VC": {"vars": ("V","C"), "compute?": False}
			#"M_VC_mask": {"vars": ("V","C"), "compute?": False}
		},
		{
			#"Cmsg": ("C","V"),
			#"Vmsg": ("V","C")
		},
		{
			"V": [
				{
					"mat": "M_VV",
					"var": "V"
				}#,
				#{
				#	"mat": "M_VC",
				#	"var": "C",
				#	"msg": "Cmsg"
				#}
			]#,
			#"C": [
			#	{
			#		"var": "C"
			#	},
			#	{
			#		"mat": "M_VC",
			#		"transpose?": True,
			#		"var": "V",
			#		"msg": "Vmsg"
			#	}
			#]
		},
		name="Coloring",
	)

	# Define V_vote
	V_vote_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "V_vote",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
		)

	# Compute number of problems
	p = tf.shape( instance_colorability )[0]

	# Placeholder for the list of number of vertices per instance
	n_vertices = tf.placeholder( tf.int32, shape = (None,), name = "n_vertices" )

	# Get the last embeddings
	V_last = gnn.last_states["V"].h
	V_vote = V_vote_MLP( V_last )

	def _vote_while_body(i, n_acc, predicted_colorability):
		# Helper for the amount of variables in this problem
		i_n = n_vertices[i]
		# Gather the vertices for that problem
		vertices = tf.gather( V_vote, tf.range( n_acc, tf.add( n_acc, i_n ) ) )
		# Concatenate positive and negative literals and average their vote values
		problem_predicted_colorability = tf.reduce_mean(vertices)
		# Update TensorArray
		predicted_colorability = predicted_colorability.write(i, problem_predicted_colorability)
		return tf.add( i, tf.constant( 1 ) ), tf.add( n_acc, i_n ), predicted_colorability
	#end

	def _vote_while_cond(i, n_acc, predicted_colorability):
		return tf.less( i, p )
	#end

	predicted_colorability = tf.TensorArray( size = p, dtype = tf.float32 )
	_, _, predicted_colorability = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), tf.constant(0, dtype=tf.int32), predicted_colorability ]
	)
	predicted_colorability = predicted_colorability.stack()

	# Define loss, optimizer, train step
	predict_costs 	= tf.nn.sigmoid_cross_entropy_with_logits( labels = instance_colorability, logits = predicted_colorability )
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
					tf.cast(instance_colorability, tf.bool ),
					tf.cast( tf.round( tf.nn.sigmoid( predicted_colorability ) ), tf.bool )
				)
				, tf.float32
			)
		)
	
	GNN["gnn"] 						= gnn
	#GNN["k"]						= gnn.num_vars["C"]
	GNN["n_vertices"]				= n_vertices
	GNN["instance_colorability"] 	= instance_colorability
	GNN["predicted_colorability"] 	= tf.nn.sigmoid( predicted_colorability )
	GNN["loss"] 					= loss
	GNN["acc"]						= acc
	GNN["train_step"] 				= train_step
	return GNN
#end build_networks

def dense_to_sparse( M ):
	n, m = M.shape
	M_i = []
	M_v = []
	M_shape = (n,m)
	for i in range( n ):
		for j in range( m ):
			if M[i,j] == 1:
				M_i.append( (i,j ) )
				M_v.append( 1 )
			#end if
		#end for
	#end for
	return (M_i,M_v,M_shape)
#end dense_to_sparse

if __name__ == '__main__':
	d 					= 128
	epochs 				= 100
	batch_size			= 32
	batches_per_epoch 	= 128
	time_steps 			= 200

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(d)

	# Create train and test instance generators
	train_generator = InstanceLoader("train")
	test_generator = InstanceLoader("test")

	# Disallow GPU use
	config = tf.ConfigProto( device_count = {"GPU":0})
	with tf.Session(config=config) as sess:
		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )

		# Run for a number of epochs
		print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
		for epoch in range( epochs ):
			# Reset generator
			train_generator.reset()
			# Run train batches
			epoch_loss = 0.0
			epoch_acc = 0.0
			for (batch_i,batch) in enumerate(itertools.islice(train_generator.get_batches(batch_size), batches_per_epoch)):

				M_VV, n_vertices, k, k_colorable = batch

				n_total = M_VV.shape[0]
				k_max = max(k)

				M_VC = np.zeros((n_total,k_max))
				for i in range(batch_size):
					n_acc = sum(n_vertices[:i])
					M_VC[n_acc:n_acc+n_vertices[i], 0:k[i]] = 1
				#end

				_, loss, acc, predictions = sess.run(
					[ GNN["train_step"], GNN["loss"], GNN["acc"], GNN["predicted_colorability"] ],
					feed_dict = {
						#GNN["k"]: k,
						GNN["n_vertices"]: n_vertices,
						GNN["gnn"].matrix_placeholders["M_VV"]: dense_to_sparse(M_VV),
						#GNN["gnn"].matrix_placeholders["M_VC"]: dense_to_sparse(M_VC),
						GNN["gnn"].time_steps: time_steps,
						GNN["instance_colorability"]: k_colorable
					}
				)
				
				epoch_loss += loss
				epoch_acc += acc
				
				#print(np.round(predictions,3))
				#print(k_colorable)
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (|V|,|E|,instances): ({V},{E},{i})\t| (Loss,Acc): ({loss:.5f},{acc:.5f})".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = batch_i,
						loss = loss,
						acc = acc,
						V = M_VV.shape[0],
						E = len(np.nonzero(M_VV)[0]),
						i = batch_size
					),
					flush = True
				)
			#end for
			# Summarize Epoch
			epoch_loss = epoch_loss / batches_per_epoch
			epoch_acc = epoch_acc / batches_per_epoch
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\t| Train Mean (Loss,Acc): ({loss:.5f},{acc:.5f})".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "all",
					loss = epoch_loss,
					V = n,
					E = len(np.nonzero(M_VV)[0]),
					acc = epoch_acc
				),
				flush = True
			)

			# Reset generator
			test_generator.reset()
			# Run test batches
			epoch_loss = 0.0
			epoch_acc = 0.0
			for (batch_i,batch) in enumerate(itertools.islice(test_generator.get_batches(batch_size), batches_per_epoch)):

				M_VV, n_vertices, k, k_colorable = batch

				n_total = M_VV.shape[0]
				k_max = max(k)

				M_VC = np.zeros((n_total,k_max))
				for i in range(batch_size):
					n_acc = sum(n_vertices[:i])
					M_VC[n_acc:n_acc+n_vertices[i], 0:k[i]] = 1
				#end

				loss, acc = sess.run(
					[ GNN["loss"], GNN["acc"] ],
					feed_dict = {
						#GNN["k"]: k,
						GNN["n_vertices"]: n_vertices,
						GNN["gnn"].matrix_placeholders["M_VV"]: dense_to_sparse(M_VV),
						#GNN["gnn"].matrix_placeholders["M_VC"]: dense_to_sparse(M_VC),
						GNN["gnn"].time_steps: time_steps,
						GNN["instance_colorability"]: k_colorable
					}
				)
				
				epoch_loss += loss
				epoch_acc += acc
			#end for
			# Summarize Epoch
			epoch_loss = epoch_loss / batches_per_epoch
			epoch_acc = epoch_acc / batches_per_epoch
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\t| Test Mean (Loss,Acc): ({loss:.5f},{acc:.5f})".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = "all",
					loss = epoch_loss,
					V = n,
					E = len(np.nonzero(M_VV)[0]),
					acc = epoch_acc
				),
				flush = True
			)
			
	#end Session
#end