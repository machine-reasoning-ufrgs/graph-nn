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
from dijkstra_util import create_graph, create_batch

def build_network(d):

	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define GNN dictionary
	GNN = {}

	# Define placeholder for result values (one per problem)
	instance_val = tf.placeholder( tf.float32, [ None ], name = "instance_val" )
	instance_m_list = tf.placeholder( tf.int32, [ None ], name = "instance_edge_num" )
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
			"T": ("N","N"), # Matrix indicating whether a node is the target
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
				},
				{
					"mat": "T"
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

	# Define L_vote
	E_vote_MLP = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "E_vote",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
		)

	# Compute the number of variables
	m = tf.shape( gnn.matrix_placeholders["Mw"] )[0]
	# Compute number of problems
	p = tf.shape( instance_val )[0]

	# Get the last embeddings
	E_n = gnn.last_states["E"].h
	E_vote_logits = E_vote_MLP( E_n )
	E_vote = tf.nn.sigmoid( E_vote_logits )
	E_objective = tf.sparse_tensor_dense_matmul( gnn.matrix_placeholders["Mw"], E_vote )

	# Reorganize votes' result to obtain a prediction for each problem instance
	def _vote_while_cond(i, m_acc, predicted_val):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, m_acc, predicted_val):
		# Helper for the amount of edges in this problem
		i_m = instance_m_list[i]
		# Gather the edges of that problem
		obj_vals = tf.gather( E_objective, tf.range( m_acc, tf.add( m_acc, i_m ) ) )
		problem_predicted_val = tf.reduce_sum( obj_vals )
		# Update TensorArray
		predicted_val = predicted_val.write( i, problem_predicted_val )
		return tf.add( i, tf.constant( 1 ) ), tf.add( m_acc, i_m ), predicted_val
	#end _vote_while_body
			
	predicted_val = tf.TensorArray( size = p, dtype = tf.float32 )
	_, _, predicted_val = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), tf.constant( 0, dtype = tf.int32 ), predicted_val ]
	)
	predicted_val = predicted_val.stack()

	# Define loss and %error
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
	GNN["instance_m"] = instance_m_list
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
	batch_n_max = 2048
	batches_per_epoch = 32
	n_size_min = 16
	n_loss_increase_threshold = 0.01
	n_size_max = 128
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
			#for b, batch in itertools.islice( enumerate( instance_generator.get_batches( batch_size ) ), batches_per_epoch ):
			for batch_i in range( batches_per_epoch ):
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
						(g1,f1),(g2,f2),_ = create_graph( g_n, edge_probability )
						Gs = Gs + [g1,g2]
						distances = distances + [f1,f2]
					else:
						break
					#end if
				#end for
				Ms, Mt, Mw, S, T, _, mlist, _ = create_batch( Gs )
				targets = [ t for (t,_) in T[0] ]
				time_steps = max_n
				batch_allowed_error = sum( distances ) * n_loss_increase_threshold
				n = Ms[2][0]
				m = Ms[2][1]
				

				_, loss, err, abserr = sess.run(
					[ GNN["train_step"], GNN["loss"], GNN["%error"], GNN["%abserror"] ],
					feed_dict = {
						GNN["gnn"].matrix_placeholders["Ms"]: Ms,
						GNN["gnn"].matrix_placeholders["Mt"]: Mt,
						GNN["gnn"].matrix_placeholders["Mw"]: Mw,
						GNN["gnn"].matrix_placeholders["S"]: S,
						GNN["gnn"].matrix_placeholders["T"]: T,
						GNN["gnn"].time_steps: time_steps,
						GNN["instance_val"]: distances,
						GNN["instance_m"]: mlist,
						GNN["instance_target"]: targets
					}
				)
				
				epoch_loss += loss
				epoch_err += err
				epoch_abserr += abserr
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
			Ms, Mt, Mw, S, T, _, mlist, _ = create_batch( Gs )
			targets = [ t for (t,_) in T[0] ]
			time_steps = max_n
			test_allowed_error = sum( distances ) * n_loss_increase_threshold
			test_n = Ms[2][0]
			test_m = Ms[2][1]

			test_loss, test_err, test_abserr = sess.run(
				[GNN["loss"],GNN["%error"],GNN["%abserror"]],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["Ms"]: Ms,
					GNN["gnn"].matrix_placeholders["Mt"]: Mt,
					GNN["gnn"].matrix_placeholders["Mw"]: Mw,
					GNN["gnn"].matrix_placeholders["S"]: S,
					GNN["gnn"].matrix_placeholders["T"]: T,
					GNN["gnn"].time_steps: time_steps,
					GNN["instance_val"]: distances,
					GNN["instance_m"]: mlist,
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
					error = test_err,
					abserror = test_abserr,
					n = test_n,
					m = test_m,
					i = instances
				),
				flush = True
			)
			
	#end Session
