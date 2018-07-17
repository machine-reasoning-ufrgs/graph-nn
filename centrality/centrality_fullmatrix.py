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
from instance_loader import InstanceLoader

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
	labels_deg = tf.placeholder( tf.float32, [ None, None ], name = "labels_deg" )
	labels_bet = tf.placeholder( tf.float32, [ None, None ], name = "labels_bet" )
	labels_clo = tf.placeholder( tf.float32, [ None, None ], name = "labels_clo" )
	labels_eig = tf.placeholder( tf.float32, [ None, None ], name = "labels_eig" )
	
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
# Define votes
	degree_MLP = Mlp(
		layer_sizes = [ 2*d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "degree_MLP",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
	)
	betweenness_MLP = Mlp(
		layer_sizes = [ 2*d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "betweenness_MLP",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
	)
	closeness_MLP = Mlp(
		layer_sizes = [ 2*d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = 1,
		name = "closeness_MLP",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
	)
	eigenvector_MLP = Mlp(
		layer_sizes = [ 2*d for _ in range(2) ],
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
	p = tf.shape( nodes_n )[0]

	# Get the last embeddings
	N_n = gnn.last_states["N"].h
	#print("N_n shape:" +str(N_n.shape))
	  

	# Reorganize votes' result to obtain a prediction for each problem instance
	def _vote_while_cond(i, arr_acc_deg, arr_cost_deg, arr_acc_bet, arr_cost_bet, arr_acc_clo, arr_cost_clo, arr_acc_eig, arr_cost_eig, n_acc):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, arr_acc_deg, arr_cost_deg, arr_acc_bet, arr_cost_bet, arr_acc_clo, arr_cost_clo, arr_acc_eig, arr_cost_eig, n_acc):
		# Gather the embeddings for that problem
		#p_embeddings = tf.gather(N_n, tf.range( n_acc, tf.add(n_acc, nodes_n[i]) ))
		p_embeddings = tf.slice( N_n, [n_acc, 0], [nodes_n[i], d]) 
		
		N_expanded = tf.expand_dims(p_embeddings, 0)
		N1 = tf.tile(N_expanded,(nodes_n[i],1,1))
		
		N_transposed = tf.transpose(N_expanded, (1,0,2))
		N2 = tf.tile(N_transposed, (1,nodes_n[i],1))
		
		N1N2 = tf.concat([N1,N2], 2)
		 
		pred_deg_matrix = tf.squeeze( degree_MLP( N1N2 ) )
		pred_bet_matrix = tf.squeeze( betweenness_MLP( N1N2 ) )
		pred_clo_matrix = tf.squeeze( closeness_MLP( N1N2 ) )
		pred_eig_matrix = tf.squeeze( eigenvector_MLP( N1N2 ) )
		
		pred_deg_matrix_prob = tf.round( tf.sigmoid( pred_deg_matrix ) )
		pred_bet_matrix_prob = tf.round( tf.sigmoid( pred_bet_matrix ) )
		pred_clo_matrix_prob = tf.round( tf.sigmoid( pred_clo_matrix ) )
		pred_eig_matrix_prob = tf.round( tf.sigmoid( pred_eig_matrix ) )
		
		

		p_labels_deg = tf.slice( labels_deg, [n_acc, n_acc], [nodes_n[i], nodes_n[i]])
		p_labels_bet = tf.slice( labels_bet, [n_acc, n_acc], [nodes_n[i], nodes_n[i]])
		p_labels_clo = tf.slice( labels_clo, [n_acc, n_acc], [nodes_n[i], nodes_n[i]])
		p_labels_eig = tf.slice( labels_eig, [n_acc, n_acc], [nodes_n[i], nodes_n[i]])
		
		
#		s_labels = tf.reduce_sum( p_labels, axis=1 )
#		_,labels_top = tf.nn.top_k( s_labels, k=10 , sorted=True )
#		
#		s_predicted = tf.reduce_sum( p_predicted, axis=1 )
#		_, predicted_top = tf.nn.top_k( s_predicted, k=10 , sorted=True )
		
		#Compare labels to predicted values
		#p_error = p_labels[n_acc:n_acc+nodes_n[i],n_acc:n_acc+nodes_n[i]]#
		p_acc_deg = tf.reduce_mean(
			tf.cast(
				tf.equal(
					pred_deg_matrix_prob, p_labels_deg
				),
				tf.float32
			)
		)
		p_acc_bet = tf.reduce_mean(
			tf.cast(
				tf.equal(
					pred_bet_matrix_prob, p_labels_bet
				),
				tf.float32
			)
		)
		p_acc_clo = tf.reduce_mean(
			tf.cast(
				tf.equal(
					pred_clo_matrix_prob, p_labels_clo
				),
				tf.float32
			)
		)
		p_acc_eig = tf.reduce_mean(
			tf.cast(
				tf.equal(
					pred_eig_matrix_prob, p_labels_eig
				),
				tf.float32
			)
		)
		
		
		#Calculate cost for this problem
		p_cost_deg = tf.losses.sigmoid_cross_entropy( multi_class_labels = p_labels_deg, logits = pred_deg_matrix)
		p_cost_bet = tf.losses.sigmoid_cross_entropy( multi_class_labels = p_labels_bet, logits = pred_bet_matrix)
		p_cost_clo = tf.losses.sigmoid_cross_entropy( multi_class_labels = p_labels_clo, logits = pred_clo_matrix)
		p_cost_eig = tf.losses.sigmoid_cross_entropy( multi_class_labels = p_labels_eig, logits = pred_eig_matrix)
			
		# Update TensorArray
		arr_acc_deg = arr_acc_deg.write( i, p_acc_deg )
		arr_cost_deg = arr_cost_deg.write(i, p_cost_deg )
		arr_acc_bet = arr_acc_bet.write( i, p_acc_bet )
		arr_cost_bet = arr_cost_bet.write(i, p_cost_bet )
		arr_acc_clo = arr_acc_clo.write( i, p_acc_clo )
		arr_cost_clo = arr_cost_clo.write(i, p_cost_clo )
		arr_acc_eig = arr_acc_eig.write( i, p_acc_eig )
		arr_cost_eig = arr_cost_eig.write(i, p_cost_eig)
		
		return tf.add( i, tf.constant( 1 ) ), arr_acc_deg, arr_cost_deg, arr_acc_bet, arr_cost_bet, arr_acc_clo, arr_cost_clo, arr_acc_eig, arr_cost_eig, tf.add( n_acc, nodes_n[i] )
	#end _vote_while_body
	
			
	arr_acc_deg = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_cost_deg = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_acc_bet = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_cost_bet = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_acc_clo = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_cost_clo = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_acc_eig = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_cost_eig = tf.TensorArray( size = p, dtype = tf.float32 )
	
	
	_, arr_acc_deg, arr_cost_deg, arr_acc_bet, arr_cost_bet, arr_acc_clo, arr_cost_clo, arr_acc_eig, arr_cost_eig ,_ = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), arr_acc_deg, arr_cost_deg, arr_acc_bet, arr_cost_bet, arr_acc_clo, arr_cost_clo, arr_acc_eig, arr_cost_eig,  tf.constant( 0, dtype = tf.int32 ) ]
	)
	
	arr_acc_deg = arr_acc_deg.stack()
	arr_cost_deg = arr_cost_deg.stack()
	arr_acc_bet = arr_acc_bet.stack()
	arr_cost_bet = arr_cost_bet.stack()
	arr_acc_clo = arr_acc_clo.stack()
	arr_cost_clo = arr_cost_clo.stack()
	arr_acc_eig = arr_acc_eig.stack()
	arr_cost_eig = arr_cost_eig.stack()
	
	

	# Define batch loss
	degree_predict_cost = tf.reduce_mean( arr_cost_deg ) 
	betweenness_predict_cost = tf.reduce_mean( arr_cost_bet ) 
	closeness_predict_cost = tf.reduce_mean( arr_cost_clo ) 
	eigenvector_predict_cost = tf.reduce_mean( arr_cost_eig ) 
	
	
	vars_cost = tf.zeros([])
	tvars = tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	loss = tf.add_n( [ degree_predict_cost, betweenness_predict_cost, closeness_predict_cost, eigenvector_predict_cost,  tf.multiply( vars_cost, parameter_l2norm_scaling ) ] )
	optimizer = tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ = tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
	train_step = optimizer.apply_gradients( zip( grads, tvars ) ) #optimizer.minimize(loss) #
	
	
	#Calculate the batch average accuracy
	degree_predict_acc = tf.reduce_mean( arr_acc_deg ) 
	closeness_predict_acc = tf.reduce_mean( arr_acc_clo )
	betweenness_predict_acc = tf.reduce_mean( arr_acc_bet )
	eigenvector_predict_acc = tf.reduce_mean( arr_acc_eig )
	
	acc = tf.reduce_mean( [degree_predict_acc, closeness_predict_acc, betweenness_predict_acc, eigenvector_predict_acc] )
	
	GNN["gnn"] = gnn
	GNN["labels_deg"] = labels_deg
	GNN["labels_bet"] = labels_bet
	GNN["labels_clo"] = labels_clo
	GNN["labels_eig"] = labels_eig
	GNN["degree_predict_cost"] = degree_predict_cost
	GNN["degree_predict_acc"] = degree_predict_acc
	GNN["betweenness_predict_cost"] = betweenness_predict_cost
	GNN["betweenness_predict_acc"] = betweenness_predict_acc
	GNN["closeness_predict_cost"] = closeness_predict_cost
	GNN["closeness_predict_acc"] = closeness_predict_acc
	GNN["eigenvector_predict_cost"] = eigenvector_predict_cost
	GNN["eigenvector_predict_acc"] = eigenvector_predict_acc
	GNN["acc"] = acc
	GNN["loss"] = loss
	GNN["nodes_n"] = nodes_n
	GNN["train_step"] = train_step
	return GNN
#end build_network


def precisionAt10(labels, predicted):
	precs = np.zeros(predicted.shape[0])
	for i in range(0, predicted.shape[0]):
		intersect = np.intersect1d( predicted[i], labels[i], assume_unique=True)
		precs[i] = intersect.size / predicted[i].size
	return np.mean( precs )


if __name__ == '__main__':
	embedding_size = 64
	epochs = 100
	batch_n_max = 4096
	batches_per_epoch = 32
	n_size_min = 20
	n_size_max = 512
	edge_probability = 0.25
	time_steps = 32
	n_instances_min = 8
	n_instances_max = 64
	n_instances = 32

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(embedding_size)
	instance_loader = InstanceLoader("./instances") 
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
			instance_loader.reset()
			epoch_loss = 0.0
			epoch_degc = 0
			epoch_degacc = 0
			epoch_betc = 0
			epoch_betacc = 0
			epoch_cloc = 0
			epoch_cloacc = 0
			epoch_eigc = 0
			epoch_eigacc = 0
			epoch_acc = 0.0
			epoch_n = 0
			epoch_m = 0
			for cbat, batch in enumerate(instance_loader.get_batches(32)):
				
				M = batch["matrix"]
				n = M[2][0]
				m = len(M[0])
				
				_,loss, acc, degc, degacc, betc, betacc, cloc, cloacc, eigc, eigacc = sess.run(
					[ GNN["train_step"], GNN["loss"], GNN["acc"], GNN["degree_predict_cost"], GNN["degree_predict_acc"], GNN["betweenness_predict_cost"], GNN["betweenness_predict_acc"], GNN["closeness_predict_cost"], GNN["closeness_predict_acc"], GNN["eigenvector_predict_cost"], GNN["eigenvector_predict_acc"] ],
					feed_dict = {
						GNN["nodes_n"]: batch["problem_n"],
						GNN["labels_deg"]: sparse_to_dense(batch["degree_compare"]),
						GNN["labels_bet"]: sparse_to_dense(batch["betweenness_compare"]),
						GNN["labels_clo"]: sparse_to_dense(batch["closeness_compare"]),
						GNN["labels_eig"]: sparse_to_dense(batch["eigenvector_compare"]),
						GNN["gnn"].time_steps: time_steps,
						GNN["gnn"].matrix_placeholders["M"]: sparse_to_dense(M)
					}
				)
				
				
				epoch_loss += loss
				epoch_degc += degc
				epoch_degacc += degacc
				epoch_betc += betc
				epoch_betacc += betacc
				epoch_cloc += cloc
				epoch_cloacc += cloacc
				epoch_eigc += eigc
				epoch_eigacc += eigacc
				epoch_acc += acc
				epoch_n += n
				epoch_m += m
				
				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,i): ({n},{m},{i})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f}, B:{bet_cost:.5f}, C:{clo_cost:.5f}, E:{eig_cost:.5f}) Acc(T:{acc:.5f}, D:{degree_acc:.5f}, B:{bet_acc:.5f}, C:{clo_acc:.5f}, E:{eig_acc:.5f}) ".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = cbat,
						loss = loss,
						acc = acc,
						degree_cost = degc,
						degree_acc = degacc,
						bet_cost = betc,
						bet_acc = betacc,
						clo_cost = cloc,
						clo_acc = cloacc,
						eig_cost = eigc,
						eig_acc = eigacc,
						n = n,
						m = m,
						i = len(batch["problem_n"])
					),
					flush = True
				)
				if(cbat == 31):
					break
			#end for
			# Summarize Epoch
			epoch_loss /= batches_per_epoch
			epoch_acc /= batches_per_epoch
			epoch_degc /= batches_per_epoch
			epoch_degacc /= batches_per_epoch
			epoch_betc /= batches_per_epoch
			epoch_betacc /= batches_per_epoch
			epoch_cloc /= batches_per_epoch
			epoch_cloacc /= batches_per_epoch
			epoch_eigc /= batches_per_epoch
			epoch_eigacc /= batches_per_epoch
			print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f}, B:{bet_cost:.5f}, C:{clo_cost:.5f}, E:{eig_cost:.5f}) Acc(T:{acc:.5f}, D:{degree_acc:.5f}, B:{bet_acc:.5f}, C:{clo_acc:.5f}, E:{eig_acc:.5f}) ".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = "all",
						loss = epoch_loss,
						acc = epoch_acc,
						degree_cost = epoch_degc,
						degree_acc = epoch_degacc,
						bet_cost = epoch_betc,
						bet_acc = epoch_betacc,
						clo_cost = epoch_cloc,
						clo_acc = epoch_cloacc,
						eig_cost = epoch_eigc,
						eig_acc = epoch_eigacc,
						n = epoch_n,
						m = epoch_m
					),
					flush = True
				)
			# TEST TIME
			# Create random graphs
			test_batch =  instance_loader.get_batch(32) 
			M = test_batch["matrix"]
			test_n = M[2][0]
			test_m = len( M[0] )
		
			test_loss, test_acc, test_degc, test_degacc, test_betc, test_betacc, test_cloc, test_cloacc, test_eigc, test_eigacc = sess.run(
					[ GNN["loss"], GNN["acc"], GNN["degree_predict_cost"], GNN["degree_predict_acc"], GNN["betweenness_predict_cost"], GNN["betweenness_predict_acc"], GNN["closeness_predict_cost"], GNN["closeness_predict_acc"], GNN["eigenvector_predict_cost"], GNN["eigenvector_predict_acc"] ],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["M"]: sparse_to_dense( M ) ,
					GNN["gnn"].time_steps: time_steps,
					GNN["labels_deg"]: sparse_to_dense(test_batch["degree_compare"]),
					GNN["labels_bet"]: sparse_to_dense(test_batch["betweenness_compare"]),
					GNN["labels_clo"]: sparse_to_dense(test_batch["closeness_compare"]),
					GNN["labels_eig"]: sparse_to_dense(test_batch["eigenvector_compare"]),
					GNN["nodes_n"]: test_batch["problem_n"]
				}
			)
		
			print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,i): ({n},{m},{i})\t| Loss(T:{loss:.5f},D:{degree_cost:.5f}, B:{bet_cost:.5f}, C:{clo_cost:.5f}, E:{eig_cost:.5f}) Acc(T:{acc:.5f}, D:{degree_acc:.5f}, B:{bet_acc:.5f}, C:{clo_acc:.5f}, E:{eig_acc:.5f}) ".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						batch = "rnd",
						loss = test_loss,
						acc = test_acc,
						degree_cost = test_degc,
						degree_acc = test_degacc,
						bet_cost = test_betc,
						bet_acc = test_betacc,
						clo_cost = test_cloc,
						clo_acc = test_cloacc,
						eig_cost = test_eigc,
						eig_acc = test_eigacc,
						n = n,
						m = m,
						i = len(batch["problem_n"])
					),
					flush = True
				)
			
			
			save_weights(sess,"rank-centrality-checkpoints")
		#end for(epochs)
	#end Session
