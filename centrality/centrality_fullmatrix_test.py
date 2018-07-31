import sys, os, time
import tensorflow as tf
import numpy as np
import networkx as nx
import itertools
import tkinter
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from graphnn_refactored import GraphNN
from mlp import Mlp
# Import tools
import itertools
from util import timestamp, memory_usage, sparse_to_dense, save_weights#, percent_error
from instance_loader import InstanceLoader
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from matplotlib import pyplot as plt

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
	
	k = tf.placeholder( tf.int32, shape=() , name = "k" )
	
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
	def _vote_while_cond(i, arr_acc_deg, arr_cost_deg, topk_deg, arr_acc_bet, arr_cost_bet, topk_bet, arr_acc_clo, arr_cost_clo, topk_clo, arr_acc_eig, arr_cost_eig, topk_eig, n_acc):
		return tf.less( i, p )
	#end _vote_while_cond

	def _vote_while_body(i, arr_acc_deg, arr_cost_deg, topk_deg, arr_acc_bet, arr_cost_bet, topk_bet, arr_acc_clo, arr_cost_clo, topk_clo, arr_acc_eig, arr_cost_eig, topk_eig, n_acc):
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
		
		#Get top k rankings for each centrality
		s_predicted = tf.reduce_sum( pred_deg_matrix_prob, axis=1 )
		_, predicted_topk_deg = tf.nn.top_k( s_predicted, k=k , sorted=True )
		
		s_predicted = tf.reduce_sum( pred_bet_matrix_prob, axis=1 )
		_, predicted_topk_bet = tf.nn.top_k( s_predicted, k=k , sorted=True )
		
		s_predicted = tf.reduce_sum( pred_clo_matrix_prob, axis=1 )
		_, predicted_topk_clo = tf.nn.top_k( s_predicted, k=k , sorted=True )
		
		s_predicted = tf.reduce_sum( pred_eig_matrix_prob, axis=1 )
		_, predicted_topk_eig = tf.nn.top_k( s_predicted, k=k , sorted=True )
		
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
		topk_deg = topk_deg.write( i, predicted_topk_deg )
		arr_acc_bet = arr_acc_bet.write( i, p_acc_bet )
		arr_cost_bet = arr_cost_bet.write(i, p_cost_bet )
		topk_bet = topk_bet.write( i, predicted_topk_bet )
		arr_acc_clo = arr_acc_clo.write( i, p_acc_clo )
		arr_cost_clo = arr_cost_clo.write(i, p_cost_clo )
		topk_clo = topk_clo.write( i, predicted_topk_clo )
		arr_acc_eig = arr_acc_eig.write( i, p_acc_eig )
		arr_cost_eig = arr_cost_eig.write(i, p_cost_eig)
		topk_eig = topk_eig.write( i, predicted_topk_eig )
		
		return tf.add( i, tf.constant( 1 ) ), arr_acc_deg, arr_cost_deg, topk_deg, arr_acc_bet, arr_cost_bet, topk_bet, arr_acc_clo, arr_cost_clo, topk_clo, arr_acc_eig, arr_cost_eig, topk_eig, tf.add( n_acc, nodes_n[i] )
	#end _vote_while_body
	
	arr_acc_deg = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_cost_deg = tf.TensorArray( size = p, dtype = tf.float32 )
	topk_deg = tf.TensorArray( size = p, dtype = tf.int32 )
	arr_acc_bet = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_cost_bet = tf.TensorArray( size = p, dtype = tf.float32 )
	topk_bet = tf.TensorArray( size = p, dtype = tf.int32 )
	arr_acc_clo = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_cost_clo = tf.TensorArray( size = p, dtype = tf.float32 )
	topk_clo = tf.TensorArray( size = p, dtype = tf.int32 )
	arr_acc_eig = tf.TensorArray( size = p, dtype = tf.float32 )
	arr_cost_eig = tf.TensorArray( size = p, dtype = tf.float32 )
	topk_eig = tf.TensorArray( size = p, dtype = tf.int32 )
	
	_, arr_acc_deg, arr_cost_deg, topk_deg, arr_acc_bet, arr_cost_bet, topk_bet, arr_acc_clo, arr_cost_clo, topk_clo, arr_acc_eig, arr_cost_eig , topk_eig, _ = tf.while_loop(
		_vote_while_cond,
		_vote_while_body,
		[ tf.constant( 0, dtype = tf.int32 ), arr_acc_deg, arr_cost_deg, topk_deg, arr_acc_bet, arr_cost_bet, topk_bet, arr_acc_clo, arr_cost_clo, topk_clo, arr_acc_eig, arr_cost_eig, topk_eig, tf.constant( 0, dtype = tf.int32 ) ]
	)
	
	arr_acc_deg = arr_acc_deg.stack()
	arr_cost_deg = arr_cost_deg.stack()
	topk_deg = topk_deg.stack()
	arr_acc_bet = arr_acc_bet.stack()
	arr_cost_bet = arr_cost_bet.stack()
	topk_bet = topk_bet.stack()
	arr_acc_clo = arr_acc_clo.stack()
	arr_cost_clo = arr_cost_clo.stack()
	topk_clo = topk_clo.stack()
	arr_acc_eig = arr_acc_eig.stack()
	arr_cost_eig = arr_cost_eig.stack()
	topk_eig = topk_eig.stack()
	
	

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
	GNN["last_embeddings"] = N_n
	GNN["labels_deg"] = labels_deg
	GNN["labels_bet"] = labels_bet
	GNN["labels_clo"] = labels_clo
	GNN["labels_eig"] = labels_eig
	GNN["k"] = k
	GNN["degree_predict_cost"] = degree_predict_cost
	GNN["degree_predict_acc"] = degree_predict_acc
	GNN["topk_deg"] = topk_deg
	GNN["betweenness_predict_cost"] = betweenness_predict_cost
	GNN["betweenness_predict_acc"] = betweenness_predict_acc
	GNN["topk_bet"] = topk_bet
	GNN["closeness_predict_cost"] = closeness_predict_cost
	GNN["closeness_predict_acc"] = closeness_predict_acc
	GNN["topk_clo"] = topk_clo
	GNN["eigenvector_predict_cost"] = eigenvector_predict_cost
	GNN["eigenvector_predict_acc"] = eigenvector_predict_acc
	GNN["topk_eig"] = topk_eig
	GNN["acc"] = acc
	GNN["loss"] = loss
	GNN["nodes_n"] = nodes_n
	GNN["train_step"] = train_step
	return GNN
#end build_network
	
	
def precisionAtK(labels, predicted, k = 10):
	#precs = np.zeros(predicted.shape[0])
	intersect = np.intersect1d( predicted, labels, assume_unique=True)
	precs = intersect.size / k # k = predicted[i].size
	return precs


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
	n_instances = 1
	k = 30
	
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(embedding_size)
	test_generator = InstanceLoader("./instances-test") 
	# Disallow GPU use
	config = tf.ConfigProto(
		device_count = {"GPU":0},
		inter_op_parallelism_threads=1,
		intra_op_parallelism_threads=1
	)
	
	# Create model saver
	saver = tf.train.Saver()
	with tf.Session() as sess:
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory= 			memory_usage() ) )
		sess.run( tf.global_variables_initializer() )

		# Restore saved weights
		print( "{timestamp}\t{memory}\tRestoring saved model ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		saver.restore(sess, "./rank-centrality-checkpoints/model.ckpt")
		test_batches = 0
		test_loss = 0.0
		test_loss_deg = 0.0
		test_loss_bet = 0.0 
		test_loss_clo = 0.0
		test_loss_eig = 0.0
		test_acc = 0.0
		test_acc_deg = 0.0
		test_acc_bet = 0.0
		test_acc_clo = 0.0
		test_acc_eig = 0.0

		test_precatk_deg = 0.0
		test_precatk_bet = 0.0
		test_precatk_clo = 0.0
		test_precatk_eig = 0.0
		print("Test number\tLoss\tDegree Loss\tBetweenness Loss\tCloseness Loss\tEigenvector Loss\tAccuracy\tDegree Acc\tBetweenness Acc\tCloseness Acc\tEigenvector Acc\tDegree PrecAtk\tBetweenness PreacAtk\tCloseness PrecAtk\tEigenvector PrecAtk")
		for b, batch in enumerate( test_generator.get_batches( n_instances ) ):
			M = batch["matrix"]
			n = M[2][0]
			m = len(M[0])
			p_embeddings, loss, acc, degc, degacc, betc, betacc, cloc, cloacc, eigc, eigacc, pred_topk_deg, pred_topk_bet, pred_topk_clo, pred_topk_eig = sess.run(
					[ GNN["last_embeddings"], GNN["loss"], GNN["acc"], GNN["degree_predict_cost"], GNN["degree_predict_acc"], GNN["betweenness_predict_cost"], GNN["betweenness_predict_acc"], GNN["closeness_predict_cost"], GNN["closeness_predict_acc"], GNN["eigenvector_predict_cost"], GNN["eigenvector_predict_acc"], GNN["topk_deg"], GNN["topk_bet"], GNN["topk_clo"], GNN["topk_eig"] ],
					feed_dict = {
						GNN["nodes_n"]: batch["problem_n"],
						GNN["labels_deg"]: sparse_to_dense(batch["degree_compare"]),
						GNN["labels_bet"]: sparse_to_dense(batch["betweenness_compare"]),
						GNN["labels_clo"]: sparse_to_dense(batch["closeness_compare"]),
						GNN["labels_eig"]: sparse_to_dense(batch["eigenvector_compare"]),
						GNN["gnn"].time_steps: time_steps,
						GNN["k"]: k,
						GNN["gnn"].matrix_placeholders["M"]: sparse_to_dense(M)
					}
			)
			label_rank_deg = np.sum(sparse_to_dense(batch["degree_compare"]), axis=1)
			label_rank_bet = np.sum(sparse_to_dense(batch["betweenness_compare"]), axis=1)
			label_rank_clo = np.sum(sparse_to_dense(batch["closeness_compare"]), axis=1)
			label_rank_eig = np.sum(sparse_to_dense(batch["eigenvector_compare"]), axis=1)
			
			
			label_topk_deg = np.argpartition(label_rank_deg, -k)[-k:]
			label_topk_deg = label_topk_deg[np.argsort(label_rank_deg[label_topk_deg])][::-1]
			
			label_topk_bet = np.argpartition(label_rank_bet, -k)[-k:]
			label_topk_bet = label_topk_bet[np.argsort(label_rank_bet[label_topk_bet])][::-1]
			
			label_topk_clo = np.argpartition(label_rank_clo, -k)[-k:]
			label_topk_clo = label_topk_clo[np.argsort(label_rank_clo[label_topk_clo])][::-1]
			label_topk_eig = np.argpartition(label_rank_eig, -k)[-k:]
			label_topk_eig = label_topk_eig[np.argsort(label_rank_eig[label_topk_eig])][::-1]
			
			print(batch["betweenness"][0][:10])
			print(batch["closeness"][0][:10])
			
			precatk_deg = precisionAtK(label_topk_deg, pred_topk_deg.squeeze(), k)
			precatk_bet = precisionAtK(label_topk_bet, pred_topk_bet.squeeze(), k)
			precatk_clo = precisionAtK(label_topk_clo, pred_topk_clo.squeeze(), k)
			precatk_eig = precisionAtK(label_topk_eig, pred_topk_eig.squeeze(), k)
			#print(pred_top_deg.shape)
			#print(pred_top_bet.shape)
			print( "{test_batches}\t{loss:.4f}\t{loss_deg:.4f}\t{loss_bet:.4f}\t{loss_clo:.4f}\t{loss_eig:.4f}\t{accuracy:.4f}\t{acc_deg:.4f}\t{acc_bet:.4f}\t{acc_clo:.4f}\t{acc_eig:.4f}\t{precatk_deg:.4f}\t{precatk_bet:.4f}\t{precatk_clo:.4f}\t{precatk_eig:.4f}".format(
			loss = loss,
			loss_deg = degc,
			loss_bet = betc,
			loss_clo = cloc,
			loss_eig = eigc,
			accuracy = acc,
			acc_deg = degacc,
			acc_bet = betacc,
			acc_clo = cloacc,
			acc_eig = eigacc,
			precatk_deg = precatk_deg,
			precatk_bet = precatk_bet,
			precatk_clo = precatk_clo,
			precatk_eig = precatk_eig,
			test_batches = test_batches
			) )
	
			test_loss += loss
			test_loss_deg += degc
			test_loss_bet += betc
			test_loss_clo += cloc
			test_loss_eig += eigc
			test_acc += acc
			test_acc_deg += degacc
			test_acc_bet += betacc
			test_acc_clo += cloacc
			test_acc_eig += eigacc
			test_precatk_deg += precatk_deg
			test_precatk_bet += precatk_bet
			test_precatk_clo += precatk_clo
			test_precatk_eig += precatk_eig
			
			
			
			if test_batches == 0:
				labels_topk = []
				labels_topk.append(label_topk_deg)
				labels_topk.append(label_topk_bet)
				labels_topk.append(label_topk_clo)
				labels_topk.append(label_topk_eig)
				labels_centralities=[]
				labels_centralities.append(batch["degree"][0])
				labels_centralities.append(batch["betweenness"][0])
				labels_centralities.append(batch["closeness"][0])
				labels_centralities.append(batch["eigenvector"][0])
				principal_components = PCA(n_components=1).fit_transform(p_embeddings)
				principal_components /= np.linalg.norm(principal_components)
				red_indexes = []
				blue_indexes = []
				red_components =[]
				blue_components = []
				red_label_centrality=[]
				blue_label_centrality=[]
				#vermelhos tao no topk
				for c in range(4):
					red_indexes.append([ i for i,state in enumerate(p_embeddings) if i in labels_topk[c] ])
					blue_indexes.append([ i for i,state in enumerate(p_embeddings) if i not in labels_topk[c] ])

					red_components.append([x for i,x in enumerate(principal_components) if i in red_indexes[c]])
					blue_components.append([x for i,x in enumerate(principal_components) if i in blue_indexes[c]])
				
					red_label_centrality.append([x for i,x in enumerate(labels_centralities[c]) if i in red_indexes[c]])
					blue_label_centrality.append([x for i,x in enumerate(labels_centralities[c]) if i in blue_indexes[c]])

					plt.scatter([x+(c%2) for i,x in enumerate(red_components[c])],[y+(c/2) for i,y in enumerate(red_label_centrality[c])], color=['red'])
					plt.scatter([x+(c%2) for i,x in enumerate(blue_components[c])],[y+(c/2) for i,y in enumerate(blue_label_centrality[c])], color=['blue'])
				plt.show()
			#end if
			
			
			test_batches += 1
		#end for
		
		test_loss /= test_batches
		test_loss_deg /= test_batches
		test_loss_bet /= test_batches
		test_loss_clo /= test_batches
		test_loss_eig /= test_batches
		test_acc /= test_batches
		test_acc_deg /= test_batches
		test_acc_bet /= test_batches
		test_acc_clo /= test_batches
		test_acc_eig /= test_batches
		test_precatk_deg /= test_batches
		test_precatk_bet /= test_batches
		test_precatk_clo /= test_batches
		test_precatk_eig /= test_batches
		print( "{test_batches}\t{loss:.4f}\t{loss_deg:.4f}\t{loss_bet:.4f}\t{loss_clo:.4f}\t{loss_eig:.4f}\t{accuracy:.4f}\t{acc_deg:.4f}\t{acc_bet:.4f}\t{acc_clo:.4f}\t{acc_eig:.4f}\t{precatk_deg:.4f}\t{precatk_bet:.4f}\t{precatk_clo:.4f}\t{precatk_eig:.4f}".format(
		loss = test_loss,
		loss_deg = test_loss_deg,
		loss_bet = test_loss_bet,
		loss_clo = test_loss_clo,
		loss_eig = test_loss_eig,
		accuracy = test_acc,
		acc_deg = test_acc_deg,
		acc_bet = test_acc_bet,
		acc_clo = test_acc_clo,
		acc_eig = test_acc_eig,
		precatk_deg = test_precatk_deg,
		precatk_bet = test_precatk_bet,
		precatk_clo = test_precatk_clo,
		precatk_eig = test_precatk_eig,
		test_batches = "Total average"
		) )
			
    	
    
