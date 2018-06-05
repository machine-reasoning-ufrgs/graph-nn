
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
from util import timestamp, memory_usage, save_weights, load_weights
from keras.utils import to_categorical

def build_network(d,n):
	"""
		d: Embedding size
		n: Sudoku board size (n² × n²)
	"""

	# Hyperparameters
	learning_rate = 2e-5
	parameter_l2norm_scaling = 1e-10
	global_norm_gradient_clipping_ratio = 0.65

	# Define GNN dictionary
	GNN = {}

	# Define givens matrix placeholder
	is_given = tf.placeholder( tf.float32, [ None ], name = "is_given" )

	# Count the number of problems
	p = tf.div(tf.shape(is_given)[0], tf.constant(n**4))

	# Define Graph neural network
	gnn = GraphNN(
		{
			"C": d
		},
		{
			"M_C": ("C","C"),
			"M_G": ("C",n**2)
		},
		{
		},
		{
			"C": [
				{
					"mat": "M_C",
					"var": "C"
				},
				{
					"mat": "M_G"
				}
			]
		},
		Cell_activation = tf.nn.sigmoid,
		Msg_last_activation = tf.nn.sigmoid,
		name="Sudoku",
	)

	# Define voting MLP
	Cvote = Mlp(
		layer_sizes = [ d for _ in range(2) ],
		activations = [ tf.nn.relu for _ in range(2) ],
		output_size = n**2,
		output_activation = tf.nn.sigmoid,
		name = "Cvote",
		name_internal_layers = True,
		kernel_initializer = tf.contrib.layers.xavier_initializer(),
		bias_initializer = tf.zeros_initializer()
	)

	# Get the last embeddings
	C_n = gnn.last_states["C"].h
	# Get predictions
	unscaled_preds 	= Cvote(C_n)
	scaled_preds 	= tf.nn.softmax(unscaled_preds)

	# Define losses, accuracies, optimizer, train steps

	"""
		The digit predicted for 'given' cells must be equal to the
		corresponding digit in the puzzle. This loss penalizes cells deviating
		from this pattern.
	"""
	loss_givens = tf.losses.softmax_cross_entropy(
		logits 			= tf.multiply(unscaled_preds, tf.matmul(tf.expand_dims(is_given,1), tf.ones((1,n**2)))),
		onehot_labels 	= gnn.matrix_placeholders["M_G"],
		)

	# The conflicts loss is the expected number of conflicts in the grid
	# Reduce sum along all cell pairs
	loss_conflicts = tf.reduce_mean(
		tf.reduce_sum(
			# Perform element-wise multiplication with adjacency matrix between
			# cells to only consider the conttribution of adjacent cells
			tf.multiply(
				gnn.matrix_placeholders["M_C"],
				# Reduce sum along digit axis, computing for each pair of cells
				# the probability that they are in conflict with each other
				tf.reduce_sum(
					# For each pair of cells i,j (among every board in the batch)
					# and for each digit k compute
					# P(cell(i)=k ∧ cell(j)=k) = 
					# P(cell(i)=k) × P(cell(j)=k)
					tf.multiply(
						tf.reshape(tf.tile(tf.reshape(scaled_preds,(p*n**4,1,n**2)), (p*n**4,1,1)), (p*n**4, p*n**4, n**2)),
						tf.reshape(tf.tile(tf.reshape(scaled_preds,(p*n**4,1,n**2)), (1,p*n**4,1)), (p*n**4, p*n**4, n**2))
						),
					axis=-1
					)
				),
				axis=1
			)
		)

	# This accuracy measures how many 'given' cells were accurately marked
	# Reduce mean along all cells from all boards in the batch
	acc_givens = tf.subtract(
		tf.constant(1.0),
		tf.divide(
			tf.reduce_sum(
				# Perform element-wise multiplication with givens mask to only
				# consider the conttribution of incorrectly marked cells in
				# 'given' positions
				tf.multiply(
					is_given,
					tf.cast(
						# Compute a boolean 'diff' array indicating which cells are incorrectly marked
						tf.not_equal(
							tf.argmax(scaled_preds, axis=1),
							tf.argmax(gnn.matrix_placeholders["M_G"], axis=1)
							),
						tf.float32
						)
					)
				),
				tf.reduce_sum(is_given)
			)
		)

	# This metric is the average number of conflicts per cell
	# Reduce mean along all cells from all boards in the batch
	avg_conflicts = tf.reduce_mean(
		tf.reduce_sum(
			# Perform element-wise multiplication with adjacency matrix to only
			# add the conttribution of adjacent cells with identical digits to the
			# conflicts count
			tf.multiply(
				gnn.matrix_placeholders["M_C"],
				tf.cast(
					# Compute, for every cell pair, whether they are marked with the same digit
					tf.equal(
						tf.argmax(tf.reshape(tf.tile(tf.reshape(scaled_preds,(p*n**4,1,n**2)), (p*n**4,1,1)), (p*n**4, p*n**4, n**2)), axis=-1),
						tf.argmax(tf.reshape(tf.tile(tf.reshape(scaled_preds,(p*n**4,1,n**2)), (1,p*n**4,1)), (p*n**4, p*n**4, n**2)), axis=-1)
						),
					tf.float32
					)
				),
			axis=1
			)
		)

	vars_cost = tf.zeros([])
	tvars = tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	
	optimizer 	= tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	
	grads_givens, _ 		= tf.clip_by_global_norm( tf.gradients( loss_givens, tvars ), global_norm_gradient_clipping_ratio )
	train_step_givens		= optimizer.apply_gradients( zip( grads_givens, tvars ) )

	grads_conflicts, _ 		= tf.clip_by_global_norm( tf.gradients( loss_conflicts, tvars ), global_norm_gradient_clipping_ratio )
	train_step_conflicts 	= optimizer.apply_gradients( zip( grads_conflicts, tvars ) )

	GNN["gnn"] 					= gnn
	GNN["prediction"] 			= scaled_preds
	GNN["error"] 				= 0
	GNN["loss_givens"] 			= loss_givens
	GNN["loss_conflicts"] 		= loss_conflicts
	GNN["acc_givens"] 			= acc_givens
	GNN["avg_conflicts"] 		= avg_conflicts
	GNN["train_step_givens"] 	= train_step_givens
	GNN["train_step_conflicts"] = train_step_conflicts
	GNN["is_given"] 			= is_given
	
	return GNN
#end

def compute_sudoku_adj_matrix(n):
	"""
		Compute a n⁴×n⁴ sudoku adjacency matrix between each pair of cells
	"""
	A = np.zeros((n**4,n**4))
	for i1 in range(n**2):
		for i2 in range(n**2):
			for j1 in range(n**2):
				for j2 in range(n**2):
					if (i1,j1) != (i2,j2) and (i1==i2 or j1==j2 or (i1//n==i2//n and j1//n==j2//n)):
						A[n**2*i1 + j1, n**2*i2 + j2] = 1
					#end
				#end
			#end
		#end
	#end
	return A
#end

def parse_givens(text):
	G = to_categorical([ int(c) for c in text ], num_classes=1+n**2)[:,1:]
	return G
#end

def create_batch(puzzles, batch_size):
	sudoku_adj_matrix = compute_sudoku_adj_matrix(n)
	M_C = np.zeros((batch_size*n**4, batch_size*n**4))
	M_G = np.zeros((batch_size*n**4, n**2))
	for i, puzzle in enumerate(puzzles):
		M_C[i*n**4 : (i+1)*n**4, i*n**4 : (i+1)*n**4]	= sudoku_adj_matrix
		M_G[i*n**4 : (i+1)*n**4, :] 					= parse_givens(puzzle)
	#end
	is_given = (np.argmax(M_G, axis=1) != 0).astype(int)
	return M_C, M_G, is_given
#end

if __name__ == '__main__':
	d = 64
	n = 3
	epochs = 10000
	batch_size = 32
	time_steps = 16

	with open("sudoku.csv") as f:
		puzzles = [ line.strip().split(",")[0] for line in f.readlines()[1:][:2**12] ]
	#end

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(d,n)

	# Disallow GPU use
	config = tf.ConfigProto( device_count = {"GPU":0})
	with tf.Session(config=config) as sess:

		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )

		# Load saved weights
		#load_weights(sess,"./sudoku-checkpoints")

		# Run for a number of epochs
		print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
		for epoch in range( epochs ):

			for batch_i, batch in [ (i//32, create_batch(puzzles[i : i + batch_size], batch_size)) for i in range(0, len(puzzles), batch_size) ]:

				M_C, M_G, is_given = batch

				_, _, loss_givens, acc_givens, loss_conflicts, avg_conflicts, prediction = sess.run(
						[ GNN["train_step_givens"], GNN["train_step_conflicts"], GNN["loss_givens"], GNN["acc_givens"], GNN["loss_conflicts"], GNN["avg_conflicts"], GNN["prediction"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["M_C"]: M_C,
							GNN["gnn"].matrix_placeholders["M_G"]: M_G,
							GNN["gnn"].time_steps: time_steps,
							GNN["is_given"]: is_given
						}
					)

				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} n: {n}\t| Givens (Loss,Acc): ({gloss:.5f}, {gacc:.3f})\t| Conflicts (Loss,Acc): ({closs:.5f}, {cavg:.3f})".format(
						timestamp 	= timestamp(),
						memory 		= memory_usage(),
						epoch 		= epoch,
						gloss 		= loss_givens,
						gacc 		= acc_givens,
						closs 		= loss_conflicts,
						cavg 		= avg_conflicts,
						batch 		= batch_i,
						n 			= n
					),
					flush = True
				)

			#end

			M_C, M_G, is_given = create_batch(puzzles[0:1],1)

			prediction = sess.run(
				GNN["prediction"],
				feed_dict = {
				GNN["gnn"].matrix_placeholders["M_C"]: M_C,
				GNN["gnn"].matrix_placeholders["M_G"]: M_G,
				GNN["gnn"].time_steps: time_steps,
				GNN["is_given"]: is_given
				}
				)

			print("Puzzle:\t\t{}".format(puzzles[0]))
			print("Prediction:\t{}".format( ''.join([str(x+1) for x in np.argmax(prediction, axis=1)])) )
			print("Diff:\t\t{}\n".format(
				''.join(
					[str(x) for x in ((np.argmax(prediction, axis=1) + 1) != [int(x) for x in puzzles[0]]).astype(int)]
					)
				))

			save_weights(sess,"./sudoku-checkpoints")

		#end
	#end
#end