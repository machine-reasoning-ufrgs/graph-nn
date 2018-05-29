
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
	blanks = tf.placeholder( tf.float32, [ None ], name = "blanks" )

	# Count the number of problems
	p = tf.div(tf.shape(blanks)[0], tf.constant(n**4))

	# Define Graph neural network
	gnn = GraphNN(
		{
			"C": d,
			"G": n**2
		},
		{
			"M_CC": ("C","C"),
			"M_CG": ("C","G")
		},
		{
			"G_msg_C": ("G","C")
		},
		{
			"C": [
				{
					"mat": "M_CC",
					"var": "C"
				},
				{
					"mat": "M_CG",
					"var": "G",
					"msg": "G_msg_C"
				}
			]
		},
		Cell_activation = tf.nn.sigmoid,
		Msg_last_activation = tf.nn.sigmoid,
		override_cell_h0 = { "G": tf.tile(np.eye(n**2, dtype=np.float32), (p,1)) },
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
	# Get prediction
	prediction = tf.nn.softmax(Cvote(C_n))

	# Define loss, optimizer, train step

	def while_body(i, prediction):

		mask.write( () )

		tf.add( i, tf.constant( 1 ) ), mask
	#end

	mask = np.zeros((p * n**4, p * n**2))


	loss_givens = tf.losses.softmax_cross_entropy(gnn.matrix_placeholders["M_CG"],
		tf.reshape(prediction, (p, n**4, p, n**2)),

	# The loss is the expected number of conflicts in the grid
	loss_conflicts = tf.reduce_sum(
		# Multiply with blanks mask in order to only consider the conttribution of blank cells
		tf.multiply(
			# Reduce mean along the second axis, obtaining the expected numer of conflicts per cell
			tf.reduce_mean(
				# Multiply with n⁴ × n⁴ adjacency matrix to only consider the conttribution of adjacent cells
				tf.multiply(
					# Compute, for each pair of cells, the probability that the
					# model will predict the same digit for them
					tf.reduce_mean(
						# Compute, for each pair of cells c1, c2 and for each
						# digit x, the probability that the model will predict x
						# for both cells
						tf.multiply(
							tf.reshape(tf.tile(prediction, (n**4,1)), (n**4,n**4,n**2)),
							tf.reshape(tf.tile(prediction, (1,n**4)), (n**4,n**4,n**2))
							)
						,
						axis = -1
						),
					gnn.matrix_placeholders["M_CC"]
					),
					axis = 1
				),
			blanks
			)
		)

	loss = loss_givens

	vars_cost = tf.zeros([])
	tvars = tf.trainable_variables()
	for var in tvars:
		vars_cost = tf.add( vars_cost, tf.nn.l2_loss( var ) )
	#end for
	optimizer 	= tf.train.AdamOptimizer( name = "Adam", learning_rate = learning_rate )
	grads, _ 	= tf.clip_by_global_norm( tf.gradients( loss, tvars ), global_norm_gradient_clipping_ratio )
	train_step 	= optimizer.apply_gradients( zip( grads, tvars ) )

	GNN["gnn"] 			= gnn
	GNN["prediction"] 	= prediction
	GNN["error"] 		= 0
	GNN["loss"] 		= loss
	GNN["train_step"] 	= train_step
	GNN["blanks"] 		= blanks
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
					if (i1,j1) != (i2,j2) and i1==i2 or j1==j2 or (i1//n==i2//n and j1//n==j2//n):
						A[n**2*i1 + j1, n**2*i2 + j2] = 1
					#end
				#end
			#end
		#end
	#end
	return A
#end

def parse_CG(text):
	CG = to_categorical([ int(c) for c in text ], num_classes=n**2+1)[:,1:]
	return CG
#end

def create_batch(puzzles):
	sudoku_adj_matrix = compute_sudoku_adj_matrix(n)
	CC = np.zeros((batch_size*n**4, batch_size*n**4))
	CG = np.zeros((batch_size*n**4, batch_size*n**2))
	for i, puzzle in enumerate(puzzles):
		CC[i*n**4 : (i+1)*n**4, i*n**4 : (i+1)*n**4] = sudoku_adj_matrix
		CG[i*n**4 : (i+1)*n**4, i*n**2 : (i+1)*n**2] = parse_CG(puzzle)
	#end
	blanks = (np.sum(CG, axis=1) == 0).astype(int)
	return CC, CG, blanks
#end

if __name__ == '__main__':
	d = 64
	n = 3
	epochs = 10000
	batch_n_max = 4096
	batch_size = 32
	time_steps = 32

	with open("sudoku.csv") as f:
		puzzles = [ line.strip().split(",")[1] for line in f.readlines()[1:1000] ]
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

		# Run for a number of epochs
		print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
		for epoch in range( epochs ):

			for batch_i, batch in [ (i, create_batch(puzzles[i : i + batch_size])) for i in range(0, len(puzzles), batch_size) ]:

				CC, CG, blanks = batch

				_, loss, prediction = sess.run(
						[ GNN["train_step"], GNN["loss"], GNN["prediction"] ],
						feed_dict = {
							GNN["gnn"].matrix_placeholders["M_CC"]: CC,
							GNN["gnn"].matrix_placeholders["M_CG"]: CG,
							GNN["gnn"].time_steps: time_steps,
							GNN["blanks"]: blanks
						}
					)

				print(
					"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n): ({n},{i})\t| Loss: {loss:.5f}".format(
						timestamp = timestamp(),
						memory = memory_usage(),
						epoch = epoch,
						loss = loss,
						batch = batch_i,
						n = n,
						i = 1
					),
					flush = True
				)

			#end

		#end
	#end
#end