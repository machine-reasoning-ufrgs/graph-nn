import sys, os, time
import tensorflow as tf
import numpy as np
import random
from itertools import islice
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import model builder
from graphnn import GraphNN
from mlp import Mlp
from util import timestamp, memory_usage, dense_to_sparse, load_weights, save_weights
from tsp_utils import InstanceLoader, create_graph_metric, get_edges_mask, to_quiver
from tsp import build_network
from matplotlib import pyplot as plt

if __name__ == '__main__':

	d = 128
	time_steps = 32
	loss_type = "edges"

	# Build model
	print( "{timestamp}\t{memory}\tBuilding model ...".format( timestamp = timestamp(), memory = memory_usage() ) )
	GNN = build_network(d)

	# Disallow GPU use
	config = tf.ConfigProto( device_count = {"GPU":0})
	with tf.Session(config=config) as sess:

		# Initialize global variables
		print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
		sess.run( tf.global_variables_initializer() )

		# Restore saved weights
		load_weights(sess,"./TSP-checkpoints-{} BACKUP".format(loss_type))

		# Run 4 tests
		n = 20
		bins = 10
		for i in range(1):

			# Create metric TSP instance
			Ma, Mw, route, nodes = create_graph_metric(n,bins)

			# Compute list of edges (as 2-uples of node indices)
			edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

			# Get edges mask
			route_edges = get_edges_mask(Ma,route)

			# Convert to 'quiver' format
			M, W = to_quiver(Ma,Mw)

			# Get edges' probabilities
			e_prob, pacc, nacc = sess.run(
				[ GNN["E_prob"], GNN["pos_edges_acc"], GNN["neg_edges_acc"] ],
				feed_dict = {
					GNN["gnn"].matrix_placeholders["M"]:	M,
					GNN["gnn"].matrix_placeholders["W"]:	10*W,
					GNN["n_vertices"]:						[Ma.shape[0]],
					GNN["n_edges"]:							[len(edges)],
					GNN["gnn"].time_steps: 					time_steps,
					GNN["route_edges"]:						route_edges
				}
			)

			true_edges = list(zip(route,route[1:]+[route[0]]))
			predicted_edges = [ edges[e] for e in np.nonzero(np.round(e_prob))[0] ]
			#predicted_edges = [ edges[e] for e in e_prob.argsort()[-20:][::-1] ]

			# Draw nodes
			plt.scatter(x=nodes[:,0], y=nodes[:,1])

			# Draw predicted edges
			for (i,j) in predicted_edges:
				x1,x2 = nodes[i,0],nodes[j,0]
				y1,y2 = nodes[i,1],nodes[j,1]
				plt.plot([x1,x2],[y1,y2], 'r--', linewidth=0.5)
			#end

			# Draw true edges
			#for (i,j) in true_edges:
			#	x1,x2 = nodes[i,0],nodes[j,0]
			#	y1,y2 = nodes[i,1],nodes[j,1]

			#	if (i,j) in predicted_edges:
			#		plt.plot([x1,x2],[y1,y2], 'b-')
			#	else:
			#		plt.plot([x1,x2],[y1,y2], 'r--')
			#	#end
			##end

			plt.show()

		#end
	#end

#end