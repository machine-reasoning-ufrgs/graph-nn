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
from matplotlib.colors import hsv_to_rgb
from matplotlib import colors as mcolors

if __name__ == '__main__':

    d = 128
    time_steps = 45
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
        load_weights(sess,"./TSP-checkpoints-{}".format(loss_type))

        k = 3
        # Run k² tests
        n = 20
        bins = 10**3
        for inst_i in range(k**2):

            # Create metric TSP instance
            Ma, Mw, route, nodes = create_graph_metric(n,bins)

            # Compute list of edges (as 2-uples of node indices)
            edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

            # Get edges mask
            route_edges = get_edges_mask(Ma,route)

            # Convert to 'quiver' format
            M, W = to_quiver(Ma,Mw)

            # Get edges' probabilities
            e_prob, precision, recall = sess.run(
                [ GNN["E_prob"], GNN["precision"], GNN["recall"] ],
                feed_dict = {
                    GNN["gnn"].matrix_placeholders["M"]:    M,
                    GNN["gnn"].matrix_placeholders["W"]:    W,
                    GNN["n_vertices"]:                      [Ma.shape[0]],
                    GNN["n_edges"]:                         [len(edges)],
                    GNN["gnn"].time_steps:                  time_steps,
                    GNN["route_edges"]:                     route_edges
                }
            )

            true_edges = list(zip(route,route[1:]+route[:1])) + list(zip(route[1:]+route[:1],route))
            predicted_edges = [ edges[e] for e in np.nonzero(np.round(e_prob))[0]  ]
            #predicted_edges = [ edges[e] for e in e_prob.argsort()[-n:][::-1]]
            print("Precision, Recall: {}, {}".format(precision,recall))

            nodes[:,:] += 1.25*np.array([inst_i//k,inst_i%k]).astype(float)

            colors = [c for c in mcolors.BASE_COLORS.keys() if c != 'w']

            ## Draw predicted edges
            for (i,j) in predicted_edges:
               x1,x2 = nodes[i,0],nodes[j,0]
               y1,y2 = nodes[i,1],nodes[j,1]
               #if (i,j) not in true_edges:
               plt.plot([x1,x2],[y1,y2], color=colors[inst_i%len(colors)], linestyle='-', linewidth=1.5, zorder=0)
               #end
            #end

            ## Draw true edges
            #for (i,j) in true_edges:
            #    if i < j: continue;
            #    x1,x2 = nodes[i,0],nodes[j,0]
            #    y1,y2 = nodes[i,1],nodes[j,1]
            #
            #    if (i,j) in predicted_edges:
            #       plt.plot([x1,x2],[y1,y2], 'b-')
            #    else:
            #       plt.plot([x1,x2],[y1,y2], 'r--')
            #    #end
            #
            #    #plt.plot([x1,x2],[y1,y2], color=['red','green','blue','black'][inst_i], linestyle='-', linewidth=1.5, zorder=0)
            ##end

            # Draw nodes
            #plt.scatter(x=nodes[:,0], y=nodes[:,1], s=20*np.ones(n), facecolors='w', edgecolors='black', zorder=1)

        #end

        plt.axis('off')
        plt.show()
        #plt.savefig('paper/figures/training-inst-examples.pdf', format='pdf')

    #end

#end