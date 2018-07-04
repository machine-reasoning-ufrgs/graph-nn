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
from tsp_utils import InstanceLoader, create_graph_metric, get_edges_mask, to_quiver, solve
from tsp import build_network
from matplotlib import pyplot as plt
from matplotlib.colors import hsv_to_rgb
from matplotlib import colors as mcolors
import itertools

def test_multiple_TSP_solutions():

    n = 20
    bins = 10**3
    samples = 100
    permutations = 10**3
    for i in range(samples):

        # Select 'n' 2D points in the unit square
        nodes = np.random.rand(n,2)

        # Build a fully connected adjacency matrix
        Ma = np.ones((n,n))-np.eye(n)
        # Build a weight matrix
        Mw = np.zeros((n,n))
        for i in range(n):
            for j in range(n):
                # Multiply by 1/√2 to normalize
                Mw[i,j] = (1.0/np.sqrt(2)) * np.sqrt((nodes[i,0]-nodes[j,0])**2+(nodes[i,1]-nodes[j,1])**2)
            #end
        #end
        # Rescale and round weights, quantizing them into 'bins' integer bins
        Mw = np.round(bins * Mw)

        edges_masks = []

        for perm in itertools.permutations(range(n)):
            
            Ma_perm, Mw_perm = np.zeros((n,n)), np.zeros((n,n))
            for i in range(n):
                for j in range(n):
                    Ma_perm[i,j], Mw_perm[i,j] = Ma[perm[i],perm[j]] , Mw[perm[i],perm[j]]
                #end
            #end

            # Solve
            route = solve(Ma_perm,Mw_perm)

            edges_mask = get_edges_mask(Ma, [ route[perm[i]] for i in range(n) ])
            edges_masks.append(edges_mask)

            if len(set([ str(edges_mask) for edges_mask in edges_masks ])) > 1:
                break
            #end

        #end

        print(len(set([ str(edges_mask) for edges_mask in edges_masks ])))

    #end
#end

def compute_route_greedy(n, edges, e_prob):

    # Init route
    route = []

    # Init visited mask
    visited = np.zeros(n)

    # Choose initial vertex
    init = np.random.randint(n)
    v = init

    # Run n iterations
    for it in range(n-1):

        visited[v] = 1

        # Choose adjacency with highest probability
        adjacencies = [ edges[e] for e in e_prob.argsort()[::-1] if (edges[e][0] == v and visited[edges[e][1]] == 0) or (edges[e][1] == v and visited[edges[e][0]] == 0) ]
        e = adjacencies[0]

        if e[0] == v:
            v2 = e[1]
        else:
            v2 = e[0]
        #end

        route.append(e)
        v = v2
    #end

    return route
#end

def compute_route_nearest_neighbor(n, edges, W):

    # Init route
    route = []

    # Init visited mask
    visited = np.zeros(n)

    # Choose initial vertex
    init = np.random.randint(n)
    v = init

    # Run n iterations
    for it in range(n-1):

        visited[v] = 1

        # Choose adjacency with highest probability
        adjacencies = [ edges[e] for e in W[:,0].argsort() if (edges[e][0] == v and visited[edges[e][1]] == 0) or (edges[e][1] == v and visited[edges[e][0]] == 0) ]
        e = adjacencies[0]

        if e[0] == v:
            v2 = e[1]
        else:
            v2 = e[0]
        #end

        route.append(e)
        v = v2
    #end

    return route
#end

if __name__ == '__main__':

    d = 128
    time_steps = 25
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
        bins = 10**6
        for inst_i in range(k**2):

            # Create metric TSP instance
            Ma, Mw, route, nodes = create_graph_metric(n,bins,connectivity=0.5)

            # Compute list of edges (as 2-uples of node indices)
            edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

            # Get edges mask
            route_edges = get_edges_mask(Ma,route)

            # Convert to 'quiver' format
            M, W, R = to_quiver(Ma,Mw)

            # Get edges' probabilities
            e_prob, precision, recall = sess.run(
                [ GNN["E_prob"], GNN["precision"], GNN["recall"] ],
                feed_dict = {
                    GNN["gnn"].matrix_placeholders["M"]:    M,
                    GNN["gnn"].matrix_placeholders["W"]:    W,
                    GNN["gnn"].matrix_placeholders["R"]:    R,
                    GNN["n_vertices"]:                      [Ma.shape[0]],
                    GNN["n_edges"]:                         [len(edges)],
                    GNN["gnn"].time_steps:                  time_steps,
                    GNN["route_edges"]:                     route_edges
                }
            )

            true_edges = list(zip(route,route[1:]+route[:1])) + list(zip(route[1:]+route[:1],route))
            
            predicted_edges = [ edges[e] for e in np.nonzero(np.round(e_prob))[0]  ]
            #predicted_edges = [ edges[e] for e in e_prob.argsort()[-n:][::-1]]
            #predicted_edges = compute_route_greedy(n,edges,e_prob)
            #NN_edges        = compute_route_nearest_neighbor(n,edges,W)

            true_cost       = np.sum([ W[e,0] for e,(i,j) in enumerate(edges) if (i,j) in true_edges or (j,i) in true_edges ])
            predicted_cost  = np.sum([ W[e,0] for e,(i,j) in enumerate(edges) if (i,j) in predicted_edges or (j,i) in predicted_edges ])
            #NN_cost         = np.sum([ W[e,0] for e,(i,j) in enumerate(edges) if (i,j) in NN_edges or (j,i) in NN_edges ])
            deviation       = (predicted_cost-true_cost) / true_cost
            #deviation_NN    = (NN_cost-true_cost) / true_cost
            
            #print("{} Predicted edges".format(len(predicted_edges)))
            print("Route cost deviation: {dev:.3f}%".format(dev = 100*deviation))
            #print("NN Route cost deviation: {dev:.3f}%".format(dev = 100*deviation_NN))
            #print("Precision, Recall: {}, {}".format(precision,recall))

            nodes[:,:] += 1.25*np.array([inst_i//k,inst_i%k]).astype(float)

            colors = [c for c in mcolors.BASE_COLORS.keys() if c != 'w']

            ### Draw predicted edges
            #for (i,j) in predicted_edges:
            #   x1,x2 = nodes[i,0],nodes[j,0]
            #   y1,y2 = nodes[i,1],nodes[j,1]
            #   #if (i,j) not in true_edges:
            #   plt.plot([x1,x2],[y1,y2], color=colors[inst_i%len(colors)], linestyle='-', linewidth=1, zorder=0)
            #   #end
            ##end

            # Draw true edges
            for (i,j) in true_edges:
                x1,x2 = nodes[i,0],nodes[j,0]
                y1,y2 = nodes[i,1],nodes[j,1]
            
                if (i,j) in predicted_edges or (j,i) in predicted_edges:
                   plt.plot([x1,x2],[y1,y2], 'b-')
                else:
                   plt.plot([x1,x2],[y1,y2], 'r--')
                #end
            #end

            # Draw nodes
            plt.scatter(x=nodes[:,0], y=nodes[:,1], s=20*np.ones(n), facecolors='w', edgecolors='black', zorder=1)

        #end

        plt.axis('off')
        plt.show()
        #plt.savefig('paper/figures/training-inst-examples.pdf', format='pdf')

    #end
#end