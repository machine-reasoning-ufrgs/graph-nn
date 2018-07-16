
import sys, os
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from graphnn_refactored import GraphNN
from tsp import build_network_v2, ensure_datasets, run_batch_v2, summarize_epoch
from tsp_utils import InstanceLoader, create_graph_metric
from util import load_weights
from sklearn.cluster import KMeans
from matplotlib import colors as mcolors
from matplotlib import cm
from itertools import islice

def extract_solution(sess, model, instance, time_steps=50):

    # Extract list of edges from instance
    Ma,Mw,route,nodes = instance
    edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))

    # Create batch of size 1
    route_cost = sum([ Mw[min(i,j),max(i,j)] for (i,j) in zip(route,route[1:]+route[:1]) ]) / len(nodes)
    target_cost = 2*route_cost
    batch = InstanceLoader.create_batch([(Ma,Mw,route)], target_cost=target_cost)
    M, W, CV, CE, edges_mask, route_exists, n_vertices, n_edges = batch

    # Define feed dict
    feed_dict = {
        model['gnn'].matrix_placeholders['M']: M,
        model['gnn'].matrix_placeholders['W']: W,
        model['gnn'].matrix_placeholders['CV']: CV,
        model['gnn'].matrix_placeholders['CE']: CE,
        model['gnn'].time_steps: time_steps,
        model['route_exists']: route_exists,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges
    }

    # Run model to extract edge embeddings
    edge_embeddings, predictions = sess.run([model['gnn'].last_states['E'].h, model['predictions']], feed_dict = feed_dict)

    # Perform 2-clustering
    two_clustering = KMeans(n_clusters=2).fit(edge_embeddings)
    if sum(two_clustering.labels_) < len(edges)//2:
        pos_indices = [ i for i,x in enumerate(two_clustering.labels_) if x==1 ]
        pos_center = two_clustering.cluster_centers_[0]
    else:
        pos_indices = [ i for i,x in enumerate(two_clustering.labels_) if x==0 ]
        pos_center = two_clustering.cluster_centers_[1]
    #end

    # Get the indices n positive embeddings closest to their center
    top_n = sorted(pos_indices, key=lambda i: LA.norm(edge_embeddings[i,:]-pos_center) )#[:n]

    print('')
    print('Is there a route with cost < {target_cost:.3f}? R: {R}'.format(target_cost=target_cost, R='Yes' if route_exists[0]==1 else 'No'))
    print('Prediction: {}'.format('Yes' if predictions[0] >= 0.5 else 'No'))

    # Get list of predicted edges
    predicted_edges = [ (i,j) for e,(i,j) in enumerate(edges) if e in top_n ]

    return predicted_edges
#end

def draw_routes():
    d = 128
    n = 20
    bins = 10**6
    connectivity = 1

    # Build model
    print('Building model ...')
    model = build_network_v2(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session(config=config) as sess:

        # Initialize global variables
        print('Initializing global variables ...')
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,'./TSP-checkpoints-decision')

        k = 3

        # Normalize item number values to colormap
        norm = mcolors.Normalize(vmin=0, vmax=k**2)

        for inst in range(k**2):

            instance = create_graph_metric(n,bins,connectivity)
            Ma,_,_,nodes = instance
            edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))

            predicted_edges = extract_solution(sess,model,instance)

            nodes_ = nodes + np.array([inst//k,inst%k])

            for (i,j) in edges:
                x0,y0 = nodes_[i]
                x1,y1 = nodes_[j]
                if (i,j) in predicted_edges:
                    plt.plot([x0,x1],[y0,y1], color = cm.jet(norm(inst)), linewidth=1, zorder=2)
                else:
                    dummy = 0
                    #plt.plot([x0,x1],[y0,y1], 'r--', linewidth=0.5, zorder=1)
                #end
            #end

            plt.scatter(x=nodes_[:,0], y=nodes_[:,1], edgecolors='black', c='w', zorder=3)
        #end

        plt.show()
    #end
#end

def test(time_steps=25):

    d                       = 128

    epochs_n                = 100
    batch_size              = 1
    test_batches_per_epoch  = 16*32

    # Create train and test loaders
    train_loader    = InstanceLoader("train", target_cost_dev=0.25*0.04)
    test_loader     = InstanceLoader("test", target_cost_dev=0.25*0.04)

    # Build model
    print("Building model ...", flush=True)
    GNN = build_network_v2(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {"GPU":0})
    with tf.Session(config=config) as sess:

        # Initialize global variables
        print("Initializing global variables ... ", flush=True)
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,'./TSP-checkpoints-decision')
        
        with open('TSP-log.dat','w') as logfile:
            # Run for a number of epochs
            for epoch_i in range(1):

                test_loader.reset()

                test_loss   = np.zeros(test_batches_per_epoch)
                test_acc    = np.zeros(test_batches_per_epoch)
                test_sat    = np.zeros(test_batches_per_epoch)
                test_pred   = np.zeros(test_batches_per_epoch)

                print("Testing model...", flush=True)
                for (batch_i, batch) in islice(enumerate(test_loader.get_batches(batch_size)), test_batches_per_epoch):
                    test_loss[batch_i], test_acc[batch_i], test_sat[batch_i], test_pred[batch_i] = run_batch_v2(sess, GNN, batch, batch_i, epoch_i, time_steps, train=False, verbose=True)
                #end
                summarize_epoch(epoch_i,test_loss,test_acc,test_sat,test_pred,train=False)
            #end
        #end
    #end
#end

if __name__ == '__main__':
    test(time_steps=sys.argv[1])
#end