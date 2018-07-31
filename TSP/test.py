
import sys, os
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))

import tensorflow as tf
import numpy as np
from numpy import linalg as LA
from matplotlib import pyplot as plt
from graphnn_refactored import GraphNN
from model import build_network
from train import ensure_datasets, run_batch, summarize_epoch
from tsp_utils import InstanceLoader, create_graph_metric, create_dataset_metric
from util import load_weights
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.decomposition import PCA
from matplotlib.colors import Colormap
from matplotlib import cm
from itertools import islice

def extract_embeddings_and_predictions(sess, model, instance, time_steps=32, target_cost_dev=0.05):

    Ma,Mw,route,nodes = instance
    n = Ma.shape[0]

    # Create batch of size 1
    route_cost = sum([ Mw[min(i,j),max(i,j)] for (i,j) in zip(route,route[1:]+route[:1]) ]) / n
    target_cost = (1+target_cost_dev)*route_cost
    batch = InstanceLoader.create_batch([(Ma,Mw,route)], target_cost=target_cost)
    EV, W, C, edges_mask, route_exists, n_vertices, n_edges = batch

    # Define feed dict
    feed_dict = {
        model['gnn'].matrix_placeholders['EV']: EV,
        model['gnn'].matrix_placeholders['W']: W,
        model['route_costs']: C,
        model['gnn'].time_steps: time_steps,
        model['route_exists']: route_exists,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges
    }

    # Run model to extract edge embeddings
    vertex_embeddings, edge_embeddings, predictions = sess.run([model['gnn'].last_states['V'].h, model['gnn'].last_states['E'].h, model['predictions']], feed_dict = feed_dict)

    return vertex_embeddings, edge_embeddings, predictions
#end

def get_k_cluster(embeddings, k):

    # Perform k clustering
    k_clustering = KMeans(n_clusters=k).fit(embeddings)

    # Organize into list of (k) clusters
    clusters = [ [ j for j,x in enumerate(k_clustering.labels_) if x==i] for i in range(k) ]

    return clusters, k_clustering.cluster_centers_
#end

def get_projections(embeddings, k):
    # Given a list of n-dimensional vertex embeddings, project them into k-dimensional space

    # Standarize dataset onto the unit scale (mean = 0, variance = 1)
    embeddings = StandardScaler().fit_transform(embeddings)

    # Get principal components
    principal_components = PCA(n_components=k).fit_transform(embeddings)

    return principal_components
#end

def extract_solution(sess, model, instance, time_steps=10):

    # Extract list of edges from instance
    Ma,Mw,route,nodes = instance
    edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))
    n = Ma.shape[0]
    m = len(edges)

    # Create batch of size 1
    route_cost = sum([ Mw[min(i,j),max(i,j)] for (i,j) in zip(route,route[1:]+route[:1]) ]) / n
    target_cost = 1.05*route_cost
    batch = InstanceLoader.create_batch([(Ma,Mw,route)], target_cost=target_cost)
    EV, W, C, edges_mask, route_exists, n_vertices, n_edges = batch

    # Define feed dict
    feed_dict = {
        model['gnn'].matrix_placeholders['EV']: EV,
        model['gnn'].matrix_placeholders['W']: W,
        model['route_costs']: C,
        model['gnn'].time_steps: time_steps,
        model['route_exists']: route_exists,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges
    }

    # Run model to extract edge embeddings
    edge_embeddings, predictions = sess.run([model['gnn'].last_states['E'].h, model['predictions']], feed_dict = feed_dict)

    # Perform 2-clustering
    two_clustering = KMeans(n_clusters=2).fit(edge_embeddings)
    if abs(sum(two_clustering.labels_)-n) < abs(sum(1-two_clustering.labels_)-n):
        pos_indices = [ i for i,x in enumerate(two_clustering.labels_) if x==1 ]
        pos_center = two_clustering.cluster_centers_[0]
    else:
        pos_indices = [ i for i,x in enumerate(two_clustering.labels_) if x==0 ]
        pos_center = two_clustering.cluster_centers_[1]
    #end

    print('# pos_indices, # neg_indices: {},{}'.format(len(pos_indices), edge_embeddings.shape[0]-len(pos_indices)))

    # Get the indices of the n positive embeddings closest to their center
    top_n = sorted(pos_indices, key=lambda i: LA.norm(edge_embeddings[i,:]-pos_center) )#[:n]

    print('')
    print('Is there a route with cost < {target_cost:.3f}? R: {R}'.format(target_cost=target_cost, R='Yes' if route_exists[0]==1 else 'No'))
    print('Prediction: {}'.format('Yes' if predictions[0] >= 0.5 else 'No'))

    # Get list of predicted edges
    predicted_edges = [ (i,j) for e,(i,j) in enumerate(edges) if e in top_n ]

    return predicted_edges
#end

def draw_projections():
    
    d = 64
    n = 20
    bins = 10**6
    connectivity = 1

    # Build model
    print('Building model ...')
    model = build_network(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session(config=config) as sess:

        # Initialize global variables
        print('Initializing global variables ...')
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,'./TSP-checkpoints-decision-0.05/epoch=100.0')

        target_cost_dev = +0.1

        # Create instance
        while True:
            instance = create_graph_metric(n,bins,connectivity)
            Ma,Mw,_,nodes = instance
            edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))
            edge_weights = [ Mw[i,j] for (i,j) in edges ]

            _,_, predictions = extract_embeddings_and_predictions(sess, model, instance, time_steps=32, target_cost_dev=target_cost_dev)
        
            if predictions[0] > 0.7:
                break
            #end
        #end

        # Define timesteps range
        timesteps = np.arange(20,32+1,4)
        # Init figure
        f, axes = plt.subplots(1, len(timesteps), dpi=200, sharex=True, sharey=True)
        # Iterate over timesteps
        for i,(t,ax) in enumerate(zip(timesteps,axes)):
            
            # Fetch embeddings and predictions
            vertex_embeddings, edge_embeddings, predictions = extract_embeddings_and_predictions(sess, model, instance, time_steps=t, target_cost_dev=target_cost_dev)

            # Compute accuracy
            acc = 100*( (target_cost_dev > 0) == (predictions[0] > 0.5) ).astype(float)

            # Obtain 2D vertex embedding projections
            vertex_projections = get_projections(vertex_embeddings,2)
            # Obtain 2D edge embedding projections
            edge_projections = get_projections(edge_embeddings,2)

            # Set subplot title
            ax.set_title('{t} steps\npred:{pred:.0f}%'.format(t=t,acc=acc,pred=100*predictions[0]))

            # Plot projections
            #ax.scatter(vertex_projections[:,0],vertex_projections[:,1], edgecolors='black')
            ax.scatter(edge_projections[:,0],edge_projections[:,1], edgecolors='black', c=edge_weights, cmap='jet')

        #end

        plt.show()

    #end
#end

def draw_routes():
    
    d = 64
    n = 20
    bins = 10**6
    connectivity = 1

    # Build model
    print('Building model ...')
    model = build_network(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {'GPU':0})
    with tf.Session(config=config) as sess:

        # Initialize global variables
        print('Initializing global variables ...')
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,'./TSP-checkpoints-decision-0.05/epoch=200.0')

        target_cost_dev = +0.1

        # Create instance
        while True:
            instance = create_graph_metric(n,bins,connectivity)
            Ma,Mw,route,nodes = instance
            edges = list(zip(np.nonzero(Ma)[0],np.nonzero(Ma)[1]))
            edge_weights = [ Mw[i,j] for (i,j) in edges ]

            _,_, predictions = extract_embeddings_and_predictions(sess, model, instance, time_steps=32, target_cost_dev=target_cost_dev)
        
            if predictions[0] < 1:
                break
            #end
        #end

        # Define timesteps range
        timesteps = np.arange(20,32+1,4)
        # Init figure
        f, axes = plt.subplots(1, len(timesteps), dpi=200, sharex=True, sharey=True)
        # Iterate over timesteps
        for i,(t,ax) in enumerate(zip(timesteps,axes)):
            
            # Fetch embeddings and predictions
            vertex_embeddings, edge_embeddings, predictions = extract_embeddings_and_predictions(sess, model, instance, time_steps=t, target_cost_dev=target_cost_dev)

            # Obtain 2D vertex embedding projections
            vertex_projections = get_projections(vertex_embeddings,2)
            # Obtain 2D edge embedding projections
            edge_projections = get_projections(edge_embeddings,2)

            # Obtain 2-clustering
            clusters, cluster_centers = get_k_cluster(edge_embeddings,2)

            print('#1 Cluster size, #2 Cluster size: {},{}'.format(len(clusters[0]),len(clusters[1])))

            # Set subplot title
            ax.set_title('{t} steps\npred:{pred:.0f}%'.format(t=t,pred=100*predictions[0]))

            if len(clusters[0]) < len(clusters[1]):
                clusters = clusters[::-1]
                cluster_centers = cluster_centers[::-1]
            #end

            # Plot edges
            for k in range(1):
                color = ['red','blue'][k]
                for e,(i,j) in enumerate(edges):
                    if e in clusters[k]:
                        x0,y0 = nodes[i,:]
                        x1,y1 = nodes[j,:]
                        ax.plot([x0,x1],[y0,y1], c=color, linewidth=0.5)
                    #end
                #end
            #end

            edge_in_route = []
            edge_not_in_route = []
            for e,(i,j) in enumerate(edges):
                if (i,j) in zip(route,route[:1]+route[1:]):
                    edge_in_route.append(e)
                else:
                    edge_not_in_route.append(e)
                #end
            #end
            
            ax.scatter(edge_projections[edge_not_in_route,0],edge_projections[edge_not_in_route,1], c='red', edgecolors='black')
            ax.scatter(edge_projections[edge_in_route,0],edge_projections[edge_in_route,1], c='blue', edgecolors='black')

        #end

        plt.show()

    #end
#end

def test(time_steps=32, target_cost_dev=0.05):

    test_samples = 32*32

    if not os.path.isdir('test-complete'):
        print('Creating {} Complete Test instances'.format(test_samples), flush=True)
        create_dataset_metric(
            20, 20,
            1, 1,
            bins=10**6,
            samples=test_samples,
            path='test-complete')
    #end

    d                       = 64
    epochs_n                = 100
    batch_size              = 1
    test_batches_per_epoch  = 16*32

    # Create test loader
    test_loader = InstanceLoader("test-complete")

    # Build model
    print("Building model ...", flush=True)
    GNN = build_network(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {"GPU":0})
    with tf.Session(config=config) as sess:

        # Initialize global variables
        print("Initializing global variables ... ", flush=True)
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        load_weights(sess,'./TSP-checkpoints-decision-0.05')
        
        with open('TSP-log.dat','w') as logfile:
            # Run for a number of epochs
            for epoch_i in range(1):

                test_loader.reset()

                test_loss   = np.zeros(test_batches_per_epoch)
                test_acc    = np.zeros(test_batches_per_epoch)
                test_sat    = np.zeros(test_batches_per_epoch)
                test_pred   = np.zeros(test_batches_per_epoch)

                print("Testing model...", flush=True)
                for (batch_i, batch) in islice(enumerate(test_loader.get_batches(batch_size, target_cost_dev=target_cost_dev)), test_batches_per_epoch):
                    test_loss[batch_i], test_acc[batch_i], test_sat[batch_i], test_pred[batch_i] = run_batch(sess, GNN, batch, batch_i, epoch_i, time_steps, train=False, verbose=True)[:4]
                #end
                summarize_epoch(epoch_i,test_loss,test_acc,test_sat,test_pred,train=False)
            #end
        #end
    #end
#end

if __name__ == '__main__':
    draw_routes()
    #test()
#end