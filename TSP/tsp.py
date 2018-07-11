#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, time
import tensorflow as tf
import numpy as np
import random
from itertools import islice
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import model builder
from graphnn_refactored import GraphNN
from mlp import Mlp
from util import timestamp, memory_usage, dense_to_sparse, load_weights, save_weights
from tsp_utils import InstanceLoader, create_dataset_metric

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def build_network(d = 128):

    # Define hyperparameters
    learning_rate = 2e-5
    l2norm_scaling = 1e-10
    global_norm_gradient_clipping_ratio = 0.65

    # Define placeholder for routes' edges (a mask of edges per problem)
    edges_mask = tf.placeholder( tf.float32, [ None ], name = 'edges_mask' )
    # Define placeholders for the list of number of vertices and edges per instance
    n_vertices  = tf.placeholder( tf.int32, shape = (None,), name = 'n_vertices')
    n_edges     = tf.placeholder( tf.int32, shape = (None,), name = 'edges')

    # Define GNN dictionary
    GNN = {}

    # Define Graph neural network
    gnn = GraphNN(
        {
            # V is the set of vertices
            "V": d,
            # E is the set of edges
            "E": d
        },
        {
            # M is a E×V adjacency matrix connecting each edge to the vertices it is connected to
            "M": ("E","V"),
            # W is a column matrix of shape |E|×1 where W[i,1] is the weight of the i-th edge
            "W": ("E",1)
        },
        {
            # Vmsg is a MLP which computes messages from vertex embeddings to edge embeddings
            "Vmsg": ("V","E"),
            # Emsg is a MLP which computes messages from edge embeddings to vertex embeddings
            "Emsg": ("E","V")
        },
        {
            # V(t+1) ← Vu( Mᵀ × Emsg(E(t)) )
            "V": [
                {
                    "mat": "M",
                    "msg": "Emsg",
                    "transpose?": True,
                    "var": "E"
                }
            ],
            # E(t+1) ← Eu( M × Vmsg(V(t)), W )
            "E": [
                {
                    "mat": "M",
                    "msg": "Vmsg",
                    "var": "V"
                },
                {
                    "mat": "W"
                }
            ]
        },
        name="TSP"
    )

    # Populate GNN dictionary
    GNN['gnn']          = gnn
    GNN['edges_mask']   = edges_mask
    GNN['n_vertices']   = n_vertices
    GNN['n_edges']      = n_edges

    # Define E_vote, which will compute one logit for each edge
    E_vote_MLP = Mlp(
        layer_sizes = [ d for _ in range(3) ],
        activations = [ tf.nn.relu for _ in range(3) ],
        output_size = 1,
        name = 'E_vote',
        name_internal_layers = True,
        kernel_initializer = tf.contrib.layers.xavier_initializer(),
        bias_initializer = tf.zeros_initializer()
        )

    # Get the last embeddings
    E_n = gnn.last_states['E'].h
    # Compute a vote for each embedding
    E_vote = tf.reshape(E_vote_MLP(E_n), [-1])
    # For each edge, compute a probability that it belongs to the optimal TSP route
    GNN['E_prob'] = tf.sigmoid(E_vote)

    # Compute the number of problems in the batch
    num_problems = tf.shape(n_vertices)[0]
    # n_edges_acc[i] = ∑{i=0..i}(n_edges[i])
    n_edges_acc = tf.map_fn(lambda i: tf.reduce_sum(tf.gather(n_edges, tf.range(0,i))), tf.range(0,num_problems))
    # Compute the true and predicted cost for each problem in the batch
    _, true_costs, predicted_costs_fuzzy, predicted_costs_binary = tf.while_loop(
      lambda i, true_costs, predicted_costs_fuzzy, predicted_costs_binary: tf.less(i, num_problems),
      lambda i, true_costs, predicted_costs_fuzzy, predicted_costs_binary:
        (
            (i+1),
            true_costs.write(
                i,
                tf.reduce_sum(
                    tf.multiply(
                        tf.gather(edges_mask, tf.range(n_edges_acc[i], n_edges_acc[i] + n_edges[i])),
                        tf.gather(gnn.matrix_placeholders['W'], tf.range(n_edges_acc[i], n_edges_acc[i] + n_edges[i]))
                    )
                )
            ),
            predicted_costs_fuzzy.write(
                i,
                tf.reduce_sum(
                    tf.multiply(
                        tf.gather(GNN['E_prob'], tf.range(n_edges_acc[i], n_edges_acc[i] + n_edges[i])),
                        tf.gather(gnn.matrix_placeholders['W'], tf.range(n_edges_acc[i], n_edges_acc[i] + n_edges[i]))
                    )
                )
            ),
            predicted_costs_binary.write(
                i,
                tf.reduce_sum(
                    tf.multiply(
                        tf.gather(tf.round(GNN['E_prob']), tf.range(n_edges_acc[i], n_edges_acc[i] + n_edges[i])),
                        tf.gather(gnn.matrix_placeholders['W'], tf.range(n_edges_acc[i], n_edges_acc[i] + n_edges[i]))
                    )
                )
            )
        ),
        [tf.constant(0), tf.TensorArray(size=num_problems, dtype=tf.float32), tf.TensorArray(size=num_problems, dtype=tf.float32), tf.TensorArray(size=num_problems, dtype=tf.float32)]
    )
    true_costs, predicted_costs_fuzzy, predicted_costs_binary = true_costs.stack(), predicted_costs_fuzzy.stack(), predicted_costs_binary.stack()
    # Compute loss as the mean squared error (elementwise) between predicted costs and route costs
    GNN['cost_loss'] = tf.losses.mean_squared_error(predicted_costs_fuzzy, true_costs)
    # Compute the average relative deviation between predicted costs and true costs
    GNN['deviation'] = tf.reduce_mean(tf.div(tf.subtract(predicted_costs_binary, true_costs), true_costs))

    # Count the number of edges that appear in the solution
    pos_edges_n = tf.reduce_sum(edges_mask)
    # Count the number of edges that do not appear in the solution
    neg_edges_n = tf.reduce_sum(tf.subtract(tf.ones_like(edges_mask), edges_mask))
    # Compute edges loss
    GNN['edges_loss'] = tf.losses.sigmoid_cross_entropy(
        multi_class_labels  = edges_mask,
        logits              = E_vote,
        weights             = tf.add(
            tf.scalar_mul(
                tf.divide(tf.add(pos_edges_n,neg_edges_n),pos_edges_n),
                edges_mask),
            tf.scalar_mul(
                tf.divide(tf.add(pos_edges_n,neg_edges_n),neg_edges_n),
                tf.subtract(tf.ones_like(edges_mask), edges_mask)
                )
            )
        )

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(name='Adam', learning_rate=learning_rate)

    # Compute cost relative to L2 normalization
    vars_cost = tf.add_n([ tf.nn.l2_loss(var) for var in tf.trainable_variables() ])
    
    # Define gradients and train step
    for loss_type in ['cost','edges']:
        grads, _ = tf.clip_by_global_norm(tf.gradients(GNN[loss_type+'_loss'] + tf.multiply(vars_cost, l2norm_scaling),tf.trainable_variables()),global_norm_gradient_clipping_ratio)
        GNN['train_step_{}'.format(loss_type)] = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
    #end
    
    # Return GNN dictionary
    return GNN
#end

def compute_acc(batch, e_prob):

    # Get features, problem sizes, labels for this batch
    M, W, edges_mask, n_vertices, n_edges = batch

    # Get number of problems in batch
    num_problems = len(n_vertices)

    degree_acc              = np.zeros(num_problems)
    visited_acc             = np.zeros(num_problems)
    conn_comp_acc           = np.zeros(num_problems)
    precision               = np.zeros(num_problems)
    recall                  = np.zeros(num_problems)
    true_negative_rate      = np.zeros(num_problems)

    # For each problem in the batch
    for prob_i in range(num_problems):
        
        n = n_vertices[prob_i]
        m = n_edges[prob_i]
        n_acc = int(np.sum(n_vertices[0:prob_i]))
        m_acc = int(np.sum(n_edges[0:prob_i]))
        
        # Get list of edges for this problem
        edges = [ tuple(np.nonzero(x)[0]-n_acc) for x in M[m_acc:m_acc+m] ]

        # Get list of true edges for this problem
        true_edges = [ tuple(np.nonzero(x)[0]-n_acc) for e,x in enumerate(M[m_acc:m_acc+m]) if edges_mask[e] == 1 ]

        # Get list of predicted edges for this problem
        predicted_edges = [ tuple(np.nonzero(x)[0]-n_acc) for e,x in enumerate(M[m_acc:m_acc+m]) if e_prob[e] > 0.5 ]

        # Compute the degree of each vertex
        degrees = np.array([ len([(x,y) for (x,y) in predicted_edges if i in [x,y]]) for i in range(n) ])

        # Compute the fraction of nodes with the correct degree (which is 2)
        degree_acc[prob_i] = np.mean((degrees == 2).astype(np.float32))

        # Compute the fraction of visited nodes
        visited_acc[prob_i] = np.mean((degrees > 0).astype(np.float32))

        # Compute the number of connected components
        connected_component = np.arange(n)
        for i in range(n):
            # Perform a BFS starting at the i-th vertex
            visited = np.zeros(n,dtype=np.bool)
            current = []
            current.append(i); visited[i] = 1
            while len(current) > 0:
                neighbors = [ j for j in range(n) if not visited[j] and any([ (x in current and j==y) or (y in current and j==x) for (x,y) in predicted_edges ]) ]
                visited[neighbors] = True
                current = neighbors
            #end
            connected_component[[x for x in visited]] = min(connected_component[[x for x in visited]])
        #end
        conn_comp_acc[prob_i] = 1.0 / len(set(connected_component))

        # Compute precision, recall and true negative rate corresponding to
        # the predicted set of edges compared with the true set of edges
        true_positives  = len([ (i,j) for (i,j) in edges if (i,j) in predicted_edges and (i,j) in true_edges ])
        false_positives = len([ (i,j) for (i,j) in edges if (i,j) in predicted_edges and (i,j) not in true_edges ])
        true_negatives  = len([ (i,j) for (i,j) in edges if (i,j) not in predicted_edges and (i,j) not in true_edges ])
        false_negatives = len([ (i,j) for (i,j) in edges if (i,j) not in predicted_edges and (i,j) in true_edges ])
        precision[prob_i]           = true_positives / (true_positives + false_positives) if (true_positives + false_positives) != 0 else float('nan')
        recall[prob_i]              = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) != 0 else float('nan')
        true_negative_rate[prob_i]  = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) != 0 else float('nan')

    #end

    return np.mean(degree_acc), np.mean(visited_acc), np.mean(conn_comp_acc), np.mean(precision), np.mean(recall), np.mean(true_negative_rate)
#end

def build_network_v2(d):

    # Define hyperparameters
    d = d
    learning_rate = 2e-5
    l2norm_scaling = 1e-10
    global_norm_gradient_clipping_ratio = 0.65

    # Define a placeholder for the answers to the decision problems
    route_exists = tf.placeholder( tf.float32, shape = (None), name = 'route_exists' )
    # Define placeholders for the list of number of vertices and edges per instance
    n_vertices  = tf.placeholder( tf.int32, shape = (None,), name = 'n_vertices')
    n_edges     = tf.placeholder( tf.int32, shape = (None,), name = 'edges')

    # Define GNN dictionary
    GNN = {}

    # Define Graph neural network
    gnn = GraphNN(
        {
            # V is the set of vertex embeddings
            'V': d,
            # E is the set of edge embeddings
            'E': d
        },
        {
            # M is a E×V adjacency matrix connecting each edge to the vertices it is connected to
            'M': ('E','V'),
            # W is a column matrix of shape |E|×1 where W[i,1] is the weight of the i-th edge
            'W': ('E',1),
            # CV and CE are column matrices of shapes |V|×1 and |E|×1
            # respectively whose function is to send to each vertex and each
            # edge embedding in the i-th problem the same target route cost
            # c_i (remember: we want to decide whether there is a route with
            # cost ⩽ c)
            'CV': ('V',1),
            'CE': ('E',1)
        },
        {
            # Vmsg is a MLP which computes messages from vertex embeddings to edge embeddings
            'Vmsg': ('V','E'),
            # Emsg is a MLP which computes messages from edge embeddings to vertex embeddings
            'Emsg': ('E','V')
        },
        {
            # V(t+1) ← Vu( Mᵀ × Emsg(E(t)), CV )
            'V': [
                {
                    'mat': 'M',
                    'msg': 'Emsg',
                    'transpose?': True,
                    'var': 'E'
                },
                {
                    'mat': 'CV'
                }
            ],
            # E(t+1) ← Eu( M × Vmsg(V(t)), W, CE )
            'E': [
                {
                    'mat': 'M',
                    'msg': 'Vmsg',
                    'var': 'V'
                },
                {
                    'mat': 'W'
                },
                {
                    'mat': 'CE'
                }
            ]
        },
        name='TSP'
    )

    # Populate GNN dictionary
    GNN['gnn']          = gnn
    GNN['route_exists'] = route_exists
    GNN['n_vertices']   = n_vertices
    GNN['n_edges']      = n_edges

    # Define E_vote, which will compute one logit for each edge
    E_vote_MLP = Mlp(
        layer_sizes = [ d for _ in range(3) ],
        activations = [ tf.nn.relu for _ in range(3) ],
        output_size = 1,
        name = 'E_vote',
        name_internal_layers = True,
        kernel_initializer = tf.contrib.layers.xavier_initializer(),
        bias_initializer = tf.zeros_initializer()
        )

    # Get the last embeddings
    E_n = gnn.last_states['E'].h
    # Compute a vote for each embedding
    E_vote = tf.reshape(E_vote_MLP(E_n), [-1])

    # Compute the number of problems in the batch
    num_problems = tf.shape(n_vertices)[0]
    # n_edges_acc[i] is the number of edges in the batch up to the i-th instance
    n_edges_acc = tf.map_fn(lambda i: tf.reduce_sum(tf.gather(n_edges, tf.range(0,i))), tf.range(0,num_problems))

    # Compute a prediction for each problem in the batch
    _, predictions = tf.while_loop(
        lambda i, predictions: tf.less(i, num_problems),
        lambda i, predictions:
            (
                (i+1),
                predictions.write(
                    i,
                    tf.reduce_mean(tf.gather(E_vote, tf.range(n_edges_acc[i], n_edges_acc[i] + n_edges[i])))
                )
            ),
        [0, tf.TensorArray(size=num_problems, dtype=tf.float32)]
        )
    predictions = predictions.stack()

    # Define loss
    GNN['loss'] = tf.losses.sigmoid_cross_entropy(multi_class_labels=route_exists, logits=predictions)

    # Define accuracy
    GNN['acc'] = tf.reduce_mean(tf.cast(tf.equal(route_exists, tf.round(predictions)), tf.float32))

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(name='Adam', learning_rate=learning_rate)

    # Compute cost relative to L2 normalization
    vars_cost = tf.add_n([ tf.nn.l2_loss(var) for var in tf.trainable_variables() ])
    
    # Define gradients and train step
    grads, _ = tf.clip_by_global_norm(tf.gradients(GNN['loss'] + tf.multiply(vars_cost, l2norm_scaling),tf.trainable_variables()),global_norm_gradient_clipping_ratio)
    GNN['train_step'] = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
    
    # Return GNN dictionary
    return GNN
#end

def run_batch_v2(sess, GNN, batch, batch_i, epoch_i, time_steps, train=False, verbose=True):
    
    """
        Obtain:
            M: an adjacency matrix ∈ {0,1}^(|E|×|V|) between edges and vertices
            W: a column matrix ∈ ℜ^(|E|×1) with the weight of each edge
            edges_mask: a binary mask ∈ {0,1}^|E| marking edges in the solution with 1s and 0s otherwise
            n_vertices: a vector with the number of vertices for each problem in the batch
            n_edges: a vector with the number of edges for each problem in the batch
    """
    M, W, CV, CE, edges_mask, route_exists, n_vertices, n_edges = batch

    # Compute the number of problems
    n_problems = n_vertices.shape[0]

    # Define feed dict
    feed_dict = {
        GNN['gnn'].matrix_placeholders['M']: M,
        GNN['gnn'].matrix_placeholders['W']: W,
        GNN['gnn'].matrix_placeholders['CV']: CV,
        GNN['gnn'].matrix_placeholders['CE']: CE,
        GNN['gnn'].time_steps: time_steps,
        GNN['route_exists']: route_exists,
        GNN['n_vertices']: n_vertices,
        GNN['n_edges']: n_edges
    }

    if train:
        outputs = [GNN['train_step'], GNN['loss'], GNN['acc']]
    else:
        outputs = [GNN['loss'], GNN['acc']]
    #end

    # Run model
    loss, acc = sess.run(outputs, feed_dict = feed_dict)[-2:]

    if verbose:
        # Print stats
        print('Epoch {epoch_i} Batch {batch_i}\t|\t(n,m,batch size)=({n},{m},{batch_size})|\t(Loss,Acc)=({loss:.3f},{acc:.3f})'.format(
            epoch_i = epoch_i,
            batch_i = batch_i,
            loss = loss,
            acc = acc,
            n = np.sum(n_vertices),
            m = np.sum(n_edges),
            batch_size = n_vertices.shape[0]
            )
        )
    #end

    return loss, acc
#end

def main():
    create_datasets     = False
    load_checkpoints    = True
    save_checkpoints    = True

    d                       = 128
    epochs                  = 100
    batch_size              = 32
    train_batches_per_epoch = 128
    test_batches_per_epoch  = 32
    time_steps              = 25
    bins                    = 10**6

    loss_type = sys.argv[1] if len(sys.argv) > 1 and sys.argv[1] in ['cost','edges'] else 'edges'

    if create_datasets:
        nmin, nmax = 20, 40
        conn_min, conn_max = 0.25, 1.0
        train_samples = batch_size*train_batches_per_epoch
        test_samples = batch_size*test_batches_per_epoch
        print("Creating {} train instances...".format(train_samples))
        create_dataset_metric(nmin, nmax, conn_min, conn_max, path="TSP-train", samples=train_samples, bins=bins)
        print("\nCreating {} test instances...".format(test_samples))
        create_dataset_metric(nmin, nmax, conn_min, conn_max, path="TSP-test", samples=test_samples, bins=bins)
        print('\n')
    #end

    # Build model
    print("Building model ...")
    GNN = build_network(d)

    # Create train, test loaders
    train_loader    = InstanceLoader("TSP-train")
    test_loader     = InstanceLoader("TSP-test")

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {"GPU":0})
    with tf.Session(config=config) as sess:
        
        # Initialize global variables
        print("Initializing global variables ... ")
        sess.run( tf.global_variables_initializer() )

        # Restore saved weights
        if load_checkpoints: load_weights(sess,'./TSP-checkpoints-{}'.format(loss_type));

        with open('log-TSP-{}.dat'.format(loss_type),'w') as logfile:
            # Run for a number of epochs
            print("Running for {} epochs\n".format(epochs))
            for epoch in range( epochs ):

                # Reset train and test loaders because we are starting a new epoch
                train_loader.reset()
                test_loader.reset()

                train_stats = {
                    'cost_loss':        np.zeros(train_batches_per_epoch),
                    'edges_loss':       np.zeros(train_batches_per_epoch),
                    'deviation':        np.zeros(train_batches_per_epoch),
                    'degree_acc':       np.zeros(train_batches_per_epoch),
                    'visited_acc':      np.zeros(train_batches_per_epoch),
                    'conn_comp_acc':    np.zeros(train_batches_per_epoch),
                    'precision':        np.zeros(train_batches_per_epoch),
                    'recall':           np.zeros(train_batches_per_epoch),
                    'true_neg':         np.zeros(train_batches_per_epoch),
                }

                test_stats = {
                    'cost_loss':        np.zeros(train_batches_per_epoch),
                    'edges_loss':       np.zeros(train_batches_per_epoch),
                    'deviation':        np.zeros(train_batches_per_epoch),
                    'degree_acc':       np.zeros(train_batches_per_epoch),
                    'visited_acc':      np.zeros(train_batches_per_epoch),
                    'conn_comp_acc':    np.zeros(train_batches_per_epoch),
                    'precision':        np.zeros(train_batches_per_epoch),
                    'recall':           np.zeros(train_batches_per_epoch),
                    'true_neg':         np.zeros(train_batches_per_epoch),
                }

                # Run test batches
                print("Training...")
                print("Perfroming stochastic gradient descent on {} loss".format(loss_type))
                print('--------------------------------------------------------------')
                for (batch_i, batch) in islice(enumerate(train_loader.get_batches(32)), train_batches_per_epoch):

                    # Get features, problem sizes, labels for this batch
                    M, W, edges_mask, n_vertices, n_edges = batch

                    # Run one SGD iteration and fetch loss and cost deviation
                    _, _, train_stats['cost_loss'][batch_i], train_stats['edges_loss'][batch_i], train_stats['deviation'][batch_i], e_prob = sess.run(
                        [ GNN['train_step_{}'.format(loss_type)], GNN['train_step_cost'], GNN['cost_loss'], GNN['edges_loss'], GNN['deviation'], GNN['E_prob'] ],
                        feed_dict = 
                        {
                            GNN['gnn'].matrix_placeholders['M']: M,
                            GNN['gnn'].matrix_placeholders['W']: W,
                            GNN['gnn'].time_steps: time_steps,
                            GNN['n_vertices']: n_vertices,
                            GNN['n_edges']: n_edges,
                            GNN['edges_mask']: edges_mask
                        }
                    )

                    # Obtain degree accuracy, visited accuracy, # connected components, precision, recall and true negative rate
                    train_stats['degree_acc'][batch_i], train_stats['visited_acc'][batch_i], train_stats['conn_comp_acc'][batch_i], train_stats['precision'][batch_i], train_stats['recall'][batch_i], train_stats['true_neg'][batch_i] = compute_acc(batch, e_prob)

                    print('Train Epoch {epoch}\tBatch {batch}\t(n,m,batch size):\t({n},{m},{batch_size})'.format(
                        epoch = epoch,
                        batch = batch_i,
                        n = np.sum(n_vertices),
                        m = np.sum(n_edges),
                        batch_size = batch_size
                        )
                    )
                    print('Cost (Loss,Deviation):\t\t\t\t({loss:.3f},{dev:.3f})'.format(
                        loss = train_stats['cost_loss'][batch_i],
                        dev = train_stats['deviation'][batch_i]
                        )
                    )
                    print('Edges Loss:\t\t\t\t\t{loss:.3f}'.format(
                        loss = train_stats['edges_loss'][batch_i]
                        )
                    )
                    print('(Degree,Visited,Conn.Comp.) Acc:\t\t({deg:.3f},{vis:.3f},{conn:.3f})'.format(
                        deg = train_stats['degree_acc'][batch_i],
                        vis = train_stats['visited_acc'][batch_i],
                        conn = train_stats['conn_comp_acc'][batch_i],
                        )
                    )
                    print('Precision,Recall,True Neg. Rate:\t\t{prec:.3f},{rec:.3f},{tneg:.3f}'.format(
                        prec = train_stats['precision'][batch_i],
                        rec = train_stats['recall'][batch_i],
                        tneg = train_stats['true_neg'][batch_i],
                        )
                    )
                    print('--------------------------------------------------------------')
                #end

                # Print train epoch summary
                print('Train Epoch {epoch} Averages'.format(
                    epoch = epoch
                    )
                )
                print('Cost (Loss,Deviation):\t\t\t\t({loss:.3f},{dev:.3f})'.format(
                    loss = np.mean(train_stats['cost_loss']),
                    dev = np.mean(train_stats['deviation'])
                    )
                )
                print('Edges Loss:\t\t\t\t\t{loss:.3f}'.format(
                    loss = np.mean(train_stats['edges_loss'])
                    )
                )
                print('(Degree,Visited,Conn.Comp.) Acc:\t\t({deg:.3f},{vis:.3f},{conn:.3f})'.format(
                    deg = np.mean(train_stats['degree_acc']),
                    vis = np.mean(train_stats['visited_acc']),
                    conn = np.mean(train_stats['conn_comp_acc'])
                    )
                )
                print('Precision,Recall,True Neg. Rate:\t\t{prec:.3f},{rec:.3f},{tneg:.3f}'.format(
                    prec = np.mean(train_stats['precision']),
                    rec = np.mean(train_stats['recall']),
                    tneg = np.mean(train_stats['true_neg'])
                    )
                )
                print('')

                # Run test batches
                print("Testing...")
                print('--------------------------------------------------------------')
                for (batch_i, batch) in islice(enumerate(test_loader.get_batches(32)), test_batches_per_epoch):

                    # Get features, problem sizes, labels for this batch
                    M, W, edges_mask, n_vertices, n_edges = batch

                    # Fetch loss and cost deviation
                    test_stats['cost_loss'][batch_i], test_stats['edges_loss'][batch_i], test_stats['deviation'][batch_i], e_prob = sess.run(
                        [ GNN['cost_loss'], GNN['edges_loss'], GNN['deviation'], GNN['E_prob'] ],
                        feed_dict = 
                        {
                            GNN['gnn'].matrix_placeholders['M']: M,
                            GNN['gnn'].matrix_placeholders['W']: W,
                            GNN['gnn'].time_steps: time_steps,
                            GNN['n_vertices']: n_vertices,
                            GNN['n_edges']: n_edges,
                            GNN['edges_mask']: edges_mask
                        }
                    )

                    # Obtain degree accuracy, visited accuracy, # connected components, precision, recall and true negative rate
                    test_stats['degree_acc'][batch_i], test_stats['visited_acc'][batch_i], test_stats['conn_comp_acc'][batch_i], test_stats['precision'][batch_i], test_stats['recall'][batch_i], test_stats['true_neg'][batch_i] = compute_acc(batch, e_prob)

                    print('Test Epoch {epoch}\tBatch {batch}\t(n,m,batch size):\t({n},{m},{batch_size})'.format(
                        epoch = epoch,
                        batch = batch_i,
                        n = np.sum(n_vertices),
                        m = np.sum(n_edges),
                        batch_size = batch_size
                        )
                    )
                    print('Cost (Loss,Deviation):\t\t\t\t({loss:.3f},{dev:.3f})'.format(
                        loss = test_stats['cost_loss'][batch_i],
                        dev = test_stats['deviation'][batch_i]
                        )
                    )
                    print('Edges Loss:\t\t\t\t\t{loss:.3f}'.format(
                        loss = test_stats['edges_loss'][batch_i]
                        )
                    )
                    print('(Degree,Visited,Conn.Comp.) Acc:\t\t({deg:.3f},{vis:.3f},{conn:.3f})'.format(
                        deg = test_stats['degree_acc'][batch_i],
                        vis = test_stats['visited_acc'][batch_i],
                        conn = test_stats['conn_comp_acc'][batch_i],
                        )
                    )
                    print('Precision,Recall,True Neg. Rate:\t\t{prec:.3f},{rec:.3f},{tneg:.3f}'.format(
                        prec = test_stats['precision'][batch_i],
                        rec = test_stats['recall'][batch_i],
                        tneg = test_stats['true_neg'][batch_i],
                        )
                    )
                    print('--------------------------------------------------------------')
                #end
                
                # Print test epoch summary
                print('Test Epoch {epoch} Averages'.format(
                    epoch = epoch
                    )
                )
                print('Cost (Loss,Deviation):\t\t\t\t({loss:.3f},{dev:.3f})'.format(
                    loss = test_stats['cost_loss'][batch_i],
                    dev = test_stats['deviation'][batch_i]
                    )
                )
                print('Edges Loss:\t\t\t\t\t{loss:.3f}'.format(
                    loss = test_stats['edges_loss'][batch_i]
                    )
                )
                print('(Degree,Visited,Conn.Comp.) Acc:\t\t({deg:.3f},{vis:.3f},{conn:.3f})'.format(
                    deg = np.mean(test_stats['degree_acc']),
                    vis = np.mean(test_stats['visited_acc']),
                    conn = np.mean(test_stats['conn_comp_acc'])
                    )
                )
                print('Precision,Recall,True Neg. Rate:\t\t{prec:.3f},{rec:.3f},{tneg:.3f}'.format(
                    prec = np.mean(test_stats['precision']),
                    rec = np.mean(test_stats['recall']),
                    tneg = np.mean(test_stats['true_neg'])
                    )
                )
                print('')

                # Save weights
                if save_checkpoints: save_weights(sess,'./TSP-checkpoints-{}'.format(loss_type));

                print('--------------------------------------------------------------------\n')

                logfile.write('{epoch} {tr_cost_loss} {tr_edges_loss} {tr_dev} {tr_precision} {tr_recall} {tr_true_neg} {tst_cost_loss} {tst_edges_loss} {tst_dev} {tst_precision} {tst_recall} {tst_true_neg}\n'.format(
                    epoch = epoch,
                    tr_cost_loss = np.mean(train_stats['cost_loss']),
                    tr_edges_loss = np.mean(train_stats['edges_loss']),
                    tr_dev = np.mean(train_stats['deviation']),
                    tr_precision = np.mean(train_stats['precision']),
                    tr_recall = np.mean(train_stats['recall']),
                    tr_true_neg = np.mean(train_stats['true_neg']),
                    tst_cost_loss = np.mean(test_stats['cost_loss']),
                    tst_edges_loss = np.mean(test_stats['edges_loss']),
                    tst_dev = np.mean(test_stats['deviation']),
                    tst_precision = np.mean(test_stats['precision']),
                    tst_recall = np.mean(test_stats['recall']),
                    tst_true_neg = np.mean(test_stats['true_neg']),
                    )
                )
                logfile.flush()                
            #end
        #end
    #end
#end

def main_v2():

    d = 128
    epochs_n = 100
    train_batches_per_epoch = 128
    test_batches_per_epoch = 32
    time_steps = 25
    target_cost_dev = 1

    # Create train and test loaders
    train_loader    = InstanceLoader("TSP-train", target_cost_dev)
    test_loader     = InstanceLoader("TSP-test", target_cost_dev)

    # Build model
    print("Building model ...")
    GNN = build_network_v2(d)

    # Disallow GPU use
    config = tf.ConfigProto( device_count = {"GPU":0})
    with tf.Session(config=config) as sess:

        # Initialize global variables
        print("Initializing global variables ... ")
        sess.run( tf.global_variables_initializer() )
        
        # Run for a number of epochs
        for epoch_i in range(epochs_n):

            print("Training model...")
            for (batch_i, batch) in islice(enumerate(train_loader.get_batches(32)), train_batches_per_epoch):
                run_batch_v2(sess, GNN, batch, batch_i, epoch_i, time_steps, train=True, verbose=True)
            #end

            print("Testing model...")
            for (batch_i, batch) in islice(enumerate(test_loader.get_batches(32)), test_batches_per_epoch):
                run_batch_v2(sess, GNN, batch, batch_i, epoch_i, time_steps, train=False, verbose=True)
            #end
        #end
    #end
#end

if __name__ == '__main__':
    main_v2()
#end