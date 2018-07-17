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
from functools import reduce

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
                    print('Cost (Loss,Deviation):\t\t\t\t({loss:.4f},{dev:.4f})'.format(
                        loss = train_stats['cost_loss'][batch_i],
                        dev = train_stats['deviation'][batch_i]
                        )
                    )
                    print('Edges Loss:\t\t\t\t\t{loss:.4f}'.format(
                        loss = train_stats['edges_loss'][batch_i]
                        )
                    )
                    print('(Degree,Visited,Conn.Comp.) Acc:\t\t({deg:.4f},{vis:.4f},{conn:.4f})'.format(
                        deg = train_stats['degree_acc'][batch_i],
                        vis = train_stats['visited_acc'][batch_i],
                        conn = train_stats['conn_comp_acc'][batch_i],
                        )
                    )
                    print('Precision,Recall,True Neg. Rate:\t\t{prec:.4f},{rec:.4f},{tneg:.4f}'.format(
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
                print('Cost (Loss,Deviation):\t\t\t\t({loss:.4f},{dev:.4f})'.format(
                    loss = np.mean(train_stats['cost_loss']),
                    dev = np.mean(train_stats['deviation'])
                    )
                )
                print('Edges Loss:\t\t\t\t\t{loss:.4f}'.format(
                    loss = np.mean(train_stats['edges_loss'])
                    )
                )
                print('(Degree,Visited,Conn.Comp.) Acc:\t\t({deg:.4f},{vis:.4f},{conn:.4f})'.format(
                    deg = np.mean(train_stats['degree_acc']),
                    vis = np.mean(train_stats['visited_acc']),
                    conn = np.mean(train_stats['conn_comp_acc'])
                    )
                )
                print('Precision,Recall,True Neg. Rate:\t\t{prec:.4f},{rec:.4f},{tneg:.4f}'.format(
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
                    print('Cost (Loss,Deviation):\t\t\t\t({loss:.4f},{dev:.4f})'.format(
                        loss = test_stats['cost_loss'][batch_i],
                        dev = test_stats['deviation'][batch_i]
                        )
                    )
                    print('Edges Loss:\t\t\t\t\t{loss:.4f}'.format(
                        loss = test_stats['edges_loss'][batch_i]
                        )
                    )
                    print('(Degree,Visited,Conn.Comp.) Acc:\t\t({deg:.4f},{vis:.4f},{conn:.4f})'.format(
                        deg = test_stats['degree_acc'][batch_i],
                        vis = test_stats['visited_acc'][batch_i],
                        conn = test_stats['conn_comp_acc'][batch_i],
                        )
                    )
                    print('Precision,Recall,True Neg. Rate:\t\t{prec:.4f},{rec:.4f},{tneg:.4f}'.format(
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
                print('Cost (Loss,Deviation):\t\t\t\t({loss:.4f},{dev:.4f})'.format(
                    loss = test_stats['cost_loss'][batch_i],
                    dev = test_stats['deviation'][batch_i]
                    )
                )
                print('Edges Loss:\t\t\t\t\t{loss:.4f}'.format(
                    loss = test_stats['edges_loss'][batch_i]
                    )
                )
                print('(Degree,Visited,Conn.Comp.) Acc:\t\t({deg:.4f},{vis:.4f},{conn:.4f})'.format(
                    deg = np.mean(test_stats['degree_acc']),
                    vis = np.mean(test_stats['visited_acc']),
                    conn = np.mean(test_stats['conn_comp_acc'])
                    )
                )
                print('Precision,Recall,True Neg. Rate:\t\t{prec:.4f},{rec:.4f},{tneg:.4f}'.format(
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

###########################
# OLD CODE   ^            #
###########################

###########################
# NEW CODE   v            #
###########################

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
            'E': d,
            # G is the set of graph embeddings (just one)
            'R': 4*d
        },
        {
            # M is a E×V adjacency matrix connecting each edge to the vertices it is connected to
            'EV': ('E','V'),
            # W is a column matrix of shape |E|×1 where W[i,1] is the weight of the i-th edge
            'W': ('E',1),
            # ER is an (fully-connected) adjacency matrix connecting each edge to its corresponding route embedding
            'ER': ('E','R'),
            # C is a column matrix assigned with sending to each route embedding its corresponding target cost
            'C': ('R',1)
        },
        {
            # V_msg_E is a MLP which computes messages from vertex embeddings to edge embeddings
            'V_msg_E': ('V','E'),
            # E_msg_V is a MLP which computes messages from edge embeddings to vertex embeddings
            'E_msg_V': ('E','V'),
            # V_msg_R is a MLP which computes messages from edge embeddings to route embeddings
            # E_msg_R is a MLP which computes messages from edge embeddings to route embeddings
            'E_msg_R': ('E','R'),
            # R_msg_E is a MLP which computes messages from route embeddings to edge embeddings
            'R_msg_E': ('R','E')
        },
        {
            # V(t+1) ← Vu( EVᵀ × E_msg_V(E(t)) )
            'V': [
                {
                    'mat': 'EV',
                    'msg': 'E_msg_V',
                    'transpose?': True,
                    'var': 'E'
                }
            ],
            # E(t+1) ← Eu( EV × V_msg_E(V(t)), ER × R_msg_E(R(t)), W )
            'E': [
                {
                    'mat': 'EV',
                    'msg': 'V_msg_E',
                    'var': 'V'
                },
                {
                    'mat': 'ER',
                    'msg': 'R_msg_E',
                    'var': 'R'
                },
                {
                    'mat': 'W'
                },
            ],
            # R(t+1) ← Ru( ERᵀ × E_msg_R(E(t)), C )
            'R': [
                {
                    'mat': 'ER',
                    'msg': 'E_msg_R',
                    'transpose?': True,
                    'var': 'E'
                },
                {
                    'mat': 'C'
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
    vote_bias = tf.get_variable(initializer=tf.zeros_initializer(), shape=(), dtype=tf.float32, name='vote_bias')

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
    predictions = predictions.stack() + vote_bias
    GNN['predictions'] = tf.sigmoid(predictions)

    # Define loss
    GNN['loss'] = tf.losses.sigmoid_cross_entropy(multi_class_labels=route_exists, logits=predictions)

    # Define accuracy
    GNN['acc'] = tf.reduce_mean(tf.cast(tf.equal(route_exists, tf.round(tf.sigmoid(predictions))), tf.float32))

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

def run_batch_v2(sess, model, batch, batch_i, epoch_i, time_steps, train=False, verbose=True):

    EV, ER, W, C, edges_mask, route_exists, n_vertices, n_edges = batch

    # Compute the number of problems
    n_problems = n_vertices.shape[0]

    # Define feed dict
    feed_dict = {
        model['gnn'].matrix_placeholders['EV']: EV,
        model['gnn'].matrix_placeholders['ER']: ER,
        model['gnn'].matrix_placeholders['C']: C,
        model['gnn'].matrix_placeholders['W']: W,
        model['gnn'].time_steps: time_steps,
        model['route_exists']: route_exists,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges
    }

    if train:
        outputs = [model['train_step'], model['loss'], model['acc'], model['predictions']]
    else:
        outputs = [model['loss'], model['acc'], model['predictions']]
    #end

    # Run model
    loss, acc, predictions = sess.run(outputs, feed_dict = feed_dict)[-3:]

    if verbose:
        # Print stats
        print('{train_or_test} Epoch {epoch_i} Batch {batch_i}\t|\t(n,m,batch size)=({n},{m},{batch_size})\t|\t(Loss,Acc)=({loss:.4f},{acc:.4f})\t|\tAvg. (Sat,Prediction)=({avg_sat:.4f},{avg_pred:.4f})'.format(
            train_or_test = 'Train' if train else 'Test',
            epoch_i = epoch_i,
            batch_i = batch_i,
            loss = loss,
            acc = acc,
            n = np.sum(n_vertices),
            m = np.sum(n_edges),
            batch_size = n_vertices.shape[0],
            avg_sat = np.mean(route_exists),
            avg_pred = np.mean(np.round(predictions))
            ),
            flush = True
        )
    #end

    return loss, acc, np.mean(route_exists), np.mean(predictions)
#end

def summarize_epoch(epoch_i, loss, acc, sat, pred, train=False):
    print('{train_or_test} Epoch {epoch_i} Average\t|\t(Loss,Acc)=({loss:.4f},{acc:.4f})\t|\tAvg. (Sat,Pred)=({avg_sat:.4f},{avg_pred:.4f})'.format(
        train_or_test = 'Train' if train else 'Test',
        epoch_i = epoch_i,
        loss = np.mean(loss),
        acc = np.mean(acc),
        avg_sat = np.mean(sat),
        avg_pred = np.mean(pred)
        ),
        flush = True
    )
#end

def ensure_datasets(nmin, nmax, conn_min, conn_max, bins, batch_size, train_batches_per_epoch, test_batches_per_epoch):

    train_samples = batch_size*train_batches_per_epoch
    test_samples = batch_size*test_batches_per_epoch
    
    if not os.path.isdir('train'):
        print('Creating {} Train instances'.format(train_samples), flush=True)
        create_dataset_metric(nmin, nmax, conn_min, conn_max, path='train', bins=bins, samples=train_samples)
    #end

    if not os.path.isdir('test'):
        print('Creating {} Test instances'.format(test_samples), flush=True)
        create_dataset_metric(nmin, nmax, conn_min, conn_max, path='test', bins=bins, samples=test_samples)
    #end
#end

def main_v2():

    d                       = 128
    time_steps              = 25
    target_cost_dev         = 0.1

    epochs_n                = 100
    batch_size              = 16
    train_batches_per_epoch = 128
    test_batches_per_epoch  = 32
    
    load_checkpoints        = False
    save_checkpoints        = True

    nmin, nmax              = 20, 40
    conn_min, conn_max      = 0.25, 0.25
    bins                    = 10**6
    
    # Ensure that train and test datasets exist and create if inexistent
    ensure_datasets(nmin,nmax,conn_min,conn_max,bins,batch_size,train_batches_per_epoch,test_batches_per_epoch)

    # Create train and test loaders
    train_loader    = InstanceLoader("train", target_cost_dev=target_cost_dev)
    test_loader     = InstanceLoader("test", target_cost_dev=target_cost_dev)

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
        if load_checkpoints: load_weights(sess,'./TSP-checkpoints-decision');
        
        with open('TSP-log.dat','w') as logfile:
            # Run for a number of epochs
            for epoch_i in range(epochs_n):

                train_loader.reset()
                test_loader.reset()

                train_loss  = np.zeros(train_batches_per_epoch)
                train_acc   = np.zeros(train_batches_per_epoch)
                train_sat   = np.zeros(train_batches_per_epoch)
                train_pred  = np.zeros(train_batches_per_epoch)

                test_loss   = np.zeros(test_batches_per_epoch)
                test_acc    = np.zeros(test_batches_per_epoch)
                test_sat    = np.zeros(test_batches_per_epoch)
                test_pred   = np.zeros(test_batches_per_epoch)

                print("Training model...", flush=True)
                for (batch_i, batch) in islice(enumerate(train_loader.get_batches(batch_size)), train_batches_per_epoch):
                    train_loss[batch_i], train_acc[batch_i], train_sat[batch_i], train_pred[batch_i] = run_batch_v2(sess, GNN, batch, batch_i, epoch_i, time_steps, train=True, verbose=True)
                #end
                summarize_epoch(epoch_i,train_loss,train_acc,train_sat,train_pred,train=True)

                print("Testing model...", flush=True)
                for (batch_i, batch) in islice(enumerate(test_loader.get_batches(batch_size)), test_batches_per_epoch):
                    test_loss[batch_i], test_acc[batch_i], test_sat[batch_i], test_pred[batch_i] = run_batch_v2(sess, GNN, batch, batch_i, epoch_i, time_steps, train=False, verbose=True)
                #end
                summarize_epoch(epoch_i,test_loss,test_acc,test_sat,test_pred,train=False)

                # Save weights
                if save_checkpoints: save_weights(sess,'./TSP-checkpoints-decision');

                logfile.write('{epoch_i} {trloss} {tracc} {trsat} {trpred} {tstloss} {tstacc} {tstsat} {tstpred}\n'.format(
                    epoch_i = epoch_i,
                    trloss = np.mean(train_loss),
                    tracc = np.mean(train_acc),
                    trsat = np.mean(train_sat),
                    trpred = np.mean(train_pred),
                    tstloss = np.mean(test_loss),
                    tstacc = np.mean(test_acc),
                    tstsat = np.mean(test_sat),
                    tstpred = np.mean(test_pred),
                    )
                )
                logfile.flush()
            #end
        #end
    #end
#end

if __name__ == '__main__':
    main_v2()
#end