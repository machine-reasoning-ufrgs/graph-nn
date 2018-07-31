
import sys, os
import tensorflow as tf

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from graphnn_refactored import GraphNN
from mlp import Mlp

def build_network(d):

    # Define hyperparameters
    d = d
    learning_rate = 2e-5
    l2norm_scaling = 1e-10
    global_norm_gradient_clipping_ratio = 0.65

    # Define a placeholder for the answers to the decision problems
    route_exists = tf.placeholder( tf.float32, shape = (None,), name = 'route_exists' )
    # Define a placeholder for the cost of each route
    route_costs = tf.placeholder( tf.float32, shape=(None,1), name='route_costs')
    # Define a placeholder for the edges mask
    edges_mask = tf.placeholder( tf.float32, shape = (None,), name = 'edges_mask' )
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
            'EV': ('E','V'),
            # W is a column matrix of shape |E|×1 where W[i,1] is the weight of the i-th edge
            'W': ('E',1),
            # C is a column matrix of shape |E|×1 mapping each edge to its corresponding target cost
            'C': ('E',1)
        },
        {
            # V_msg_E is a MLP which computes messages from vertex embeddings to edge embeddings
            'V_msg_E': ('V','E'),
            # E_msg_V is a MLP which computes messages from edge embeddings to vertex embeddings
            'E_msg_V': ('E','V')
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
            # E(t+1) ← Eu( EV × V_msg_E(V(t)), W, C )
            'E': [
                {
                    'mat': 'EV',
                    'msg': 'V_msg_E',
                    'var': 'V'
                },
                {
                    'mat': 'W'
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
    GNN['route_costs']  = route_costs
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
    vote_bias = tf.get_variable(initializer=tf.zeros_initializer(), shape=(), dtype=tf.float32, name='vote_bias')

    # Get the last embeddings
    E_n = gnn.last_states['E'].h
    # Compute a vote for each embedding
    #E_vote = tf.reshape(E_vote_MLP(tf.concat([E_n,route_costs], axis=1)), [-1])
    E_vote = tf.reshape(E_vote_MLP(E_n), [-1])
    E_prob = tf.sigmoid(E_vote)

    # Compute the number of problems in the batch
    num_problems = tf.shape(n_vertices)[0]
    # n_edges_acc[i] is the number of edges in the batch up to the i-th instance
    n_edges_acc = tf.map_fn(lambda i: tf.reduce_sum(tf.gather(n_edges, tf.range(0,i))), tf.range(0,num_problems))

    # Compute decision predictions (one per problem)
    _, pred_logits = tf.while_loop(
        lambda i, predictions: tf.less(i, num_problems),
        lambda i, predictions:
            (
                (i+1),
                predictions.write(
                    i,
                    #tf.reduce_mean(tf.gather(E_vote, tf.range(n_edges_acc[i], n_edges_acc[i] + n_edges[i])))
                    tf.reduce_mean( E_vote[n_edges_acc[i]:n_edges_acc[i]+n_edges[i]] )
                )
            ),
        [0, tf.TensorArray(size=num_problems, dtype=tf.float32)]
        )
    pred_logits = pred_logits.stack() + vote_bias
    GNN['predictions'] = tf.sigmoid(pred_logits)

    # Count the number of edges that appear in the solution
    pos_edges_n = tf.reduce_sum(edges_mask)
    # Count the number of edges that do not appear in the solution
    neg_edges_n = tf.reduce_sum(tf.subtract(tf.ones_like(edges_mask), edges_mask))
    # Define edges loss
    GNN['loss_edges'] = tf.losses.sigmoid_cross_entropy(
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
    # Define decision loss
    GNN['loss_decision'] = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=route_exists, logits=pred_logits))

    # Compute true positives, false positives, true negatives, false negatives
    GNN['true_pos']     = tf.reduce_sum(tf.multiply(route_exists, tf.cast(tf.equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['false_pos']    = tf.reduce_sum(tf.multiply(route_exists, tf.cast(tf.not_equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['true_neg']     = tf.reduce_sum(tf.multiply(1-route_exists, tf.cast(tf.equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))
    GNN['false_neg']    = tf.reduce_sum(tf.multiply(1-route_exists, tf.cast(tf.not_equal(route_exists, tf.round(GNN['predictions'])), tf.float32)))

    # Define edges accuracy
    GNN['acc_edges'] = tf.reduce_mean(tf.cast(tf.equal(edges_mask, tf.round(E_prob)), tf.float32))
    # Define decision accuracy
    GNN['acc_decision'] = tf.reduce_mean(tf.cast(tf.equal(route_exists, tf.round(GNN['predictions'])), tf.float32))

    # Define optimizer
    optimizer = tf.train.AdamOptimizer(name='Adam', learning_rate=learning_rate)

    # Compute cost relative to L2 normalization
    vars_cost = tf.add_n([ tf.nn.l2_loss(var) for var in tf.trainable_variables() ])
    
    # Define gradients and train step
    for loss_type in ['edges','decision']:
        grads, _ = tf.clip_by_global_norm(tf.gradients(GNN['loss_' + loss_type] + tf.multiply(vars_cost, l2norm_scaling),tf.trainable_variables()),global_norm_gradient_clipping_ratio)
        GNN['train_step_' + loss_type] = optimizer.apply_gradients(zip(grads, tf.trainable_variables()))
    #end
    
    # Return GNN dictionary
    return GNN
#end