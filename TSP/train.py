#!/usr/bin/python
# -*- coding: utf-8 -*-

import sys, os, time, shutil, random, argparse
import tensorflow as tf
import numpy as np
from itertools import islice
from functools import reduce

# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from model import build_network

from util import timestamp, memory_usage, dense_to_sparse, load_weights, save_weights
from tsp_utils import InstanceLoader, create_dataset_metric

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def run_batch(sess, model, batch, batch_i, epoch_i, time_steps, loss_type='decision', train=False, verbose=True):

    EV, W, C, edges_mask, route_exists, n_vertices, n_edges = batch

    # Compute the number of problems
    n_problems = n_vertices.shape[0]

    # Define feed dict
    feed_dict = {
        model['gnn'].matrix_placeholders['EV']: EV,
        model['gnn'].matrix_placeholders['W']: W,
        model['gnn'].matrix_placeholders['C']: C,
        model['gnn'].time_steps: time_steps,
        model['route_exists']: route_exists,
        model['edges_mask']: edges_mask,
        model['n_vertices']: n_vertices,
        model['n_edges']: n_edges
    }

    if train:
        outputs = [model['train_step_'+loss_type], model['loss_'+loss_type], model['acc_'+loss_type], model['predictions'], model['true_pos'], model['false_pos'], model['true_neg'], model['false_neg']]
    else:
        outputs = [model['loss_'+loss_type], model['acc_'+loss_type], model['predictions'], model['true_pos'], model['false_pos'], model['true_neg'], model['false_neg']]
    #end

    # Run model
    loss, acc, predictions, true_pos, false_pos, true_neg, false_neg = sess.run(outputs, feed_dict = feed_dict)[-7:]

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

    return loss, acc, np.mean(route_exists), np.mean(predictions), true_pos, false_pos, true_neg, false_neg
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

def ensure_datasets(batch_size, train_params, test_params):
    
    if not os.path.isdir('train'):
        print('Creating {} Train instances'.format(train_params['samples']), flush=True)
        create_dataset_metric(
            train_params['n_min'], train_params['n_max'],
            train_params['conn_min'], train_params['conn_max'],
            bins=train_params['bins'],
            samples=train_params['samples'],
            path='train')
    #end

    if not os.path.isdir('test'):
        print('Creating {} Test instances'.format(test_params['samples']), flush=True)
        create_dataset_metric(
            test_params['n_min'], test_params['n_max'],
            test_params['conn_min'], test_params['conn_max'],
            bins=test_params['bins'],
            samples=test_params['samples'],
            path='test')
    #end
#end

if __name__ == '__main__':
    
    # Define argument parser
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('-d', default=64, type=int, help='Embedding size for vertices and edges')
    parser.add_argument('-timesteps', default=32, type=int, help='# Timesteps')
    parser.add_argument('-loss', default='decision', help='Loss type (either "decision" or "edges")')
    parser.add_argument('-dev', default=0.05, type=float, help='Target cost deviation')
    parser.add_argument('-epochs', default=32, type=int, help='Training epochs')
    parser.add_argument('-batchsize', default=16, type=int, help='Batch size')
    parser.add_argument('-seed', type=int, default=42, help='RNG seed for Python, Numpy and Tensorflow')
    parser.add_argument('--load', const=True, default=False, action='store_const', help='Load model checkpoint?')
    parser.add_argument('--save', const=True, default=False, action='store_const', help='Save model?')
    parser.add_argument('--newdatasets', const=True, default=False, action='store_const', help='Recreate datasets?')

    # Parse arguments from command line
    args = parser.parse_args()

    # Set RNG seed for Python, Numpy and Tensorflow
    random.seed(vars(args)['seed'])
    np.random.seed(vars(args)['seed'])
    tf.set_random_seed(vars(args)['seed'])

    # Setup parameters
    d                       = vars(args)['d']
    time_steps              = vars(args)['timesteps']
    loss_type               = vars(args)['loss']
    target_cost_dev         = vars(args)['dev']
    epochs_n                = vars(args)['epochs']
    batch_size              = vars(args)['batchsize']
    load_checkpoints        = vars(args)['load']
    save_checkpoints        = vars(args)['save']

    train_params = {
        'n_min': 20,
        'n_max': 40,
        'conn_min': 1,
        'conn_max': 1,
        'bins': 10**6,
        'batches_per_epoch': 128,
        'samples': 2**20
    }

    test_params = {
        'n_min': train_params['n_max'],
        'n_max': 2*train_params['n_max'],
        'conn_min': 1,
        'conn_max': 1,
        'bins': 10**6,
        'batches_per_epoch': 32,
        'samples': 1024
    }

    # Delete datasets if requested
    if vars(args)['newdatasets']:
        shutil.rmtree('train')
        shutil.rmtree('test')
    #end
    
    # Ensure that train and test datasets exist and create if inexistent
    ensure_datasets(batch_size, train_params, test_params)

    # Create train and test loaders
    train_loader    = InstanceLoader("train")
    test_loader     = InstanceLoader("test")

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
        if load_checkpoints: load_weights(sess,'./TSP-checkpoints-{loss_type}-{target_cost_dev}/epoch=100.0'.format(loss_type=loss_type,target_cost_dev=target_cost_dev));
        
        print('Performing Stochastic Gradient Descent on {} loss...'.format(loss_type))

        with open('TSP-log-{loss_type}-{target_cost_dev}.dat'.format(loss_type=loss_type,target_cost_dev=target_cost_dev),'w') as logfile:
            # Run for a number of epochs
            for epoch_i in 100 + np.arange(epochs_n):

                train_loader.reset()
                test_loader.reset()

                train_stats = {
                    'loss': np.zeros(train_params['batches_per_epoch']),
                    'acc': np.zeros(train_params['batches_per_epoch']),
                    'sat': np.zeros(train_params['batches_per_epoch']),
                    'pred': np.zeros(train_params['batches_per_epoch']),
                    'true_pos': np.zeros(train_params['batches_per_epoch']),
                    'false_pos': np.zeros(train_params['batches_per_epoch']),
                    'true_neg': np.zeros(train_params['batches_per_epoch']),
                    'false_neg': np.zeros(train_params['batches_per_epoch']),
                }

                test_stats = {
                    'loss': np.zeros(test_params['batches_per_epoch']),
                    'acc': np.zeros(test_params['batches_per_epoch']),
                    'sat': np.zeros(test_params['batches_per_epoch']),
                    'pred': np.zeros(test_params['batches_per_epoch']),
                    'true_pos': np.zeros(test_params['batches_per_epoch']),
                    'false_pos': np.zeros(test_params['batches_per_epoch']),
                    'true_neg': np.zeros(test_params['batches_per_epoch']),
                    'false_neg': np.zeros(test_params['batches_per_epoch']),
                }

                print("Training model...", flush=True)
                for (batch_i, batch) in islice(enumerate(train_loader.get_batches(batch_size, target_cost_dev)), train_params['batches_per_epoch']):
                    train_stats['loss'][batch_i], train_stats['acc'][batch_i], train_stats['sat'][batch_i], train_stats['pred'][batch_i], train_stats['true_pos'][batch_i], train_stats['false_pos'][batch_i], train_stats['true_neg'][batch_i], train_stats['false_neg'][batch_i] = run_batch(sess, GNN, batch, batch_i, epoch_i, time_steps, train=True, verbose=True, loss_type=loss_type)
                #end
                summarize_epoch(epoch_i,train_stats['loss'],train_stats['acc'],train_stats['sat'],train_stats['pred'],train=True)

                print("Testing model...", flush=True)
                for (batch_i, batch) in islice(enumerate(test_loader.get_batches(batch_size, target_cost_dev)), test_params['batches_per_epoch']):
                    test_stats['loss'][batch_i], test_stats['acc'][batch_i], test_stats['sat'][batch_i], test_stats['pred'][batch_i], test_stats['true_pos'][batch_i], test_stats['false_pos'][batch_i], test_stats['true_neg'][batch_i], test_stats['false_neg'][batch_i] = run_batch(sess, GNN, batch, batch_i, epoch_i, time_steps, train=False, verbose=True, loss_type=loss_type)
                #end
                summarize_epoch(epoch_i,test_stats['loss'],test_stats['acc'],test_stats['sat'],test_stats['pred'],train=False)

                # Save weights
                savepath = './TSP-checkpoints-{loss_type}-{target_cost_dev}/epoch={epoch}'.format(loss_type=loss_type,target_cost_dev=target_cost_dev,epoch=100*np.ceil((epoch_i+1)/100))
                os.makedirs(savepath, exist_ok=True)
                if save_checkpoints: save_weights(sess, savepath);

                logfile.write('{epoch_i} {trloss} {tracc} {trsat} {trpred} {tr_truepos} {tr_falsepos} {tr_trueneg} {tr_falseneg} {tstloss} {tstacc} {tstsat} {tstpred} {tst_truepos} {tst_falsepos} {tst_trueneg} {tst_falseneg}\n'.format(
                    
                    epoch_i = epoch_i,

                    trloss = np.mean(train_stats['loss']),
                    tracc = np.mean(train_stats['acc']),
                    trsat = np.mean(train_stats['sat']),
                    trpred = np.mean(train_stats['pred']),
                    tr_truepos = np.mean(train_stats['true_pos']),
                    tr_falsepos = np.mean(train_stats['false_pos']),
                    tr_trueneg = np.mean(train_stats['true_neg']),
                    tr_falseneg = np.mean(train_stats['false_neg']),

                    tstloss = np.mean(test_stats['loss']),
                    tstacc = np.mean(test_stats['acc']),
                    tstsat = np.mean(test_stats['sat']),
                    tstpred = np.mean(test_stats['pred']),
                    tst_truepos = np.mean(test_stats['true_pos']),
                    tst_falsepos = np.mean(test_stats['false_pos']),
                    tst_trueneg = np.mean(test_stats['true_neg']),
                    tst_falseneg = np.mean(test_stats['false_neg']),
                    )
                )
                logfile.flush()
            #end
        #end
    #end
#end
