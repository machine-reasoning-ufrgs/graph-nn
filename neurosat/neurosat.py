import sys, os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tensorflow as tf
# Import model builder
from model import build_neurosat
# Import tools
import instance_loader
import itertools
from util import timestamp, memory_usage
from logutil import run_and_log_batch

if __name__ == '__main__':
  epochs = 30
  d = 2#128
  
  timesteps = 26
  batch_size = 64
  batches_per_epoch = 128

  # Build model
  print( "{timestamp}\t{memory}\tBuilding model ...".format  ( timestamp = timestamp(), memory = memory_usage() ) )
  neurosat = build_neurosat(d)

  # Create batch loader
  print( "{timestamp}\t{memory}\tLoading instances ...".format( timestamp = timestamp(), memory = memory_usage() ) )
  instance_generator = instance_loader.InstanceLoader( "./instances" )
  test_instance_generator = instance_loader.InstanceLoader( "./test_instances" )

  with tf.Session() as sess:
    
    # Initialize global variables
    print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
    sess.run( tf.global_variables_initializer() )

    # Run for a number of epochs
    print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
    for epoch in range( epochs ):
      # Run batches
      instance_generator.reset()
      epoch_loss = 0.0
      epoch_accuracy = 0.0
      epoch_n = 0
      epoch_m = 0
      for b, batch in itertools.islice( enumerate( instance_generator.get_batches( batch_size ) ), batches_per_epoch ):

        sats = np.array(batch.sat).astype(int)
        n_vars = np.array(batch.n)
        M = batch.get_sparse_matrix()

        n, m =  M[2][0]//2, M[2][1]

        _, loss, accuracy = sess.run( [neurosat["train_step"], neurosat["loss"], neurosat["accuracy"]], feed_dict={
          neurosat["gnn"].matrix_placeholders["M"]: M,
          neurosat["gnn"].time_steps: timesteps,
          neurosat["instance_SAT"]: sats,
          neurosat["num_vars_on_instance"]: n_vars
          } )
          
        epoch_loss += loss
        epoch_accuracy += accuracy
        epoch_n += n
        epoch_m += m
        
        print(
          "{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Solver (Loss, Acc): ({loss:.5f}, {accuracy:.5f})".format(
            timestamp = timestamp(),
            memory = memory_usage(),
            epoch = epoch,
            batch = b,
            loss = loss,
            accuracy = accuracy,
            n = n,
            m = m,
          ),
          flush = True
        )
      #end for
      # Summarize Epoch
      epoch_loss = epoch_loss / batches_per_epoch
      epoch_accuracy = epoch_accuracy / batches_per_epoch
      print(
        "{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m}) | Solver (Mean Loss, Mean Acc): ({loss:.5f}, {accuracy:.5f})".format(
          timestamp = timestamp(),
          memory = memory_usage(),
          epoch = epoch,
          batch = "all",
          loss = epoch_loss,
          accuracy = epoch_accuracy,
          n = epoch_n,
          m = epoch_m,
        ),
        flush = True
      )
    #end for
  #end Session
