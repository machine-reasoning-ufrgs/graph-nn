import sys, os, time
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
sys.path.insert(1, os.path.join(sys.path[0], '..'))
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from model import build_neurosat
import instance_loader
import itertools
from logutil import test_with
from util import timestamp, memory_usage
from cnf import ensure_datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
from numpy import linalg as LA

if __name__ == "__main__":
  print( "{timestamp}\t{memory}\tMaking sure ther datasets exits ...".format( timestamp = timestamp(), memory = memory_usage() ) )
  ensure_datasets( make_critical = True )
  if not os.path.isdir( "tmp" ):
    sys.exit(1)
  #end if
  d = 128
  
  # Build model
  solver = build_neurosat( d )
  
  # Create model saver
  saver = tf.train.Saver()

  with tf.Session() as sess:

    # Initialize global variables
    print( "{timestamp}\t{memory}\tInitializing global variables ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
    sess.run( tf.global_variables_initializer() )
    
    # Restore saved weights
    print( "{timestamp}\t{memory}\tRestoring saved model ... ".format( timestamp = timestamp(), memory = memory_usage() ) )
    saver.restore(sess, "./tmp backup/neurosat.ckpt")

    # Define loader and get a batch with size 1 (one instance)
    loader = instance_loader.InstanceLoader('test-instances/sat/')
    batch = list(itertools.islice(loader.get_batches(1),1))[0]

    time_steps = 100

    votes = np.zeros((time_steps,2*batch.n[0]))

    for t in range(0,time_steps):

        votes[t,:] = sess.run(
            solver["votes"],
            feed_dict = {
                solver["gnn"].time_steps: t,
                solver["gnn"].matrix_placeholders["M"]: batch.get_dense_matrix(),
                solver["instance_SAT"]: np.array( [1] ),
                solver["num_vars_on_instance"]: batch.n
                }
            )[:,0]
    #end

    plt.imshow(votes)
    plt.show()

#    fig, ax = plt.subplots()
#    fig.set_tight_layout(True)
#
#    ax.set_xlim(-1, 1), ax.set_xticks([])
#    ax.set_ylim(-1, 1), ax.set_yticks([])
#
#    # Get cluster centers
#    states = sess.run(
#            solver["gnn"].last_states["L"].h,
#            feed_dict = {
#                solver["gnn"].time_steps: 26,
#                solver["gnn"].matrix_placeholders["M"]: batch.get_dense_matrix(),
#                solver["instance_SAT"]: np.array( [1] ),
#                solver["num_vars_on_instance"]: batch.n
#                }
#        )
#    kmeans = KMeans(n_clusters=2, random_state=0).fit(states)
#    c1, c2 = kmeans.cluster_centers_[0,:], kmeans.cluster_centers_[1,:]
#
#    red_components  = [ i for i,state in enumerate(states) if LA.norm(state-c1) <= LA.norm(state-c2) ]
#    blue_components = [ i for i,state in enumerate(states) if LA.norm(state-c1) >  LA.norm(state-c2) ]
#
#    scatter_red     = ax.scatter(np.zeros(len(red_components)), np.zeros(len(red_components)), c='r')
#    scatter_blue    = ax.scatter(np.zeros(len(blue_components)), np.zeros(len(blue_components)), c='b')
#
#    def update(i):
#
#        time_steps = i
#
#        states = sess.run(
#            solver["gnn"].last_states["L"].h,
#            feed_dict = {
#                solver["gnn"].time_steps: time_steps,
#                solver["gnn"].matrix_placeholders["M"]: batch.get_dense_matrix(),
#                solver["instance_SAT"]: np.array( [1] ),
#                solver["num_vars_on_instance"]: batch.n
#                }
#        )
#
#        principal_components = PCA(n_components=2).fit_transform(states)
#        principal_components /= np.linalg.norm(principal_components)
#
#        scatter_red.set_offsets([ x for i,x in enumerate(principal_components) if i in red_components ])
#        scatter_blue.set_offsets([ x for i,x in enumerate(principal_components) if i in blue_components ])
#
#        label = 'timestep {0}'.format(i)
#        ax.set_xlabel(label)
#    #end
#
#    anim = FuncAnimation(fig, update, frames=np.arange(0, 100), interval=200)
#    anim.save('line.gif', dpi=80, writer='imagemagick')

#end