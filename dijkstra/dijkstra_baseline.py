import sys, os, time
import numpy as np
import networkx as nx
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import tools
#import itertools
from util import timestamp, memory_usage
from dijkstra_util import baseline_error

if __name__ == '__main__':
	d = 64
	epochs = 100
	batch_n_max = 4096
	batches_per_epoch = 32
	n_size_min = 16
	n_loss_increase_threshold = 0.01
	n_size_max = 512
	edge_probability = 0.25

	# Run for a number of epochs
	print( "{timestamp}\t{memory}\tRunning for {} epochs".format( epochs, timestamp = timestamp(), memory = memory_usage() ) )
	n_size = n_size_min
	for epoch in range( epochs ):
		# Run batches
		#instance_generator.reset()
		epoch_loss = 0.0
		epoch_err = 0
		epoch_abserr = 0
		epoch_n = 0
		epoch_m = 0
		for batch_i in range( batches_per_epoch ):
			# Create random graphs
			batch_n_size = np.random.randint( n_size_min, n_size+1 )
			n_acc = 0
			max_n = 0
			instances = 0
			Gs = []
			distances = []
			err, abserr = 0, 0
			while True:
				g_n = np.random.randint( batch_n_size//2, batch_n_size*2 )
				if n_acc + g_n * 2 < batch_n_max:
					n_acc += g_n * 2
					instances += 2
					max_n = max( max_n, g_n )
					g1,g2 = baseline_error( g_n, edge_probability )
					err += g1["%error"] + g1["%error"]
					abserr += g1["%abserror"] + g2["%abserror"]
				else:
					break
				#end if
			#end for
			n = n_acc
			m = edge_probability * n*n
			
			epoch_err += err
			epoch_abserr += err
			epoch_n += n_acc
			epoch_m += m
			
			print(
				"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m,instances): ({n},{m},{i})\t| (Loss,%Error|%Error|): ({loss:.5f},{error:.5f}|{abserror:.5f}|)".format(
					timestamp = timestamp(),
					memory = memory_usage(),
					epoch = epoch,
					batch = batch_i,
					loss = 0,
					error = err,
					abserror = abserr,
					n = n,
					m = m,
					i = instances
				),
				flush = True
			)
		#end for
		# Summarize Epoch
		epoch_err /= batches_per_epoch
		epoch_abserr /= batches_per_epoch
		print(
			"{timestamp}\t{memory}\tEpoch {epoch}\tBatch {batch} (n,m): ({n},{m})\t| Mean (Loss,%Error|%Error|): ({loss:.5f},{error:.5f}|{abserror:.5f}|)".format(
				timestamp = timestamp(),
				memory = memory_usage(),
				epoch = epoch,
				batch = "all",
				loss = 0,
				error = epoch_err,
				abserror = epoch_abserr,
				n = epoch_n,
				m = epoch_m,
			),
			flush = True
		)
	#end for
