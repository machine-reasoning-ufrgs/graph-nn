
import os
import random
import itertools
import numpy as np
from generator import read_graph

class InstanceLoader(object):

	def __init__(self,path):
		self.path = path

		self.filenames = [ path + '/' + x for x in os.listdir(path) ]
		self.reset()
	#end

	def get_instances(self, n_instances):
		for i in range(n_instances):
			yield read_graph(self.filenames[self.index])
			self.index += 1
		#end
	#end

	def create_batch(self,instances):
		n_vertices 	= [ x[0].shape[0] for x in instances ]
		k 			= [ x[1] for x in instances ]
		k_colorable = [ x[2] for x in instances ]
		total_n = sum(n_vertices)

		M_all = np.zeros((total_n,total_n))
		for (i,M) in enumerate([ x[0] for x in instances ]):
			n_acc = sum(n_vertices[0:i])
			M_all[n_acc:n_acc+n_vertices[i], n_acc:n_acc+n_vertices[i]] = M
		#end



		return M_all, n_vertices, k, k_colorable
	#end

	def get_batches(self, batch_size):
		for i in range( len(self.filenames) // batch_size ):
			yield self.create_batch(list(self.get_instances(batch_size)))
		#end
	#end

	def reset(self):
		random.shuffle( self.filenames )
		self.index = 0
	#end
#end

if __name__ == '__main__':

	loader = InstanceLoader("train")
	instances = loader.get_instances(10)
	#print(list(instances))
	loader.create_batch(list(instances))

#end
