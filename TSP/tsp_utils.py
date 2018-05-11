
import sys, os
import numpy as np
import random
import tensorflow as tf

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
		n_vertices 	= np.array([ x[0].shape[0] for x in instances ])
		n_edges		= np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
		solution	= np.array([ x[2] for x in instances ])
		total_n 	= sum(n_vertices)

		Ma_all, Mw_all = np.zeros((total_n,total_n)), np.zeros((total_n,total_n))
		for (i,Ma,Mw) in [ (i,x[0],x[1]) for (i,x) in enumerate(instances) ]:
			n_acc = sum(n_vertices[0:i])
			Ma_all[n_acc:n_acc+n_vertices[i], n_acc:n_acc+n_vertices[i]] = Ma
			Mw_all[n_acc:n_acc+n_vertices[i], n_acc:n_acc+n_vertices[i]] = Mw
		#end

		return Ma_all, Mw_all, n_vertices, n_edges, solution
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

def write_graph(Ma, W, solution, filepath):
	with open(filepath,"w") as out:
		# Write header 'p |V| |E| solution'
		out.write("p {} {} {}\n".format(Ma.shape[0], len(np.nonzero(Ma)[0]), solution))

		# Write edges
		for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
			out.write("{} {} {}\n".format(i,j,W[i,j]))
		#end
	#end
#end

def read_graph(filepath):
	with open(filepath,"r") as f:
		n, m, solution = [ int(x) for x in f.readline().split()[1:]]
		Ma = np.zeros((n,n),dtype=int)
		Mw = np.zeros((n,n),dtype=int)
		for edge in range(m):
			i,j,w = [ int(x) for x in f.readline().split() ]
			Ma[i,j] = 1
			Mw[i,j] = w
		#end
	#end
	return Ma,Mw,solution
#end