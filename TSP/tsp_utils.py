

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
			Ma,Mw,route = read_graph(self.filenames[self.index])
			yield Ma,Mw,route
			self.index += 1
		#end
	#end

	def create_batch(self,instances):

		# n_instances: number of instances
		n_instances = len(instances)
		
		# n_vertices[i]: number of vertices in the i-th instance
		n_vertices 	= np.array([ x[0].shape[0] for x in instances ])
		# n_edges[i]: number of edges in the i-th instance
		n_edges		= np.array([ len(np.nonzero(x[0])[0]) for x in instances ])

		# total_vertices: total number of vertices among all instances
		total_vertices 	= sum(n_vertices)
		# total_edges: total number of edges among all instances
		total_edges = sum(n_edges)
		
		# route_edges[i]: 1 if the i-th edge belongs to the route of some instance
		route_edges	= np.zeros(total_edges, dtype=int)

		# cost[i]: the cost of the solution to the i-th instance
		cost = np.zeros(n_instances)

		# Compute grouped matrices Ma_all and Mw_all
		Ma_all, Mw_all = np.zeros((total_vertices,total_vertices)), np.zeros((total_vertices,total_vertices))
		for (i,Ma,Mw) in [ (i,x[0],x[1]) for (i,x) in enumerate(instances) ]:
			n_acc = sum(n_vertices[0:i])
			Ma_all[n_acc:n_acc+n_vertices[i], n_acc:n_acc+n_vertices[i]] = Ma
			Mw_all[n_acc:n_acc+n_vertices[i], n_acc:n_acc+n_vertices[i]] = Mw
		#end

		# Assign an index to each edge (among all instances)
		# edge_index[i,j] = index of the edge (i,j)
		edge_index = np.zeros((total_vertices,total_vertices), dtype=int)
		edge_count = 0
		for i in range(total_vertices):
			for j in range(total_vertices):
				if Ma_all[i,j] == 1:
					edge_index[i,j] = edge_count
					edge_count += 1
				#end
			#end
		#end

		for (i,route) in [ (i,x[2]) for (i,x) in enumerate(instances) ]:
			n_acc = sum(n_vertices[0:i])
			route_relabelled = [ n_acc+x for x in route ]
			for x,y in zip(route_relabelled, route_relabelled[1:] + route_relabelled[0:1]):
				route_edges[edge_index[x,y]] = 1
				cost[i] += Mw_all[x,y]
			#end
		#end

		return Ma_all, Mw_all, n_vertices, n_edges, route_edges, cost
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

def write_graph(Ma, W, route, filepath):
	with open(filepath,"w") as out:
		# Write header 'p |V| |E| \n route'
		out.write("p {} {}\n{}\n".format(Ma.shape[0], len(np.nonzero(Ma)[0]), ' '.join([str(x) for x in route])))

		# Write edges
		for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
			out.write("{} {} {}\n".format(i,j,W[i,j]))
		#end
	#end
#end

def read_graph(filepath):
	with open(filepath,"r") as f:
		n, m = [ int(x) for x in f.readline().split()[1:]]
		route = [ int(x) for x in f.readline().split() ]
		Ma = np.zeros((n,n),dtype=int)
		Mw = np.zeros((n,n),dtype=int)
		for edge in range(m):
			i,j,w = [ int(x) for x in f.readline().split() ]
			Ma[i,j] = 1
			Mw[i,j] = w
		#end
	#end
	return Ma,Mw,route
#end