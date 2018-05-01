
import os
import numpy as np
import random
from constraint import *

def create_graph_pair(n,k):
	"""
		Starts with a disjoint set of n vertices (zero edges). Iteratively
		adds edges selected uniformly at random while the resulting graph
		admits a k-coloring. The penultimate and last of such graphs, which
		differ only by a single edge, are respectively k-colorable and not
		k-colorable.
	"""
	k_colorable = True
	M1 = np.zeros((n,n),dtype=int)
	M2 = np.zeros((n,n),dtype=int)
	while k_colorable:
		
		# Choose (and add) an edge uniformly at random
		a,b = random.choice([ (i,j) for i in range(n) for j in range(n) if i != j and M1[i,j]==0])
		M2[a,b] = 1

		# Create a CSP problem with n integer variables each of which can assume k different values ("colors")
		problem = Problem()
		problem.addVariables(range(n), range(k))
		for i in range(n):
			for j in range(n):
				if M2[i,j] == 1:
					# Add one constraint for each edge
					problem.addConstraint(lambda a,b: a != b, (i,j))
				#end
			#end
		#end

		# Check if k-colorable
		k_colorable = problem.getSolution() is not None

		if k_colorable:
			M1[a,b] = 1
		#end
	#end

	return M1, M2
#end

def write_graph(M, k, k_colorable, path):
	with open(path,"w") as out:
		out.write("p {} {} {} {}\n".format(M.shape[0], len(np.nonzero(M)[0]), k, k_colorable ))

		for (i,j) in zip(list(np.nonzero(M)[0]), list(np.nonzero(M))[1]):
			out.write("{} {}\n".format(i,j))
		#end
	#end
#end

def read_graph(path):
	with open(path,"r") as f:
		n, m, k, k_colorable = [ int(x) for x in f.readline().split()[1:]]
		M = np.zeros((n,n),dtype=int)
		for edge in range(m):
			i,j = [ int(x) for x in f.readline().split() ]
			M[i,j] = 1
		#end
	#end
	return M,k,k_colorable
#end

def create_dataset(n, samples = 1000, path="instances"):

	if not os.path.exists(path):
		os.makedirs(path)
	#end if

	for i in range(samples//2):
		# Select a random k ~ Bernouilli(0.3) + Geo(0.4)
		k = random.choice(range(2,5+1))
		print("Creating instances pair with k={}".format(k))
		M1, M2 = create_graph_pair(n,k)
		write_graph(M1,k,0,"{}/{}.graph".format(path,2*i))
		write_graph(M2,k,1,"{}/{}.graph".format(path,2*i+1))
	#end for
#end

def generate(n, batch_size):

	k_colorable, k_uncolorable = [], []
	for i in range(batch_size//2):
		M1,M2 = create_graph_pair(n,k)
		k_colorable.append(M1)
		k_uncolorable.append(M2)
	#end

	features 	= np.zeros((batch_size,n,n))
	labels 		= np.zeros((batch_size,))

	for i in range(batch_size//2):
		features[2*i,  :,:] = k_colorable[i]
		features[2*i+1,:,:] = k_uncolorable[i]
		labels[2*i] = 1
		labels[2*i+1] = 0
	#end

	yield features, labels
#end

if __name__ == '__main__':
	create_dataset(20, samples = 32*128, path = "train")
	create_dataset(20, samples = 32*128, path = "test")
#end