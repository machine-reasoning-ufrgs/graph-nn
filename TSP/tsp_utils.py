
import sys, os
import numpy as np
import random
import tensorflow as tf
import untangle # For processing XML
import itertools
import math
from ortools.constraint_solver import pywrapcp          # For solving TSP instances
from ortools.constraint_solver import routing_enums_pb2 # For solving TSP instances
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from concorde.tsp import TSPSolver
from redirector import Redirector
from functools import reduce

STDOUT = 1
STDERR = 2

class InstanceLoader(object):

    def __init__(self,path,target_cost_dev):
        self.path = path
        self.filenames = [ path + '/' + x for x in os.listdir(path) ]
        self.target_cost_dev = target_cost_dev
        self.reset()
    #end

    def get_instances(self, n_instances):
        for i in range(n_instances):
            Ma,Mw,route = read_graph(self.filenames[self.index])
            yield Ma,Mw,route
            self.index += 1
        #end
    #end

    def create_batch(instances, target_cost_dev=None, target_cost=None):

        # n_instances: number of instances
        n_instances = len(instances)
        
        # n_vertices[i]: number of vertices in the i-th instance
        n_vertices  = np.array([ x[0].shape[0] for x in instances ])
        # n_edges[i]: number of edges in the i-th instance
        n_edges     = np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
        # total_vertices: total number of vertices among all instances
        total_vertices  = sum(n_vertices)
        # total_edges: total number of edges among all instances
        total_edges     = sum(n_edges)

        # Compute matrices M, W, CV, CE
        # and vectors edges_mask and route_exists
        EV              = np.zeros((total_edges,total_vertices))
        W               = np.zeros((total_edges,1))
        VR              = np.zeros((total_vertices,n_instances))
        ER              = np.zeros((total_edges,n_instances))
        C               = np.zeros((n_instances,1))
        edges_mask      = np.zeros(total_edges)
        route_exists    = np.zeros(n_instances)
        for (i,(Ma,Mw,route)) in enumerate(instances):
            
            # Get the number of vertices (n) and edges (m) in this graph
            n, m = n_vertices[i], n_edges[i]
            # Get the number of vertices (n_acc) and edges (m_acc) up until the i-th graph
            n_acc = sum(n_vertices[0:i])
            m_acc = sum(n_edges[0:i])

            # Get the list of edges in this graph
            edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))

            # Get the list of edges in the optimal TSP route for this graph
            route_edges = [ (min(x,y),max(x,y)) for (x,y) in zip(route,route[1:]+route[0:1]) ]

            # Compute the optimal (normalized) TSP cost for this graph
            cost = sum([ Mw[x,y] for (x,y) in route_edges ]) / n

            # Choose a target cost and fill CV and CE with it
            if target_cost is None:
                #delta = np.random.uniform(0,target_cost_dev*cost)
                delta = target_cost_dev*cost
                #CV[n_acc:n_acc+n,0] = cost + delta if i%2==0 else cost - delta
                #CE[m_acc:m_acc+m,0] = cost + delta if i%2==0 else cost - delta
                C[i,0] = cost + delta if i%2==0 else cost - delta
                route_exists[i] = 1 if i%2==0 else 0
            else:
                #CV[n_acc:n_acc+n,0] = target_cost
                #CE[m_acc:m_acc+m,0] = target_cost
                C[i,0] = target_cost
                route_exists[i] = 1 if target_cost >= cost else 0
            #end

            # Populate EV, W and edges_mask
            for e,(x,y) in enumerate(edges):
                EV[m_acc+e,n_acc+x] = 1
                EV[m_acc+e,n_acc+y] = 1
                W[m_acc+e] = Mw[x,y]
                if (x,y) in route_edges:
                    edges_mask[m_acc+e] = 1
                #end
            #end

            # Populate VR and ER
            VR[n_acc:n_acc+n, i] = 1
            ER[m_acc:m_acc+m, i] = 1

        #end

        return EV, VR, ER, W, C, edges_mask, route_exists, n_vertices, n_edges
    #end

    def get_batches(self, batch_size):
        for i in range( len(self.filenames) // batch_size ):
            instances = list(self.get_instances(batch_size))
            instances = reduce(lambda x,y: x+y, zip(instances,instances))
            yield InstanceLoader.create_batch(instances, self.target_cost_dev)
        #end
    #end

    def reset(self):
        random.shuffle( self.filenames )
        self.index = 0
    #end
#end

def solve(Ma, Mw):

    write_graph(Ma,Mw,[],'tmp',int_weights=True)
    redirector_stdout = Redirector(fd=STDOUT)
    redirector_stderr = Redirector(fd=STDERR)
    redirector_stderr.start()
    redirector_stdout.start()
    solver = TSPSolver.from_tspfile('tmp')
    solution = solver.solve(verbose=False)
    redirector_stderr.stop()
    redirector_stdout.stop()

    return solution.tour if solution.found_tour else []
#end

def create_graph_metric(n, bins, connectivity=1):
    
    # Select 'n' 2D points in the unit square
    nodes = np.random.rand(n,2)

    # Build an adjacency matrix with given connectivity
    Ma = (np.random.rand(n,n) < connectivity).astype(int)
    for i in range(n):
        Ma[i,i] = 0
        for j in range(i+1,n):
            Ma[i,j] = Ma[j,i]
        #end
    #end
    
    # Build a weight matrix
    Mw = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # Multiply by 1/√2 to normalize
            Mw[i,j] = (1.0/np.sqrt(2)) * np.sqrt((nodes[i,0]-nodes[j,0])**2+(nodes[i,1]-nodes[j,1])**2)
        #end
    #end

    # Add huge costs to inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = n+1

    # Rescale and round weights, quantizing them into 'bins' integer bins
    Mw = np.round(bins * Mw)

    # Solve
    route = solve(Ma,Mw)
    if route == []: print('Unsolvable');

    # Check if route contains edges which are not in the graph and add them
    for (i,j) in [ (i,j) for (i,j) in zip(route,route[1:]+route[0:1]) if Ma[i,j] == 0 ]:
        Ma[i,j] = Ma[j,i] = 1
        Mw[i,j] = Mw[j,i] = 1
    #end

    # Remove huge costs from inexistent edges to simulate a disconnected instance
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                Mw[i,j] = 0

    # Rescale weights such that they are all ∈ [0,1]
    Mw = Mw / bins

    return np.triu(Ma), Mw, ([] if route is None else route), nodes
#end

def write_graph(Ma, Mw, route, filepath, int_weights=False):
    with open(filepath,"w") as out:

        n, m = Ma.shape[0], len(np.nonzero(Ma)[0])
        
        out.write('TYPE : TSP\n')

        out.write('DIMENSION: {n}\n'.format(n = n))

        out.write('EDGE_DATA_FORMAT: EDGE_LIST\n')
        out.write('EDGE_WEIGHT_TYPE: EXPLICIT\n')
        out.write('EDGE_WEIGHT_FORMAT: FULL_MATRIX \n')
        
        # List edges in the (generally not complete) graph
        out.write('EDGE_DATA_SECTION\n')
        for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
            out.write("{} {}\n".format(i,j))
        #end
        out.write('-1\n')

        # Write edge weights as a complete matrix
        out.write('EDGE_WEIGHT_SECTION\n')
        for i in range(n):
            if int_weights:
                out.write('\t'.join([ str(int(Mw[i,j])) for j in range(n)]))
            else:
                out.write('\t'.join([ str(float(Mw[i,j])) for j in range(n)]))
            #end
            out.write('\n')
        #end

        # Write route as a concorde commentary
        out.write('TOUR_SECTION\n')
        out.write('{}\n'.format(' '.join([str(x) for x in route])))

        out.write('EOF\n')
    #end
#end

def read_graph_old(filepath):
    with open(filepath,"r") as f:
        n, m = [ int(x) for x in f.readline().split()]
        route = [ int(x) for x in f.readline()[1:].split() ]
        Ma = np.zeros((n,n),dtype=int)
        Mw = np.zeros((n,n),dtype=float)
        for edge in range(m):
            i,j,w = [ float(x) for x in f.readline().split() ]
            i,j = int(i), int(j)
            Ma[i,j] = 1
            Mw[i,j] = w
        #end
    #end
    return Ma,Mw,route
#end

def read_graph(filepath):
    with open(filepath,"r") as f:

        line = ''

        while 'DIMENSION' not in line: line = f.readline();

        n = int(line.split()[1])
        Ma = np.zeros((n,n),dtype=int)
        Mw = np.zeros((n,n),dtype=float)

        while 'EDGE_DATA_SECTION' not in line: line = f.readline();
        line = f.readline()
        
        while '-1' not in line:
            i,j = [ int(x) for x in line.split() ]
            Ma[i,j] = 1
            line = f.readline()
        #end
        line = f.readline()

        for i in range(n):
            Mw[i,:] = [ float(x) for x in f.readline().split() ]
        #end
        line = f.readline()

        route = [ int(x) for x in f.readline().split() ]

    #end
    return Ma,Mw,route
#end

def create_dataset_metric(nmin, nmax, conn_min, conn_max, path, bins=10**6, connectivity=1, samples=1000):

    if not os.path.exists(path):
        os.makedirs(path)
    #end if

    route_cost = np.zeros(samples)

    solvable = 0
    for i in range(samples):

        route = []
        while route == []:
            n = np.random.randint(nmin,nmax)
            connectivity = np.random.uniform(conn_min,conn_max)
            Ma,Mw,route,nodes = create_graph_metric(n,bins,connectivity)
            # Compute route cost
            route_cost[i] = sum([ Mw[min(i,j),max(i,j)] for (i,j) in zip(route,route[1:]+route[:1]) ]) / n
        #end

        write_graph(Ma,Mw,route,"{}/{}.graph".format(path,i))
        if (i-1) % (samples//10) == 0:
            print('{}% Complete'.format(np.round(100*i/samples)), flush=True)
        #end
    #end

    # Return mean and standard deviation for the set of (normalized) route costs
    return np.mean(route_cost), np.std(route_cost)
#end