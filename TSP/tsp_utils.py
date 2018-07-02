

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
        n_vertices  = np.array([ x[0].shape[0] for x in instances ])
        # n_edges[i]: number of edges in the i-th instance
        n_edges     = np.array([ len(np.nonzero(x[0])[0]) for x in instances ])
        # total_vertices: total number of vertices among all instances
        total_vertices  = sum(n_vertices)
        # total_edges: total number of edges among all instances
        total_edges     = sum(n_edges)

        # Compute grouped matrices Ma_all and Mw_all
        Ma_all, Mw_all = np.zeros((total_vertices,total_vertices)), np.zeros((total_vertices,total_vertices))
        for (i,Ma,Mw) in [ (i,x[0],x[1]) for (i,x) in enumerate(instances) ]:
            n_acc = sum(n_vertices[0:i])
            Ma_all[n_acc:n_acc+n_vertices[i], n_acc:n_acc+n_vertices[i]] = Ma
            Mw_all[n_acc:n_acc+n_vertices[i], n_acc:n_acc+n_vertices[i]] = Mw
        #end

        # edges: a list of node index 2-uples, each corresponding to an edge in Ma_all
        edges = list(zip(np.nonzero(Ma_all)[0], np.nonzero(Ma_all)[1]))
        # edge_index[(i,j)]: the index of the edge (i,j)
        edge_index = { (i,j):e for (e,(i,j)) in enumerate(edges) }

        # route_edges[i]: 1 if the i-th edge belongs to the route of some instance
        route_edges = np.zeros(total_edges, dtype=int)
        for (i,route) in [ (i,x[2]) for (i,x) in enumerate(instances) ]:
            route_relabelled = [ sum(n_vertices[0:i])+x for x in route ]
            for x,y in zip(route_relabelled, route_relabelled[1:] + route_relabelled[0:1]):
                route_edges[edge_index[(min(x,y), max(x,y))]] = 1
            #end
        #end

        return Ma_all, Mw_all, n_vertices, n_edges, route_edges
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

def solve(Ma, Mw):
    """
        Find the optimal TSP tour given vertex adjacencies given by the binary
        matrix Ma and edge weights given by the real-valued matrix W
    """

    n = Ma.shape[0]

    # Create a routing model
    routing = pywrapcp.RoutingModel(n, 1, 0)

    def dist(i,j):
        return Mw[i,j]
    #end

    # Define edge weights
    routing.SetArcCostEvaluatorOfAllVehicles(dist)

    # Remove connections where Ma[i,j] = 0
    for i in range(n):
        for j in range(n):
            if Ma[i,j] == 0:
                routing.NextVar(i).RemoveValue(j)
            #end
        #end
    #end

    assignment = routing.Solve()

    def route_generator():
        index = 0
        for i in range(n):
            yield index
            index = assignment.Value(routing.NextVar(index))
        #end
    #end

    return list(route_generator()) if assignment is not None else []
#end

def create_graph_metric(n, bins):
    
    # Select 'n' 2D points in the unit square
    nodes = np.random.rand(n,2)

    # Build a fully connected adjacency matrix
    Ma = np.ones((n,n))-np.eye(n)
    # Build a weight matrix
    Mw = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            # Multiply by 1/√2 to normalize
            Mw[i,j] = (1.0/np.sqrt(2)) * np.sqrt((nodes[i,0]-nodes[j,0])**2+(nodes[i,1]-nodes[j,1])**2)
        #end
    #end

    # Rescale and round weights, quantizing them into 'bins' integer bins
    Mw = np.round(bins * Mw)

    # Solve
    solution = solve(Ma,Mw)

    # Rescale weights such that they are all ∈ [0,1]
    Mw = Mw / bins

    return np.triu(Ma), Mw, ([] if solution is None else solution), nodes
#end

def write_graph(Ma, Mw, route, filepath):
    with open(filepath,"w") as out:
        # Write header 'p |V| |E| \n route'
        out.write("p {} {}\n{}\n".format(Ma.shape[0], len(np.nonzero(Ma)[0]), ' '.join([str(x) for x in route])))

        # Write edges
        for (i,j) in zip(list(np.nonzero(Ma))[0], list(np.nonzero(Ma))[1]):
            out.write("{} {} {}\n".format(i,j,Mw[i,j]))
        #end
    #end
#end

def read_graph(filepath):
    with open(filepath,"r") as f:
        n, m = [ int(x) for x in f.readline().split()[1:]]
        route = [ int(x) for x in f.readline().split() ]
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

def create_dataset_metric(nmin, nmax, path, bins=10, samples=1000):

    if not os.path.exists(path):
        os.makedirs(path)
    #end if

    for i in range(samples):
        n = np.random.randint(nmin,nmax)
        Ma,Mw,solution,_ = create_graph_metric(n,bins)
        write_graph(Ma,Mw,solution,"{}/{}.graph".format(path,i))
        if (i-1) % (samples//10) == 0:
            print('{}% Complete'.format(np.round(100*i/samples)))
        #end
    #end
#end

def to_quiver(Ma, Mw):

    total_vertices  = Ma.shape[0]
    total_edges     = len(np.nonzero(Ma)[0])

    # Define matrices M and W
    M = np.zeros((total_edges,total_vertices))
    W = np.zeros((total_edges,1))
    R = np.zeros((total_edges,1))
    for (e,(i,j)) in enumerate(zip(list(np.nonzero(Ma)[0]), list(np.nonzero(Ma)[1]))):
        M[e,i] = 1
        M[e,j] = 1
        W[e,0] = Mw[i,j]
        R[e,0] = e / total_edges
    #end
    return M,W,R
#end

def get_edges_mask(Ma,route):
    """
        Compute a binary mask marking which edges are in the route, given an
        adjacency matrix and list of edge indices
    """
    edges = list(zip(np.nonzero(Ma)[0], np.nonzero(Ma)[1]))
    edge_index = { (i,j):e for (e,(i,j)) in enumerate(edges) }

    route_edges = np.zeros(len(edges))
    for (i,j) in zip(route,route[1:]+route[:1]):
        route_edges[edge_index[(min(i,j),max(i,j))]] = 1
    #end

    return route_edges
#end