

import sys, os
import numpy as np
import random
import tensorflow as tf
import untangle # For processing XML
import itertools
import math
from ortools.constraint_solver import pywrapcp			# For solving TSP instances
from ortools.constraint_solver import routing_enums_pb2	# For solving TSP instances
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
			#if len(route) > 0:
			yield Ma,Mw,route
			#end
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

def create_graph_random(n,connectivity,max_dist):	
	
	Ma = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			if np.random.rand() < connectivity:
				Ma[i,j] = 1
				Ma[j,i] = 1
			#end
		#end
	#end
	
	Mw = (np.random.randint(1,max_dist,(n,n)))

	solution = solve(Ma,Mw)

	return Ma, Mw, ([] if solution is None else solution)
#end

def create_graph_metric(n, bins):
	# Select 'n' 2D points in the unit square
	nodes = np.random.rand(n,2)

	# Build a fully-connected adjacency matrix
	Ma = np.ones((n,n))
	# Build a weight matrix
	Mw = np.zeros((n,n))
	for i in range(n):
		for j in range(n):
			Mw[i,j] = np.sqrt((nodes[i,0]-nodes[j,0])**2+(nodes[i,1]-nodes[j,1])**2)
		#end
	#end

	# Rescale and round weights, quantizing them into 'bins' integer bins
	Mw = (bins * Mw).astype(int)

	# Solve
	solution = solve(Ma,Mw)

	return Ma, Mw, ([] if solution is None else solution)
#end

def XML_to_adjacency_dict(map_path):
	# Read OSM file
	obj 	= untangle.parse(map_path)
	
	# Get lists of nodes and ways
	nodes 	= obj.osm.node
	ways 	= obj.osm.way
	ways	= [ way for way in ways if [ child for child in way.children if child._name == 'tag' and child['k'] == 'highway'] ]
	#tag_keys = list(set(itertools.chain.from_iterable([ [child['k'] for child in way.children if child._name == 'tag'] for way in ways ])))

	"""
		Create a dictionary where nodes' IDs are the keys and the
		corresponding values dictionaries with the following fields:
		1. latitutde
		2. longitude
		3. the node's adjacency list (initially empty))
	"""
	adjacency_dict = {}
	for node in nodes:
		adjacency_dict[node['id']] = {'lat': float(node['lat']), 'lon': float(node['lon']), 'adjacencies': []}
	#end

	"""
		Each 'way' is composed of a sequence of references to adjacent nodes,
		forming a directed path. Now we populate the adjacency list of each
		node by adding every subsequence of two nodes in a way as an edge to
		our graph. We will also add a directed edge in the opposite direction
		(i.e. (i,j) and (j,i))
	"""
	for way in ways:
		# Get the sequence of node_ids corresponding to this way
		node_ids = [x['ref'] for x in way.nd]

		# zip 'node_ids' with itself to produce a list of directed edges
		edges = zip(node_ids[:-1], node_ids[1:])

		for (i,j) in edges:
			adjacency_dict[i]['adjacencies'].append(j)
			adjacency_dict[j]['adjacencies'].append(i)
		#end
	#end

	return adjacency_dict
#end

def preprocess_adjacency_dict(adjacency_dict):
	"""
		Util to pre-process an adjacency dict in order to replace sequences of
		edges linked by a single path (a single 'way' in OSM parlance) by a
		single edge connecting the two endpoints
	"""

	# Keep record of the list of removed node IDs

	iterations = 0
	while any([len(adjacency_dict[node_id]['adjacencies']) == 2 for node_id in adjacency_dict.keys()]):
		
		iterations += 1
		
		# While there are nodes with degree 2
		for node_id in [node_id for node_id in adjacency_dict.keys() if len(adjacency_dict[node_id]['adjacencies']) == 2]:
			"""
				If node has degree 2, 'collapse' it with the least latitude,
				least longitude (in lexicographical order) neighbor in its
				adjacency list
			"""
			neighbor_ids = adjacency_dict[node_id]['adjacencies']
			neighbor_id = neighbor_ids[0] if adjacency_dict[neighbor_ids[0]]['lat'] < adjacency_dict[neighbor_ids[1]]['lat'] or adjacency_dict[neighbor_ids[0]]['lat'] == adjacency_dict[neighbor_ids[1]]['lat'] and (adjacency_dict[neighbor_ids[0]]['lon'] < adjacency_dict[neighbor_ids[1]]['lon']) else neighbor_ids[1]
			
			collapsed_id = 'collapsed_{}'.format(iterations)
			#print("Collapsing node IDs {} and {} into {}...".format(node_id, neighbor_id, collapsed_id))

			# Add collapsed node to adjacency dict
			adjacency_dict[collapsed_id] = {
			'lat': adjacency_dict[neighbor_id]['lat'],
			'lon': adjacency_dict[neighbor_id]['lon'],
			'adjacencies': adjacency_dict[neighbor_id]['adjacencies'] + [ x for x in neighbor_ids if x != neighbor_id ]
			}
	
			# Filter removed nodes from adjacency dict
			adjacency_dict = { k:v for (k,v) in adjacency_dict.items() if k!=node_id and k!=neighbor_id }

			# For every other node
			for node_id2 in adjacency_dict.keys():
				"""
					Replace 'node_id' and 'neighbor_id' by 'collapsed_id'
					everywhere in the adjacencies of 'node_id2'
				"""
				for (i,neighbor_id2) in enumerate(adjacency_dict[node_id2]['adjacencies']):
					if neighbor_id2 == node_id or neighbor_id2 == neighbor_id:
						adjacency_dict[node_id2]['adjacencies'][i] = collapsed_id
					#end
				#end
			#end
	
			# Now that we've updated the adjacency_dict, break the inner for loop
			break
		#end
	#end

	return adjacency_dict
#end

def create_subgraph_from_adjacency_dict(n, adjacency_dict):
	"""
		Create a graph by selecting a (connected) subset of n nodes from the
		graph given as an adjacency dictionary
	"""

	# Keep record of the list of added nodes' ids
	node_ids = []

	"""
		Select an initial node at random.

		It is possible that our selection is unfortunate and leads to the
		random walk algorithm being unable to fetch n nodes. To mitigate this
		probability, let's choose a node with a degree of at least 2.
	"""
	#current_node = np.random.choice(list(adjacency_dict.keys()))
	current_node = np.random.choice([ node_id for node_id in adjacency_dict.keys() if len(adjacency_dict[node_id]['adjacencies']) > 2])
	node_ids.append(current_node)

	# At each step, add a (previously unvisited) node from the combined list
	# of adjacencies of all nodes in our list
	while len(node_ids) < n:
		# Get the list of all nodes adjacent to any node in our current list. Once again, choose only from nodes with degree at least 2
		all_adjacent 		= itertools.chain.from_iterable([ adjacency_dict[node_id]['adjacencies'] for node_id in node_ids if len(adjacency_dict[node_id]['adjacencies']) > 2 ])
		# Filter out visited nodes
		unvisited_adjacent 	= [ node_id for node_id in all_adjacent if node_id not in node_ids]

		if unvisited_adjacent == []:
			break
		#end

		node_ids.append(np.random.choice(unvisited_adjacent))
	#end

	# Return the adjacency dictionary for the corresponding subgraph
	subgraph_adjacency_dict = {}
	for node_id in node_ids:
		subgraph_adjacency_dict[node_id] = {
		'lat': adjacency_dict[node_id]['lat'],
		'lon': adjacency_dict[node_id]['lon'],
		'adjacencies': [ x for x in adjacency_dict[node_id]['adjacencies'] if x in node_ids] }
	#end

	return subgraph_adjacency_dict
#end

def GPS_distance(lat1, lon1, lat2, lon2):
	phi1 = math.radians(lat1)
	phi2 = math.radians(lat2)
	lambda1 = math.radians(lon1)
	lambda2 = math.radians(lon2)

	delta_phi 		= phi1 - phi2
	delta_lambda 	= lambda1 - lambda2

	# R is the earth's radius (6371 x 10³ meters)
	R = 6371e3

	a = math.sin(delta_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda/2)**2
	c = 2 * math.atan2(math.sqrt(a),math.sqrt(1-a))
	d = R * c

	return d
#end

def adjacency_dict_to_matrix(adjacency_dict):
	"""
		Get a binary adjacency matrix (Ma) and a floating-poing weight matrix
		(Mw) from the given adjacency dictionary
	"""
	# n: number of nodes in the graph
	n = len(list(adjacency_dict.keys()))

	Ma = np.zeros((n,n))
	Mw = np.zeros((n,n))

	# dict to convert node IDs to integers
	id_to_int = {}
	count = 0
	for node_id in adjacency_dict.keys():
		id_to_int[node_id] = count
		count += 1
	#end

	# For each node in our graph
	for node_id in adjacency_dict.keys():
		# Get adjacent nodes
		for neighbor_id in adjacency_dict[node_id]['adjacencies']:
			# Mark adjacency matrix
			Ma[id_to_int[node_id],id_to_int[neighbor_id]] = 1
			# Mark weight matrix with GPS distance
			Mw[id_to_int[node_id],id_to_int[neighbor_id]] = GPS_distance(
				adjacency_dict[node_id]['lat'],
				adjacency_dict[node_id]['lon'],
				adjacency_dict[neighbor_id]['lat'],
				adjacency_dict[neighbor_id]['lon'])
		#end
	#end

	return Ma, Mw
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

def create_dataset_random(n, path, max_dist=2, min_density=0.5, max_density=0.5, samples=1000):

	if not os.path.exists(path):
		os.makedirs(path)
	#end if

	for i,d in enumerate(np.linspace(min_density,max_density,samples)):
		Ma,W,solution = create_graph_random(n,d,max_dist)
		print("Writing graph file n,m=({},{})".format(Ma.shape[0], len(np.nonzero(Ma)[0])))
		write_graph(Ma,(1.0/max_dist)*W,solution,"{}/{}.graph".format(path,i))
	#end
#end

def create_dataset_metric(n, path, bins=10, samples=1000):

	if not os.path.exists(path):
		os.makedirs(path)
	#end if

	for i in range(samples):
		Ma,Mw,solution = create_graph_metric(n,bins)
		print("Writing graph file n,m=({},{})".format(Ma.shape[0], len(np.nonzero(Ma)[0])))
		write_graph(Ma,Mw,solution,"{}/{}.graph".format(path,i))
	#end
#end

def draw_adjacency_dict(adjacency_dict):
	for node_id in list(adjacency_dict.keys()):
		for neighbor_id in adjacency_dict[node_id]['adjacencies']:
			#print("Drawing line between nodes ID={} and ID={}".format(node_id, neighbor_id))
			plt.plot(
				[adjacency_dict[node_id]['lat'],adjacency_dict[neighbor_id]['lat']],
				[adjacency_dict[node_id]['lon'], adjacency_dict[neighbor_id]['lon']]
				)
		#end
	#end
	plt.show()
#end

def create_dataset_from_map(n, map_path, output_path, samples = 1000):
	
	# Process the map XML file into an adjacency dictionary
	adjacency_dict = XML_to_adjacency_dict(map_path)
	adjacency_dict = preprocess_adjacency_dict(adjacency_dict)

	print("Solving redenção's graph...")
	solution = solve(Ma,Mw)
	print("Solvable: {}".format(solution != []))

	# Create one subgraph per sample
	count_solvable = 0
	for i in range(samples):
		# Get random subgraph
		subgraph_adjacency_dict = create_subgraph_from_adjacency_dict(50,adjacency_dict)
		# Draw subgraph
		draw_adjacency_dict(subgraph_adjacency_dict)
		# Convert to matrix format
		Ma, Mw = adjacency_dict_to_matrix(subgraph_adjacency_dict)
		# Solve instance
		solution = solve(Ma,Mw)
		if len(solution) > 0:
			count_solvable += 1
		#end
		print("Solvable: {}".format(solution != []))
	#end
	print("{}% Solvable".format(count_solvable))
#end