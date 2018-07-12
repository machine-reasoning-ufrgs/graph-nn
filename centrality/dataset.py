import sys, os, time
import numpy as np
import networkx as nx
import json
# Add the parent folder path to the sys.path list for importing
sys.path.insert(1, os.path.join(sys.path[0], '..'))
# Import tools
from util import timestamp, memory_usage, sparse_to_dense

def calc_degree( G, normalized = True ):
  # Calculates the degree centrality, non-normalized
  degree = nx.degree_centrality( G )
  if not normalized:
    degree = { node:value * ( G.number_of_nodes() - 1 ) for node, value in degree.items() }
  #end
  return degree
#end calc_degree

def calc_betweenness( G, normalized = True ):
  # Calculates the betweenness centrality, non-normalized
  return nx.betweenness_centrality( G, normalized = normalized )
#end calc_betweenness

def calc_closeness( G, normalized = True ):
  # Calculates the closeness centrality
  closeness = { node:nx.closeness_centrality( G, node ) for node in G }
  if not normalized:
    g_n = len( G )
    closeness = { node: value * ( G.number_of_nodes() - 1 ) / ( len( nx.node_connected_component( G, node ) ) - 1 ) if 0 < len( nx.node_connected_component( G, node ) ) else 0 for node,value in closeness.items() }
  #end if
  return closeness
#end calc_closeness

def calc_eigenvector( G ):
  # Calculates the eigenvector centrality
  return nx.eigenvector_centrality( G )
#end calc_eigenvector

def create_graph( g_n, erdos_renyi_p = 0.25, powerlaw_gamma = 3, smallworld_k = 4, smallworld_p = 0.25, powerlaw_cluster_m = 3, powerlaw_cluster_p = 0.1 ):
  # Tries to create a graph until all of its centralities can be computed
  Gs = None
  while Gs is None:
    Gs = _create_graph( g_n, erdos_renyi_p, powerlaw_gamma, smallworld_k, smallworld_p, powerlaw_cluster_m, powerlaw_cluster_p )
  #end while
  return Gs
#end create_graph

def _create_graph( g_n, erdos_renyi_p = 0.25, powerlaw_gamma = 3, smallworld_k = 4, smallworld_p = 0.25, powerlaw_cluster_m = 3, powerlaw_cluster_p = 0.1 ):
  # Tries to create a graph and returns None if it fails at any point.
  # Create a graph from a random distribution
  graph_type = np.random.randint( 0, 4 )
  if graph_type == 0:
    G = nx.fast_gnp_random_graph( g_n, erdos_renyi_p )
  elif graph_type == 1:
    try:
      G = nx.random_graphs.random_powerlaw_tree( g_n, powerlaw_gamma )
    except nx.NetworkXError as e:
      print( e, file = sys.stderr, flush = True )
      return None
    #end try
  elif graph_type == 2:
    try:
      G = nx.random_graphs.connected_watts_strogatz_graph( g_n, smallworld_k, smallworld_p )
    except nx.NetworkXError as e:
      print( e, file = sys.stderr, flush = True )
      return None
    #end try
  elif graph_type == 3:
    try:
      G = nx.random_graphs.powerlaw_cluster_graph( g_n, powerlaw_cluster_m, powerlaw_cluster_p )
    except nx.NetworkXError as e:
      print( e, file = sys.stderr, flush = True )
      return None
    #end try
  #end if
  for s, t in G.edges:
    G[s][t]["weight"] = np.random.rand()
  #end for
  # Define a random target
  T = np.random.randint( 0, g_n )
  # Calculate centrality measures
  try:
    eigenvector = calc_eigenvector( G, T, g_n )
  except nx.exception.PowerIterationFailedConvergence as e:
    print( e, file = sys.stderr, flush = True )
    return None
  #end try
  degree = calc_degree( G, T, g_n )
  betweenness = calc_betweenness( G, T, g_n )
  closeness = calc_closeness( G, T, g_n )
  # Build matrices
  # TODO: Get values from the edges
  M_index, M_values = build_M_from_graph( G, key = "weight" )
  M = [M_index,M_values,(g_n,g_n)]
  return M,T,degree,betweenness,closeness,eigenvector
#end _create_graph

def create_dataset(
  instances,
  min_n        = 32,
  max_n        = 128,
  path         = "instances/erdos",
  graph_type   = "erdos_renyi",
  graph_args   = list(),
  graph_kwargs = dict()
):
  i = 0
  while i < instances:
    g_n = np.random.randint( min_n, max_n )
    
    if graph_type == "erdos_renyi":
      G = nx.fast_gnp_random_graph( g_n, *graph_args, **graph_kwargs )
    elif graph_type == "powerlaw_tree":
      try:
        G = nx.random_graphs.random_powerlaw_tree( g_n, *graph_args, **graph_kwargs )
      except nx.NetworkXError as e:
        print( e, file = sys.stderr, flush = True )
        continue
      #end try
    elif graph_type == "watts_strogatz":
      try:
        G = nx.random_graphs.connected_watts_strogatz_graph( g_n, *graph_args, **graph_kwargs )
      except nx.NetworkXError as e:
        print( e, file = sys.stderr, flush = True )
        continue
      #end try
    elif graph_type == "powerlaw_cluster":
      try:
        G = nx.random_graphs.powerlaw_cluster_graph( g_n, *graph_args, **graph_kwargs )
      except nx.NetworkXError as e:
        print( e, file = sys.stderr, flush = True )
        continue
      #end try
    else:
      raise InvalidArgumentError( "Graph type not supported" )
    #end if
    
    if len( nx.node_connected_component( G, 0 ) ) != g_n:
      # No disjoint subgraphs allowed
      continue
    #end
    
    degree_normalized = calc_degree( G, normalized = True )
    degree_not_normalized = calc_degree( G, normalized = False )
    betweenness_normalized = calc_betweenness( G, normalized = True )
    betweenness_not_normalized = calc_betweenness( G, normalized = False )
    closeness_normalized = calc_betweenness( G, normalized = True )
    closeness_not_normalized = calc_betweenness( G, normalized = False )
    try:
      eigenvector = calc_eigenvector( G )
    except nx.exception.PowerIterationFailedConvergence as e:
      continue
    #end
    
    for node in G:
      G.nodes[node]["d_n"] = degree_normalized[ node ]
      G.nodes[node]["d_p"] = degree_not_normalized[ node ]
      G.nodes[node]["b_n"] = betweenness_normalized[ node ]
      G.nodes[node]["b_p"] = betweenness_not_normalized[ node ]
      G.nodes[node]["c_n"] = closeness_normalized[ node ]
      G.nodes[node]["c_p"] = closeness_not_normalized[ node ]
      G.nodes[node]["e_n"] = eigenvector[ node ]
    #end for
    
    degree_rank = sorted( [ (centrality,node) for node,centrality in degree_normalized.items() ], key = lambda x: x[0], reverse = True )
    betweenness_rank = sorted( [ (centrality,node) for node,centrality in betweenness_normalized.items() ], key = lambda x: x[0], reverse = True )
    closeness_rank = sorted( [ (centrality,node) for node,centrality in closeness_normalized.items() ], key = lambda x: x[0], reverse = True )
    eigenvector_rank = sorted( [ (centrality,node) for node,centrality in eigenvector.items() ], key = lambda x: x[0], reverse = True )
    
    rank = 0
    while rank < G.number_of_nodes():
      _, node = degree_rank[ rank ]
      G.nodes[node]["d_r"] = rank
      _, node = betweenness_rank[ rank ]
      G.nodes[node]["b_r"] = rank
      _, node = closeness_rank[ rank ]
      G.nodes[node]["c_r"] = rank
      _, node = eigenvector_rank[ rank ]
      G.nodes[node]["e_r"] = rank
      rank += 1
    #end while
        
    data = nx.readwrite.json_graph.node_link_data( G )
    s = json.dumps( data )
    with open(
      "{path}/{i}.g".format(
        path=path,
        i=i
      ),
      mode = "w"
      
    ) as f:
      f.write(s)
    #end with open f
    
    i += 1
  #end while
#end create_dataset

if __name__ == "__main__":
  n_instances = int( sys.argv[1] )
  # Erdos Renyi dataset
  os.makedirs( "instances/erdos", exist_ok = True )
  create_dataset(
    instances = n_instances,
    min_n        = 32,
    max_n        = 128,
    path         = "instances/erdos",
    graph_type   = "erdos_renyi",
    graph_args   = [0.25],
    graph_kwargs = dict()
  )
  # Random Powerlaw Tree dataset
  os.makedirs( "instances/pltree", exist_ok = True )
  create_dataset(
    instances = n_instances,
    min_n        = 32,
    max_n        = 128,
    path         = "instances/pltree",
    graph_type   = "powerlaw_tree",
    graph_args   = [3],
    graph_kwargs = dict()
  )
  # Random Watts Strogatz Smallworld dataset
  os.makedirs( "instances/smallworld", exist_ok = True )
  create_dataset(
    instances = n_instances,
    min_n        = 32,
    max_n        = 128,
    path         = "instances/smallworld",
    graph_type   = "watts_strogatz",
    graph_args   = [4,0.25],
    graph_kwargs = dict()
  )
  # Random Powerlaw Tree dataset
  os.makedirs( "instances/plcluster", exist_ok = True )
  create_dataset(
    instances = n_instances,
    min_n        = 32,
    max_n        = 128,
    path         = "instances/plcluster",
    graph_type   = "powerlaw_cluster",
    graph_args   = [3,0.1],
    graph_kwargs = dict()
  )
