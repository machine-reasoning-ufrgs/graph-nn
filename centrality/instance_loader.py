import numpy as np
import networkx as nx
import json
import os, sys, random
sys.path.insert(1, os.path.join(sys.path[0], '..'))
from util import sparse_to_dense

class InstanceLoader(object):

  def __init__(self,path):
    assert os.path.isdir( path ), "Path is not a directory. Path {}".format( path ) 
    if path[-1] == "/":
      path = path[0:-1]
    #end if
    folders = [path]
    self.filenames = []
    while len(folders)>0:
      newfolders = []
      for folder in folders:
        newfolders += [ f.path for f in os.scandir(folder) if f.is_dir() ]
        self.filenames += [ f.path for f in os.scandir(folder) if f.is_file() and f.path.endswith(".g") ]
      #end for
      folders = newfolders
    #end while
    self.reset()
  #end

  def get_instances(self, n_instances):
    for i in range(n_instances):
      with open( self.filenames[self.index], mode = "r" ) as f:
        s = f.read()
      #end with open f
      data = json.loads( s )
      G = nx.readwrite.json_graph.node_link_graph( data )
      g_n = G.number_of_nodes()
      M_index, M_values = build_M_from_graph( G )
      M = [M_index,M_values,(g_n,g_n)]
      degree = [ G.nodes[node]["d_n"] for node in G.nodes ]
      degree_compare = build_node_comparison_M_from_graph( G, "d_n" )
      betweenness = [ G.nodes[node]["b_n"] for node in G.nodes ]
      betweenness_compare = build_node_comparison_M_from_graph( G, "b_n" )
      closeness = [ G.nodes[node]["c_n"] for node in G.nodes ]
      closeness_compare = build_node_comparison_M_from_graph( G, "c_n" )
      eigenvector = [ G.nodes[node]["e_n"] for node in G.nodes ]
      eigenvector_compare = build_node_comparison_M_from_graph( G, "e_n" )
      yield M,degree,degree_compare,betweenness,betweenness_compare,closeness,closeness_compare,eigenvector,eigenvector_compare
      self.index += 1
    #end
  #end

  def get_batches(self, batch_size):
    for i in range( len(self.filenames) // batch_size ):
      yield create_batch(self.get_instances(batch_size))
    #end
  #end

  def get_batch(self, batch_size):
    return create_batch(self.get_instances(batch_size))
  #end

  def reset(self):
    random.shuffle( self.filenames )
    self.index = 0
  #end
#end


def build_M_from_graph( G, undirected = True, key = None ):
  # Build the sparse tensor matrices from a graph
  getval = lambda a: 1 if key is None else a[key]
  M_index = []
  M_values = []
  for s, t in G.edges:
    M_index.append( ( s, t ) )
    M_values.append( getval( G[s][t] ) )
    if undirected:
      M_index.append( ( t, s ) )
      M_values.append( getval( G[s][t] ) )
    #end if
  #end for
  return M_index, M_values
#end build_M_from_graph

def build_node_comparison_M_from_graph( G, key = None, comp = lambda a, b: 1.0 if a > b else 0.0 ):
  if key is None:
    raise ValueError( "Key for comparison cannot be None" )
  #end if
  # Build the sparse tensor matrices from a graph
  M_index = []
  M_values = []
  for a in G.nodes:
    for b in G.nodes:
      if a == b: continue
      M_index.append( ( a, b ) )
      M_values.append( comp( G.nodes[a][key], G.nodes[b][key] ) )
    #end for
  #end for
  return M_index, M_values
#end build_M_from_graph

def reindex_matrix( n, m, M ):
  # Reindex a sparse matrix
  new_index = []
  new_value = []
  for i, v in zip( M[0], M[1] ):
    s, t = i
    new_index.append( (n + s, m + t) )
    new_value.append( v )
  #end for
  return zip( new_index, new_value )
#end reindex_matrix

def create_batch(problems):
  # Create a problem-batch from the problem list passed
  n = 0
  m = 0
  batch_M_index = []
  batch_M_value = []
  deg = []
  deg_cmp_index = []
  deg_cmp_value = []
  bet = []
  bet_cmp_index = []
  bet_cmp_value = []
  clo = []
  clo_cmp_index = []
  clo_cmp_value = []
  eig = []
  eig_cmp_index = []
  eig_cmp_value = []
  problem_n = []
  problem_m = []
  for p in problems:
    if p is None:
      continue
    #end if
    M,degree,degree_compare,betweenness,betweenness_compare,closeness,closeness_compare,eigenvector,eigenvector_compare = p
    # Reindex the matrix to the new indexes
    for i, v in reindex_matrix( n, n, M ):
      batch_M_index.append( i )
      batch_M_value.append( v )
    #end for
    deg.append( degree )
    for i, v in reindex_matrix( n, n, degree_compare ):
      deg_cmp_index.append( i )
      deg_cmp_value.append( v )
    #end for
    bet.append( betweenness )
    for i, v in reindex_matrix( n, n, betweenness_compare ):
      bet_cmp_index.append( i )
      bet_cmp_value.append( v )
    #end for
    clo.append( closeness )
    for i, v in reindex_matrix( n, n, closeness_compare ):
      clo_cmp_index.append( i )
      clo_cmp_value.append( v )
    #end for
    eig.append( eigenvector )
    for i, v in reindex_matrix( n, n, eigenvector_compare ):
      eig_cmp_index.append( i )
      eig_cmp_value.append( v )
    #end for
    problem_n.append( M[2][0] )
    problem_m.append( len(M[0]) )
    # Update n and m
    n += M[2][0]
    m += len(M[0])
  #end for
  M = [batch_M_index,batch_M_value,(n,n)]
  deg_cmp_M = [deg_cmp_index,deg_cmp_value,(n,n)]
  bet_cmp_M = [bet_cmp_index,bet_cmp_value,(n,n)]
  clo_cmp_M = [clo_cmp_index,clo_cmp_value,(n,n)]
  eig_cmp_M = [eig_cmp_index,eig_cmp_value,(n,n)]
  batch = {
    "matrix": M,
    "degree": deg,
    "degree_compare": deg_cmp_M,
    "betweenness": bet,
    "betweenness_compare": bet_cmp_M,
    "closeness": clo,
    "closeness_compare": clo_cmp_M,
    "eigenvector": eig,
    "eigenvector_compare": eig_cmp_M,
    "problem_n": problem_n,
    "problem_m": problem_m
  }
  return batch
#end create_batch

if __name__ == '__main__':
  instance_loader = InstanceLoader("./instances")
  np.set_printoptions(threshold=np.nan,linewidth=np.nan)
  for b, batch in enumerate( instance_loader.get_batches(4) ):
    print( b, batch["problem_n"]) #, batch["degree"], np.sum( sparse_to_dense( batch["matrix"] ), axis = 1 ), np.sum( sparse_to_dense( batch["degree_compare"] ), axis = 1 ) )
  #end
#end
