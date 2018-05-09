import networkx as nx
import numpy as np

def _get_graph( g_n, edge_probability, max_multiplier, factor ):
	nb = g_n//2
	na = g_n - nb
	
	Ga = nx.fast_gnp_random_graph( na, edge_probability )
	for s, t in Ga.edges:
		Ga[s][t]["distance"] = np.random.rand()
	#end
	Gb = nx.fast_gnp_random_graph( nb, edge_probability )
	for s, t in Gb.edges:
		Gb[s][t]["distance"] = np.random.rand()
	#end
	
	Gb = nx.relabel.convert_node_labels_to_integers(Gb, first_label=na)
	
	sa = np.random.randint( 0, na )
	distances = [ ( ta, nx.shortest_path_length( Ga, sa, ta, weight="distance" ) ) for ta in range(0,na) if ta != sa and nx.has_path( Ga, sa, ta ) ]
	distances.sort( key = lambda ta_d: ta_d[1] )
	if len( distances ) == 0:
		return None
	#end if
	ta,da = distances[-1]
	sb = np.random.randint( na, na+nb )
	distances = [ ( tb, nx.shortest_path_length( Gb, sb, tb, weight="distance" ) ) for tb in range(na, na+nb) if tb != sb and nx.has_path( Gb, sb, tb ) ]
	distances = [ (tb,db) for tb,db in distances if db < 2 + da/factor ]
	if len( distances ) == 0:
		return None
	#end if
	tb,db = distances[ np.random.randint( 0, len( distances ) ) ]
	
	G1 = nx.union( Ga, Gb )
	G1.add_edge( sa, sb )
	G1[sa][sb]["distance"] = np.random.rand()
	G2 = G1.copy()
	G2.add_edge( ta, tb )
	G2[ta][tb]["distance"] = np.random.rand()
	return G1, G2, sa, ta
#end _get_graph

def edge_in_path( path, s, t):
	if (s in path):
		tindex = path.index(s) + 1
	else:
		return False
	#end if
	return ( t in path ) and ( tindex < len( path ) ) and ( path[tindex] == t )
#end edge_in_path

def reindex_matrix( n, m, M ):
	new_index = []
	new_value = []
	for i, v in zip( M[0], M[1] ):
		s, t = i
		new_index.append( (n + s, m + t) )
		new_value.append( v )
	#end for
	return zip( new_index, new_value )
#end reindex_matrix

def create_graph( g_n, edge_probability, max_multiplier = 2 ):
	Gs = None
	while Gs is None:
		Gs = _create_graph( g_n, edge_probability, max_multiplier )
	#end while
	return Gs
#end create_graph

def _create_graph( g_n, edge_probability, max_multiplier = 2, factor = 2 ):
	prob = _get_graph( g_n, edge_probability, max_multiplier, factor )
	if prob is None:
		return None
	#end if
	G1, G2, sa, ta = prob
	
	d1 = nx.shortest_path_length( G1, sa, ta, weight="distance" )
	p1 = nx.shortest_path( G1, sa, ta, weight="distance" )
	d2 = nx.shortest_path_length( G2, sa, ta, weight="distance" )
	p2 = nx.shortest_path( G2, sa, ta, weight="distance" )
	if d1 < d2 or p1 == p2:
		return None
	#end if
	Ms_1_index = []
	Mt_1_index = []
	Mw_1_index = []
	Mw_1_values = []
	E1 = [ 0 for _ in G1.edges ]
	Ms_2_index = []
	Mt_2_index = []
	Mw_2_index = []
	Mw_2_values = []
	E2 = [ 0 for _ in G2.edges ]
	d1_norm = 0
	d2_norm = 0
	for e, (s, t) in enumerate( G1.edges ):
		Ms_1_index.append( (t, e) )
		Mt_1_index.append( (s, e) )
		Mt_1_index.append( (t, e) )
		Mw_1_index.append( (e, e) )
		Mw_1_values.append( G1[s][t]["distance"] )
		d1_norm += G1[s][t]["distance"]
		E1[e] = 1.0 if edge_in_path(p1,s,t) or edge_in_path(p1,t,s) else 0.0
	#end for
	for e, (s, t) in enumerate( G2.edges ):
		Ms_2_index.append( (t, e) )
		Mt_2_index.append( (s, e) )
		Mt_2_index.append( (t, e) )
		Mw_2_index.append( (e, e) )
		Mw_2_values.append( G2[s][t]["distance"] )
		d2_norm += G2[s][t]["distance"]
		E2[e] = 1.0 if edge_in_path(p2,s,t) or edge_in_path(p2,t,s) else 0.0
	#end for
	g1_m = len( G1.edges )
	g2_m = len( G2.edges )
	l1= lambda l: [1 for _ in l]
	Ms_1 = [Ms_1_index,l1(Ms_1_index),(g_n,g1_m)]
	Mt_1 = [Mt_1_index,l1(Mt_1_index),(g_n,g1_m)]
	Mw_1 = [Mw_1_index,Mw_1_values,(g1_m,g1_m)]
	Ms_2 = [Ms_2_index,l1(Ms_2_index),(g_n,g2_m)]
	Mt_2 = [Mt_2_index,l1(Mt_2_index),(g_n,g2_m)]
	Mw_2 = [Mw_2_index,Mw_2_values,(g2_m,g2_m)]
	S_mat = [[(sa,sa)],[1],(g_n,g_n)]
	T_mat = [[(ta,ta)],[1],(g_n,g_n)]
	d1_norm = d1_norm if d1_norm > 0 else 1
	d2_norm = d1_norm if d2_norm > 0 else 1
	return ((Ms_1,Mt_1,Mw_1,S_mat,T_mat,E1),d1/d1_norm), ((Ms_2,Mt_2,Mw_2,S_mat,T_mat,E2), d2/d2_norm), None
#end _create_graph

def create_batch(problems):
	n = 0
	m = 0
	batch_Ms_index = []
	batch_Ms_value = []
	batch_Mt_index = []
	batch_Mt_value = []
	batch_Mw_index = []
	batch_Mw_value = []
	batch_S_index = []
	batch_S_value = []
	batch_T_index = []
	batch_T_value = []
	batch_n_list = []
	batch_m_list = []
	batch_paths = []
	for p in problems:
		if p is None:
			continue
		#end if
		Ms, Mt, Mw, S, T, P = p
		for i, v in reindex_matrix( n, m, Ms ):
			batch_Ms_index.append( i )
			batch_Ms_value.append( v )
		#end for
		for i, v in reindex_matrix( n, m, Mt ):
			batch_Mt_index.append( i )
			batch_Mt_value.append( v )
		#end for
		for i, v in reindex_matrix( m, m, Mw ):
			batch_Mw_index.append( i )
			batch_Mw_value.append( v )
		#end for
		for i, v in reindex_matrix( n, n, S ):
			batch_S_index.append( i )
			batch_S_value.append( v )
		#end for
		for i, v in reindex_matrix( n, n, T ):
			batch_T_index.append( i )
			batch_T_value.append( v )
		#end for
		batch_paths += P
		bn = Ms[2][0]
		bm = Ms[2][1]
		n += bn
		m += bm
		batch_n_list.append( bn )
		batch_m_list.append( bm )
	#end for
	Ms = [batch_Ms_index,batch_Ms_value,(n,m)]
	Mt = [batch_Mt_index,batch_Mt_value,(n,m)]
	Mw = [batch_Mw_index,batch_Mw_value,(m,m)]
	S = [batch_S_index,batch_S_value,(n,n)]
	T = [batch_T_index,batch_T_value,(n,n)]
	return (Ms,Mt,Mw,S,T,batch_n_list,batch_m_list,batch_paths)
#end create_batch

def create_graph_nonquiver( g_n, edge_probability, max_multiplier = 2, factor = 2, normalize = False ):
	Gs = None
	while Gs is None:
		Gs = _create_graph_nonquiver( g_n, edge_probability, max_multiplier, factor, normalize )
	#end while
	return Gs
#end create_graph_nonquiver

def _create_graph_nonquiver( g_n, edge_probability, max_multiplier = 2, factor = 2, normalize = False ):
	prob = _get_graph( g_n, edge_probability, max_multiplier, factor )
	if prob is None:
		return None
	#end if
	G1, G2, sa, ta = prob
	
	d1 = nx.shortest_path_length( G1, sa, ta, weight="distance" )
	p1 = nx.shortest_path( G1, sa, ta, weight="distance" )
	d2 = nx.shortest_path_length( G2, sa, ta, weight="distance" )
	p2 = nx.shortest_path( G2, sa, ta, weight="distance" )
	if d1 < d2 or p1 == p2:
		return None
	#end if
	M1_index = []
	M1_values = []
	M2_index = []
	M2_values = []
	d1_norm = 0
	d2_norm = 0
	for s, t in G1.edges:
		M1_index.append( ( s, t ) )
		M1_values.append( 1 / ( 1 + G1[s][t]["distance"] ) )
		M1_index.append( ( t, s ) )
		M1_values.append( 1 / ( 1 + G1[s][t]["distance"] ) )
		d1_norm += G1[s][t]["distance"]
	#end for
	for s, t in G2.edges:
		M2_index.append( ( s, t ) )
		M2_values.append( 1 / ( 1 + G2[s][t]["distance"] ) )
		M2_index.append( ( t, s ) )
		M2_values.append( 1 / ( 1 + G2[s][t]["distance"] ) )
		d2_norm += G2[s][t]["distance"]
	#end for
	M1 = [M1_index,M1_values,(g_n,g_n)]
	M2 = [M2_index,M2_values,(g_n,g_n)]
	S_mat = [[(sa,sa)],[1],(g_n,g_n)]
	T_mat = [[(ta,ta)],[1],(g_n,g_n)]
	d1_norm = d1_norm if d1_norm > 0 and normalize else 1
	d2_norm = d2_norm if d2_norm > 0 and normalize else 1
	
	return ((M1,S_mat,T_mat),d1/d1_norm), ((M2,S_mat,T_mat),d2/d2_norm), None
#end _create_graph_nonquiver

def create_batch_nonquiver(problems):
	n = 0
	m = 0
	batch_M_index = []
	batch_M_value = []
	batch_S_index = []
	batch_S_value = []
	batch_T_index = []
	batch_T_value = []
	for p in problems:
		if p is None:
			continue
		#end if
		M, S, T = p
		for i, v in reindex_matrix( n, n, M ):
			batch_M_index.append( i )
			batch_M_value.append( v )
		#end for
		for i, v in reindex_matrix( n, n, S ):
			batch_S_index.append( i )
			batch_S_value.append( v )
		#end for
		for i, v in reindex_matrix( n, n, T ):
			batch_T_index.append( i )
			batch_T_value.append( v )
		#end for
		n += M[2][0]
		m += len(M[0])
	#end for
	M = [batch_M_index,batch_M_value,(n,n)]
	S = [batch_S_index,batch_S_value,(n,n)]
	T = [batch_T_index,batch_T_value,(n,n)]
	return (M,S,T)
#end create_batch_nonquiver
