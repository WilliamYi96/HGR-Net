import networkx as nx
import random
import json

graph_edges = json.load(open('process_results/graph_edges.json', 'r'))
G = nx.DiGraph()
G.add_edges_from(graph_edges)

splits = json.load(open('process_results/splits_for_tree.json', 'r'))
avai = splits['all']

nodes = list(G.nodes())
nodes_to_remove = set(nodes) - set(avai)
nodes_to_remove.remove('fall11')

for rm in nodes_to_remove:
 in_edges = list(G.in_edges(rm))
 parents = [in_edge[0] for in_edge in in_edges]
 out_edges = list(G.out_edges(rm))
 children = [out_edge[1] for out_edge in out_edges]

 if rm in G.nodes:
  G.remove_node(rm)

 if len(out_edges) == 0:
  G.remove_edges_from(in_edges)
 else:
  edges_to_add = []
  for p in parents:
   for c in children:
    edges_to_add.append((p, c))
  G.remove_edges_from(in_edges + out_edges)
  G.add_edges_from(edges_to_add)

json.dump(list(G.edges), open('process_results/graph_edges_cls.json', 'w'))