import json
import xml.etree.ElementTree as ET
import json
import networkx as nx

tree = ET.parse('official/structure_release.xml')
root = tree.getroot()
fall11 = root[1]
all_nodes = set([node.attrib['wnid'] for node in fall11.iter()])
num_nodes_old = len(all_nodes)
print('# nodes (deduplicated):', num_nodes_old)

food = fall11[-1].findall(".//synset[@wnid='n00021265']")[0]
fall11.remove(fall11[-1])
fall11.append(food)
all_nodes = set([node.attrib['wnid'] for node in fall11.iter()])
all_nodes.add('fall11')
num_nodes = len(all_nodes)
print('# nodes (deduplicated) (removing fa11misc but adding food):', num_nodes)

def gen_edges(root, edge_set = None):
    assert(root.attrib['wnid'] in all_nodes)

    if edge_set is None:
        edge_set = []

    followed_nodes = [child for child in root]
    if len(followed_nodes) == 0:
        return

    for node in followed_nodes:
        assert(node.attrib['wnid'] in all_nodes)
        edge = (root.attrib['wnid'], node.attrib['wnid'])
        if edge not in edge_set:
            edge_set.append(edge)
        gen_edges(node, edge_set)

    return edge_set

edges = gen_edges(fall11)
graph = nx.DiGraph()
graph.add_edges_from(edges)
nodes = [node for node in graph.nodes()]
edges = [edge for edge in graph.edges()]

splits = json.load(open('official/imagenet-testsets.json', 'r'))
winter_2021 = [item.strip('\n') for item in open('official/winter_2021.txt', 'r').readlines()]

hop2 = []
counts = 0
for item in splits['2-hops']:
    if item in nodes and item in winter_2021:
        hop2.append(item)
        counts += 1
print('The rest set now contains {} classes, previously {}'.format(counts, len(splits['2-hops'])))

hop3 = []
counts = 0
for item in splits['3-hops']:
    if item in nodes and item in winter_2021:
        hop3.append(item)
        counts += 1
print('The rest set now contains {} classes, previously {}'.format(counts, len(splits['3-hops'])))

hop3_pure = []
counts = 0
for item in splits['3-hops-pure']:
    if item in nodes and item in winter_2021:
        hop3_pure.append(item)
        counts += 1
print('The rest set now contains {} classes, previously {}'.format(counts, len(splits['3-hops-pure'])))

target = {}
target['hop2'] = hop2
target['hop3'] = hop3
target['hop3_pure'] = hop3_pure


json.dump(target, open('process_results/splits_for_hops.json', 'w'))

"""
The rest set now contains 1533 classes, previously 1549
The rest set now contains 6986 classes, previously 7860
The rest set now contains 5453 classes, previously 6311
"""