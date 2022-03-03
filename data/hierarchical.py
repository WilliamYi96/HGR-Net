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
# saving edges will take care of everything
json.dump(edges, open('process_results/graph_edges.json', 'w'))
print('# Edegs, nodes in networkx: ', len(edges), len(nodes))


splits = json.load(open('official/imagenet-testsets.json', 'r'))
winter_2021 = [item.strip('\n') for item in open('official/winter_2021.txt', 'r').readlines()]

new_train = []

counts = 0

for item in splits['train'] :
    if item in nodes and item in winter_2021:
        new_train.append(item)
        counts += 1
print('The train set now contains {} classes, previously {}'.format(counts, len(splits['train'])))

rest = []
counts = 0
for item in splits['all']:
    if item in nodes and item in winter_2021:
        rest.append(item)
        counts += 1
print('The rest set now contains {} classes, previously {}'.format(counts, len(splits['all'])))

all = []
for item in new_train:
    if item not in all:
        all.append(item)

for item in rest:
    if item not in all:
        all.append(item)

print('Now all sets contains {} elements, previously {}'.format(len(all), len(splits['train']) + len(splits['all'])))

target = {}
target['train'] = new_train
target['rest'] = rest
target['all'] = all


json.dump(target, open('process_results/splits_for_tree.json', 'w'))