import argparse
import json
import os.path as osp

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from models.resnet import make_resnet50_base
from datasets.imagenet import ImageNet
from utils import set_gpu, pick_vectors
from collections import defaultdict
import networkx as nx
import copy
import numpy as np
import gc

test_index = torch.from_numpy(np.arange(0,18278)).cuda()

def map_label(names, all_names, batch=False):
    if batch:
        return [all_names.index(name) for name in names]
    else:
        return all_names.index(names)

def gen_tree(nodes_name):
    graph_edges = json.load(open('data/process_results/graph_edges_cls.json', 'r'))

    G = nx.DiGraph()
    G.add_edges_from(graph_edges)
    nodes = [node for node in G.nodes()]
    nodes.remove('fall11')
    start_up = map_label(list(G['fall11']), nodes_name, batch=True)

    p2c = []
    for node in nodes_name:
        children = list(G[node])
        p2c.append(map_label(children, nodes_name, batch=True))

    c2p = []
    for node in nodes_name:
        parents = nx.shortest_path(G, source='fall11', target=node)[1:-1]
        c2p.append(map_label(parents, nodes_name, batch=True))

    for i, node in enumerate(nodes_name):
        for idx, p in enumerate(c2p[i]):
            cur_set = p2c[p]
            if idx < len(c2p[i]) - 1:
                assert(c2p[i][idx + 1] in cur_set)
            else:
                break
            
    d2n = defaultdict(list)
    for i, node in enumerate(nodes_name):
        depth_node = len(c2p[i])
        if i not in d2n[depth_node]:
            d2n[depth_node].append(i)

    return p2c, c2p, d2n, nodes_name, start_up

class tree():
    def __init__(self,nodes_name):
        self.p2c, self.c2p, self.d2n, self.nodes, self.start_up = gen_tree(nodes_name)

def test_on_subset(dataset, cnn, n, pred_vectors, all_label,tree,consider_trains):
    torch.cuda.empty_cache()
    gc.collect()
    top = [1, 2, 5, 10, 20]
    hits = torch.zeros(len(top)).cuda()
    tot = 0
    hits_all=0
    len_parents_all=0
    path_all=0
    point_all=0

    loader = DataLoader(dataset=dataset, batch_size=256,
                        shuffle=False, num_workers=12)

    for batch_id, batch in enumerate(loader, 1):
        data, label = batch
        target=torch.tensor(tree.nodes.index(label[0]))
        targets=target.expand(len(label))
        data = data.cuda().half()

        feat = cnn(data) # (batch_size, d)
        feat = torch.cat([feat, torch.ones(len(feat)).view(-1, 1).cuda()], dim=1).half()
        fcs = pred_vectors.t()

        table = torch.matmul(feat, fcs)
        logits=copy.copy(table)
        
        if not consider_trains:
            table[:, :n] = 1e-7
        
        logits_ = logits[:, test_index]

        # _, pred = logits_.topk(20, 1, True, True)
        # pred = test_index[pred]
        # pred = pred.t()
        # correct = pred.eq(targets.reshape(1, -1).expand_as(pred))

        target = targets[0].item()
        parents = copy.copy(tree.c2p[target])
        parents.append(target)
        parent=torch.tensor(parents).expand(len(targets),len(parents))
        _, pred = logits_.topk(1, 1, True, True)
        pred = test_index[pred].cpu()
        pred = pred.expand(len(targets),len(parents))
        correct=pred.eq(parent).reshape(-1).float().sum()
        hits_all += correct

        dict_path = torch.zeros(len(targets), len(parents))
        for k, p in enumerate(parents):
            level = len(tree.c2p[p])
            same_l = copy.copy(tree.d2n[level])
            if p not in same_l:
                same_l.append(p)
                print("{} not in its level".format(p))
            rest = torch.tensor(list(set(list(range(len(tree.nodes)))) - set(same_l))).cuda()
            logit_k = logits.detach().clone()
            logit_k = logit_k.index_fill(1, rest, -1)
            logit_k = logit_k[:, test_index]
            _, pred = logit_k.topk(1, 1, True, True)
            pred = test_index[pred]
            pred = pred.squeeze()
            dict_path[:, k] = pred
        edge = 0
        point = 0
        for i in range(dict_path.shape[0]):
            if (len(parents) - 1) == 0 and parents[0] == dict_path[i][0]:
                path_all += 1
            for j in range(len(parents) - 1):
                if parents[j] == dict_path[i][j]:
                    point += 1
                if parents[j] == dict_path[i][j] and parents[j + 1] == dict_path[i][j + 1]:
                    edge += 1
            if parents[len(parents) - 1] == dict_path[i][len(parents) - 1]:
                point += 1
        if (len(parents) - 1) != 0:
            path_all += edge / (len(parents) - 1)
        point_all += point / (len(parents))

        gth_score = table[:, all_label].repeat(table.shape[1], 1).t()
        rks = (table >= gth_score).sum(dim=1)

        assert (table[:, all_label] == gth_score[:, all_label]).min() == 1

        for i, k in enumerate(top):
            hits[i] += (rks <= k).sum().item()
        tot += len(data)

    return hits, tot, hits_all, len_parents_all, path_all, point_all


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn',default='FSL/dgp/1/epoch_1.pth')
    parser.add_argument('--pred',default='gcn_dense_att/zsl/epoch_3000.pred')

    parser.add_argument('--test-set',default='rest')

    parser.add_argument('--output', default=None)

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--keep-ratio', type=float, default=1.0)
    parser.add_argument('--consider-trains', action='store_true')
    parser.add_argument('--test-train', action='store_true')
    parser.add_argument('--split', default='ls_test',type=str)

    args = parser.parse_args()

    set_gpu(args.gpu)

    test_sets = json.load(open('data/process_results/splits_for_tree.json', 'r'))
    train_wnids = test_sets['train']
    test_wnids = test_sets[args.test_set]

    nodes_name=train_wnids+test_wnids
    tree=tree(nodes_name)

    print('test set: {}, {} classes, ratio={}'
          .format(args.test_set, len(test_wnids), args.keep_ratio),flush=True)
    print('consider train classifiers: {}'.format(args.consider_trains),flush=True)

    pred_file = torch.load(args.pred)
    pred_wnids = pred_file['wnids']
    pred_vectors = pred_file['pred']
    # assert pred_wnids==test_sets['all']
    pred_dic = dict(zip(pred_wnids, pred_vectors))
    pred_vectors = pick_vectors(pred_dic, train_wnids + test_wnids, is_tensor=True)

    pred_vectors = pred_vectors.cuda().half()

    n = len(train_wnids)
    m = len(test_wnids)
    
    cnn = make_resnet50_base()
    cnn.load_state_dict(torch.load(args.cnn))
    cnn = cnn.cuda().half()
    cnn.eval()

    TEST_TRAIN = args.test_train

    imagenet_path = "/ibex/ai/reference/CV/ILSVR/2021/data/raw/train"
    dataset = ImageNet(imagenet_path,args)
    dataset.set_keep_ratio(args.keep_ratio)

    s_hits = torch.FloatTensor([0, 0, 0, 0, 0]).cuda() # top 1 2 5 10 20
    s_tot = 0
    s_to=0
    s_path=0
    s_point=0
    s_parents=0

    results = {}

    if TEST_TRAIN:
        for i, wnid in enumerate(train_wnids, 1):
            subset = dataset.get_subset(wnid)
            hits, tot = test_on_subset(subset, cnn, n, pred_vectors, i - 1,consider_trains=args.consider_trains)
            results[wnid] = (hits / tot).tolist()

            s_hits += hits
            s_tot += tot

            print('{}/{}, {}:'.format(i, len(train_wnids), wnid), end=' ')
            for i in range(len(hits)):
                print('{:.0f}%({:.2f}%)'
                      .format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ')
            print('x{}({})'.format(tot, s_tot))
    else:
        for i, wnid in enumerate(test_wnids, 1):
            subset = dataset.get_subset(wnid)
            if len(subset)>0:
                hits, tot, hits_all, len_parents_all, path_all, point_all = test_on_subset(subset, cnn, n, pred_vectors, n + i - 1 ,tree, consider_trains=args.consider_trains)
                results[wnid] = (hits / tot).tolist()

                s_hits += hits
                s_tot += tot
                s_to += hits_all
                s_parents+=len_parents_all
                s_path+=path_all
                s_point+=point_all

                print('{}/{}, {}:'.format(i, len(test_wnids), wnid), end=' ',flush=True)
                for i in range(len(hits)):
                    print('{:.0f}%({:.2f}%)'
                        .format(hits[i] / tot * 100, s_hits[i] / s_tot * 100), end=' ',flush=True)
                print('To:{:.2f}%'.format(s_to/s_tot*100), end=' ',flush=True)
                print('Path:{:.2f}%'.format(s_path/s_tot*100), end=' ',flush=True)
                print('Point:{:.2f}%'.format(s_point/s_tot*100), end=' ',flush=True)
                print('x{}({})'.format(tot, s_tot),flush=True)

    print('summary:', end=' ')
    for s_hit in s_hits:
        print('{:.2f}%'.format(s_hit / s_tot * 100), end=' ',flush=True)
    print('total {}'.format(s_tot),flush=True)

    if args.output is not None:
        json.dump(results, open(args.output, 'w'))
