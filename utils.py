import os
import os.path as osp
import networkx as nx
import json
import clip
import data.templates as template
from nltk.corpus import wordnet as wn
import torch
from tqdm import tqdm
import numpy as np
import torch.nn as nn
import ipdb
from collections import defaultdict


def map_label(names, all_names, batch=False):
    if batch:
        return [all_names.index(name) for name in names]
    else:
        return all_names.index(names)

def listdir_nohidden(path, sort=False):
    """List non-hidden items in a directory.
    Args:
         path (str): directory path.
         sort (bool): sort the items.
    """
    items = [f for f in os.listdir(path) if not f.startswith(".")]
    if sort:
        items.sort()
    return items

def is_file_exists(filepath):
    return osp.exists(filepath)

def get_synsets(wnid):
    return wn.synset_from_pos_and_offset('n', int(wnid[1:]))

def gen_tree(opts):
    graph_edges = json.load(open(opts.graph_path, 'r'))

    G = nx.DiGraph()
    G.add_edges_from(graph_edges)
    nodes = [node for node in G.nodes()]
    nodes.remove('fall11')
    start_up = map_label(list(G['fall11']), nodes, batch=True)

    p2c = []
    for node in nodes:
        children = list(G[node])
        p2c.append(map_label(children, nodes, batch=True))

    c2p = []
    for node in nodes:
        parents = nx.shortest_path(G, source='fall11', target=node)[1:-1]
        c2p.append(map_label(parents, nodes, batch=True))

    for i, node in enumerate(nodes):
        for idx, p in enumerate(c2p[i]):
            cur_set = p2c[p]
            if idx < len(c2p[i]) - 1:
                assert(c2p[i][idx + 1] in cur_set)
            else:
                break
            
    d2n = defaultdict(list)
    for i, node in enumerate(nodes):
        depth_node = len(c2p[i])
        if i not in d2n[depth_node]:
            d2n[depth_node].append(i)

    return p2c, c2p, d2n, nodes, start_up

def assign_learning_rate(param_group, new_lr):
    param_group["lr"] = new_lr


def _warmup_lr(base_lr, warmup_length, step):
    return base_lr * (step + 1) / warmup_length


def cosine_lr(optimizer, base_lrs, warmup_length, steps):
    if not isinstance(base_lrs, list):
        base_lrs = [base_lrs for _ in optimizer.param_groups]
    assert len(base_lrs) == len(optimizer.param_groups)
    def _lr_adjuster(step):
        for param_group, base_lr in zip(optimizer.param_groups, base_lrs):
            if step < warmup_length:
                lr = _warmup_lr(base_lr, warmup_length, step)
            else:
                e = step - warmup_length
                es = steps - warmup_length
                lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
            assign_learning_rate(param_group, lr)
    return _lr_adjuster


def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        p.grad.data = p.grad.data.float()

def convert_weights(model):
    """Convert applicable model parameters to fp16"""
    def _convert_weights_to_fp16(l):
        if isinstance(l, (nn.Conv1d, nn.Conv2d, nn.Linear)):
            l.weight.data = l.weight.data.half()
            if l.bias is not None:
                l.bias.data = l.bias.data.half()

        if isinstance(l, nn.MultiheadAttention):
            for attr in [*[f"{s}_proj_weight" for s in ["in", "q", "k", "v"]], "in_proj_bias", "bias_k", "bias_v"]:
                tensor = getattr(l, attr)
                if tensor is not None:
                    tensor.data = tensor.data.half()

        for name in ["text_projection", "proj", 'zsl_weights']:
            if hasattr(l, name):
                attr = getattr(l, name)
                if attr is not None:
                    attr.data = attr.data.half()

    model.apply(_convert_weights_to_fp16)

def accuracy(pred, labels):
    assert len(pred) == len(labels)

    pred = np.array(pred)
    labels = np.array(labels)

    acc = (pred == labels).sum() / len(pred)

    return acc

def count_acc(hits_dict, num_tot):
    out_str = ""
    acc_dict=dict()
    for key, value in hits_dict.items():
        acc = value / num_tot * 100.0
        acc_dict[key]=acc
        out_str += "Top@{}(%):{:.2f}".format(key, acc)
        if key != list(hits_dict.keys())[-1]:
            out_str += ", "
        else:
            out_str += '.'

    return out_str,acc_dict