import torch
import clip
from PIL import Image
from dataset import ImageNet, DataManager, DataManager_test
from torch.utils.data import DataLoader
from utils import cosine_lr, convert_models_to_fp32, convert_weights, accuracy, count_acc
import data.templates as templates
from utils import map_label, get_synsets
from utils import map_label
import argparse
import copy
import json
import torch.nn.functional as F
from collections import defaultdict
import networkx as nx
import numpy as np
import gc
import os
from torch.nn.parallel import DistributedDataParallel as DDP

parser = argparse.ArgumentParser(description='CLIP tree baseline')
parser.add_argument('--graph_path', default='data/process_results/graph_edges_cls.json', type=str)
parser.add_argument('--split_path', default='data/process_results/splits_for_tree.json', type=str)
parser.add_argument('--model_train', default='all', type=str)
parser.add_argument('--model_test', default='rest', type=str)
parser.add_argument('--data_train', default='train', type=str)
parser.add_argument('--data_test', default='rest', type=str)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--batch_size', default=64, type=int)
parser.add_argument('--test_batch_size', default=512, type=int)
parser.add_argument('--k_shots', default=-1, type=int)
parser.add_argument('--serial_batches', type=eval, default=True, choices=[True, False])
parser.add_argument('--n_episodes', default=-1, type=int)
parser.add_argument('--data_split_train', default='train', type=str, help="train, ls_train")
parser.add_argument('--data_split_test', default='zsl_test', type=str, help="val, ls_test, zsl_test")
parser.add_argument('--print_freq', default=1000, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--from_epoch', default=-1, type=int)
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument('--train', type=eval, default=True, choices=[True, False])
parser.add_argument('--lr', default=3e-7, type=float)
parser.add_argument('--exp_name', default='normal', type=str)
parser.add_argument('--load_path', type=str)
opts = parser.parse_args()

def gen_tree(opts,train_classes):
    graph_edges = json.load(open(opts.graph_path, 'r'))

    G = nx.DiGraph()
    G.add_edges_from(graph_edges)
    nodes = [node for node in G.nodes()]
    nodes.remove('fall11')
    ordered_nodes=[]
    for train_class in train_classes:
        ordered_nodes.append(train_class)
    for node in nodes:
        if node not in ordered_nodes:
            ordered_nodes.append(node)
    
    start_up = map_label(list(G['fall11']), ordered_nodes, batch=True)

    p2c = []
    for node in ordered_nodes:
        children = list(G[node])
        p2c.append(map_label(children, ordered_nodes, batch=True))

    c2p = []
    for node in ordered_nodes:
        parents = nx.shortest_path(G, source='fall11', target=node)[1:-1]
        c2p.append(map_label(parents, ordered_nodes, batch=True))

    for i, node in enumerate(ordered_nodes):
        for idx, p in enumerate(c2p[i]):
            cur_set = p2c[p]
            if idx < len(c2p[i]) - 1:
                assert(c2p[i][idx + 1] in cur_set)
            else:
                break
            
    d2n = defaultdict(list)
    for i, node in enumerate(ordered_nodes):
        depth_node = len(c2p[i])
        if i not in d2n[depth_node]:
            d2n[depth_node].append(i)

    return p2c, c2p, d2n, ordered_nodes, start_up


def test(test_index,train_index,model,p2c,c2p,d2n,nodes,start_up,splits,device,node_tokens):
    model.eval()
    with torch.no_grad():
            text_feats_1 = model.encode_text(node_tokens[:10000].cuda())
            text_feats_2 = model.encode_text(node_tokens[10000:].cuda())
            text_feats = torch.cat([text_feats_1, text_feats_2])
            text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)
    print('Loading datasets', flush=True)

    data = DataManager_test(opts=opts, split=opts.data_split_test, node_set=nodes,
                            candidates=splits[opts.data_test], resolution=224)
    loader_test = data.get_data_loader()
    num_batches = loader_test.batch_sampler.num_batch
    print('number of batches:{}'.format(num_batches))

    print('Running.', flush=True)
    with torch.no_grad():

        topk = (1, 2, 5, 10, 20)
        hits_dict = dict(zip(topk, [0] * len(topk)))
        num_sample = 0

        hits_all = 0
        path_all = 0
        path_all_count = 0
        point_all = 0

        for i, data in enumerate(loader_test):
            classes+=1
            imgs, targets = data['img'].to(device)[0], data['label'].to(device)[0]  # imgs [batch,3,224,224]
            feats=model.encode_image(imgs)
            feats /= feats.norm(dim=-1, keepdim=True)
            logits = (feats @ text_features.T)
            logits_ = logits[:, test_index]  # [batch_size,len(test)]
            maxk = max(topk)
            _, pred = logits_.topk(maxk, 1, True, True)
            pred = test_index[pred]
            pred = pred.t().cuda()
            correct = pred.eq(targets.reshape(1, -1).expand_as(pred))

            logits_ = logits[:, train_index]

            for k in topk:
                num_correct_k = correct[:k].reshape(-1).float().sum()
                hits_dict[k] += num_correct_k
            num_sample += len(targets)
            path_all_count += len(targets)

            target = targets[0].item()
            parents = copy.copy(c2p[target])
            parents.append(target)
            parent=torch.tensor(parents).expand(len(targets),len(parents))
            _, pred = logits_.topk(1, 1, True, True)
            pred = train_index[pred].cpu()
            pred = pred.expand(len(targets),len(parents))
            correct=pred.eq(parent).reshape(-1).float().sum()
            hits_all += correct

            
            dict_path = torch.zeros(len(targets), len(parents))
            for k, p in enumerate(parents):
                level = len(c2p[p])
                same_l = copy.copy(d2n[level])
                if p not in same_l:
                    same_l.append(p)
                    print("{} not in its level".format(p))
                rest = torch.tensor(list(set(list(range(len(nodes)))) - set(same_l))).cuda()
                logit_k = logits.detach().clone()
                logit_k = logit_k.index_fill(1, rest, -1)
                logit_k = logit_k[:, train_index]
                _, pred = logit_k.topk(1, 1, True, True)
                pred = train_index[pred]
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

            if i % opts.print_freq == 0:
                out_str = "\n"
                tmp_str, tmp_acc = count_acc(hits_dict, num_sample)
                # acc_list.append(tmp_acc)
                out_str += tmp_str
                hit_ratio = hits_all / num_sample * 100.0
                out_str += ' hit_ratio(%):{:.2f}'.format(hit_ratio)
                path_ratio = path_all / path_all_count * 100.0
                out_str += ' path_ratio(%):{:.2f}'.format(path_ratio)
                point_ratio = point_all / num_sample * 100.0
                out_str += ' point_ratio(%):{:.2f}'.format(point_ratio)
                print(out_str, flush=True)

        print('End of testing.')
        out_str = "\n"
        tmp_str, tmp_acc = count_acc(hits_dict, num_sample)
        out_str += tmp_str
        hit_ratio = hits_all / num_sample * 100.0
        out_str += ' hit_ratio(%):{:.2f}'.format(hit_ratio)
        path_ratio = path_all / path_all_count * 100.0
        out_str += ' path_ratio(%):{:.2f}'.format(path_ratio)
        point_ratio = point_all / num_sample * 100.0
        out_str += ' point_ratio(%):{:.2f}'.format(point_ratio)

        print(out_str, flush=True)

def train(model,train_dataloader,optim,node_tokens,num_batches):
    model.eval()
    for i, data in enumerate(train_dataloader):
        optim.zero_grad()
        imgs, targets = data['img'][0].cuda(), data['label'][0].cuda()
        feats=model.encode_image(imgs)
        feats = feats/feats.norm(dim=-1, keepdim=True)
        text_feats = model.encode_text(node_tokens[:983].cuda())
        text_features = text_feats / text_feats.norm(dim=-1, keepdim=True)
        logits = (feats @ text_features.T)*model.logit_scale.exp()
        loss = F.cross_entropy(logits, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        convert_models_to_fp32(model)
        optim.step()
        convert_weights(model)
        if i % opts.print_freq == 0:
            print('loss: {:.2f}, {}/{}'.format(loss, i, num_batches), flush=True)

def main():
    splits = json.load(open(opts.split_path, 'r'))
    train_classes=splits['train']
    p2c,c2p,d2n,nodes,start_up = gen_tree(opts,train_classes)
    template = getattr(templates, 'TEMPLATES_SIMPLE')[0]
    node_name = []
    for node in nodes:
        synset = get_synsets(node)
        name = synset.name().split('.')[0].replace('_', ' ')
        name = template.format(name)
        node_name.append(name)
    with torch.no_grad():
        node_tokens = clip.tokenize(node_name)
    splits = json.load(open(opts.split_path, 'r'))
    candidates_test=splits[opts.model_test]
    test_index = torch.tensor([nodes.index(item) for item in candidates_test])
    candidates_train=splits[opts.model_train]
    train_index = torch.tensor([nodes.index(item) for item in candidates_train])
    print('Creating models')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess = clip.load(name='RN50', device=device, download_root='pretrained')
    save_path='clip_check_points/{}/'.format(opts.exp_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if opts.load:
        model.load_state_dict(torch.load(opts.load_path))
    if opts.train:
        optim = torch.optim.AdamW(model.parameters(), lr=opts.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.StepLR(optim, gamma=0.1, step_size=25)
        data = DataManager(opts=opts, split=opts.data_split_train, node_set=nodes,
                    candidates=splits[opts.data_train], resolution=224)
        train_dataloader = data.get_data_loader()
        num_batches = data.n_episodes
        for epoch in range(10):
            print('epoch',epoch,flush=True)
            train(model,train_dataloader,optim,node_tokens,num_batches)
            file_path=save_path+'clip_{}'.format(epoch)
            torch.save(model.state_dict(), file_path)
            scheduler.step()
        test(test_index,train_index,model,p2c,c2p,d2n,nodes,start_up,splits,device,node_tokens)
    else:
        test(test_index,train_index,model,p2c,c2p,d2n,nodes,start_up,splits,device,node_tokens)

if __name__ == '__main__':
    main()

