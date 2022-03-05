from networkx.classes import ordered
import numpy as np
from dataset import ImageNet, DataManager, DataManager_test
np.random.seed(1)
import torch
from utils import map_label, get_synsets
torch.manual_seed(1)
import torch.nn as nn
import torch.nn.functional as F
from time import time
from torchvision import models, transforms,datasets
from torch.utils.data import DataLoader,Dataset
from PIL import Image
import PIL
from tqdm import tqdm
import json
import argparse
from torch.autograd import Variable
import networkx as nx
import copy
from collections import defaultdict
import gc
import clip
import data.templates as template

parser = argparse.ArgumentParser(description='CNZSL')
parser.add_argument('--print_freq', default=1, type=int)
parser.add_argument('--data_train', default='train', type=str)
parser.add_argument('--data_test', default='rest', type=str,help="rest,all")
parser.add_argument('--model_train', default='all', type=str,help="rest,all")
parser.add_argument('--model_test', default='rest', type=str,help="rest,all")
parser.add_argument('--graph_path', default='data/process_results/graph_edges_cls.json', type=str)
parser.add_argument('--split_path', default='data/process_results/splits_for_tree.json', type=str)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--test_batch_size', default=256, type=int)
parser.add_argument('--k_shots', default=-1, type=int)
parser.add_argument('--serial_batches', type=eval, default=True, choices=[True, False])
parser.add_argument('--n_episodes', default=-1, type=int)
parser.add_argument('--epoch', default=9, type=int)
parser.add_argument('--data_split_train', default='train', type=str, help="train, ls_train")
parser.add_argument('--data_split_test', default='zsl_test', type=str, help="val, ls_test, zsl_test,21kp_test")
parser.add_argument('--attr', default='transformer', type=str, help="w2v, transformer")
# resume
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument('--file_path', default='cnzsl/cnzsl_', type=str)
parser.add_argument('--train', default=True, type=eval, choices=[True, False])
parser.add_argument('--from_epoch', default=-1, type=int)
parser.add_argument('--cn', type=eval, default=True, choices=[True, False])
parser.add_argument('--init', type=eval, default=True, choices=[True, False])
parser.add_argument('--gzsl', type=eval, default=False, choices=[True, False])
opts = parser.parse_args()

USE_CLASS_STANDARTIZATION = opts.cn # i.e. equation (9) from the paper
USE_PROPER_INIT = opts.init # i.e. equation (10) from the paper

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        self.net = models.resnet50(pretrained=True).cuda()
 
    def forward(self, input):
        output = self.net.conv1(input)
        output = self.net.bn1(output)
        output = self.net.relu(output)
        output = self.net.maxpool(output)
        output = self.net.layer1(output)
        output = self.net.layer2(output)
        output = self.net.layer3(output)
        output = self.net.layer4(output)
        output = self.net.avgpool(output)
        return output


DEVICE = 'cuda'  # Set to 'cpu' if a GPU is not available

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

def get_attribute(ordered_nodes):
    attribute=json.load(open('attr.json','r'))
    attrs = torch.ones(len(ordered_nodes), 500)
    for i,node in enumerate(ordered_nodes):
        if node in attribute:
            attrs[i] = torch.tensor(attribute[node])
    return attrs

def count_acc(hits_dict, num_tot):
    out_str = ""
    for key, value in hits_dict.items():
        acc = value / num_tot * 100.0
        out_str += "Top@{}(%):{:.2f}".format(key, acc)
        if key != list(hits_dict.keys())[-1]:
            out_str += ", "
        else:
            out_str += '.'

    return out_str

class ClassStandardization(nn.Module):
    """
    Class Standardization procedure from the paper.
    Conceptually, it is equivalent to nn.BatchNorm1d with affine=False,
    but for some reason nn.BatchNorm1d performs slightly worse.
    """

    def __init__(self, feat_dim: int):
        super().__init__()

        self.running_mean = nn.Parameter(torch.zeros(feat_dim), requires_grad=False)
        self.running_var = nn.Parameter(torch.ones(feat_dim), requires_grad=False)

    def forward(self, class_feats):
        """
        Input: class_feats of shape [num_classes, feat_dim]
        Output: class_feats (standardized) of shape [num_classes, feat_dim]
        """
        if self.training:
            batch_mean = class_feats.mean(dim=0)
            batch_var = class_feats.var(dim=0)

            # Normalizing the batch
            result = (class_feats - batch_mean.unsqueeze(0)) / (batch_var.unsqueeze(0) + 1e-5)

            # Updating the running mean/std
            self.running_mean.data = 0.9 * self.running_mean.data + 0.1 * batch_mean.detach()
            self.running_var.data = 0.9 * self.running_var.data + 0.1 * batch_var.detach()
        else:
            # Using accumulated statistics
            # Attention! For the test inference, we cant use batch-wise statistics,
            # only the accumulated ones. Otherwise, it will be quite transductive
            result = (class_feats - self.running_mean.unsqueeze(0)) / (self.running_var.unsqueeze(0) + 1e-5)

        return result

train_classes = json.load(open(opts.split_path, 'r'))['train']
p2c,c2p,d2n,ordered_nodes,start_up = gen_tree(opts,train_classes)
train_mask=np.arange(0,len(train_classes))
splits = json.load(open('data/process_results/splits_for_hops.json', 'r'))
# splits = json.load(open(opts.split_path, 'r'))
if opts.attr=='w2v':
    attribute=get_attribute(ordered_nodes).to(DEVICE)
else:
    text_feats=json.load(open('text_feats.json','r'))
    attribute=torch.cat([torch.tensor(feat) for feat in text_feats],0).cuda()
# else:
#     clip_model, _ = clip.load(name='RN50', device=DEVICE, download_root='pretrained')
#     clip_model.eval()
candidates_test=splits[opts.data_test]
test_index = torch.tensor([ordered_nodes.index(item) for item in candidates_test]).to(DEVICE)

class CNZSLModel(nn.Module):
    def __init__(self, attr_dim: int, hid_dim: int, proto_dim: int):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(attr_dim, hid_dim),
            nn.ReLU(),

            nn.Linear(hid_dim, hid_dim),
            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.ReLU(),

            ClassStandardization(hid_dim) if USE_CLASS_STANDARTIZATION else nn.Identity(),
            nn.Linear(hid_dim, proto_dim),
            nn.ReLU(),
        )

        if USE_PROPER_INIT:
            weight_var = 1 / (hid_dim * proto_dim)
            b = np.sqrt(3 * weight_var)
            self.model[-2].weight.data.uniform_(-b, b)

    def forward(self, x, attrs): # attr [batch_size,x_dim]
        protos = self.model(attrs)  # [num_classes, x_dim]
        x_ns = 5 * x / x.norm(dim=1, keepdim=True)  # [batch_size, x_dim]
        protos_ns = 5 * protos / protos.norm(dim=1, keepdim=True)  # [num_classes, x_dim]
        logits = x_ns @ protos_ns.t()  # [batch_size, num_classes]

        return logits

def test():
    data = DataManager_test(opts=opts, split=opts.data_split_test, node_set=ordered_nodes,
                        candidates=splits[opts.data_test], resolution=224)
    test_dataloader = data.get_data_loader()
    model.eval()  # Important! Otherwise we would use unseen batch statistics
    topk = (1, 2, 5, 10, 20)
    hits_dict = dict(zip(topk, [0]*len(topk)))
    maxk = max(topk)
    num_sample=0
    hits_all=0
    len_parents_all=0
    path_all=0
    point_all=0
    path_all_count=0
    for i, data in enumerate(test_dataloader):
        imgs, targets = data['img'][0].to(DEVICE), data['label'][0].to(DEVICE)
        x = Variable(imgs.to(DEVICE).float(), requires_grad=False).to(DEVICE)
        feats = feature_encoder(x).view(-1,2048).to(DEVICE).detach()
        # print(feats,flush=True)
        logits = model(feats, attribute)
        logits_ = logits[:, test_index]
        # print(logits_.shape,flush=True)
        _, pred = logits_.topk(maxk, 1, True, True)
        pred = test_index[pred]
        pred = pred.t()
        correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
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
        pred = test_index[pred].cpu()
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
            rest = torch.tensor(list(set(list(range(len(ordered_nodes)))) - set(same_l))).cuda()
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

        if i % opts.print_freq == 0:
            out_str = "\n"
            tmp_str = count_acc(hits_dict, num_sample)
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
    tmp_str = count_acc(hits_dict, num_sample)
    out_str += tmp_str
    hit_ratio = hits_all / num_sample * 100.0
    out_str += ' hit_ratio(%):{:.2f}'.format(hit_ratio)
    path_ratio = path_all / path_all_count * 100.0
    out_str += ' path_ratio(%):{:.2f}'.format(path_ratio)
    point_ratio = point_all / num_sample * 100.0
    out_str += ' point_ratio(%):{:.2f}'.format(point_ratio)
    print(out_str, flush=True)

    log_all = 'result_cnzsl/cnzsl.txt'
    with open(log_all, 'a') as f:
        info = 'attr:{},CN:{},INIT:{}:'.format(opts.attr, opts.cn, opts.init)
        f.writelines(info + '\n')
        f.writelines(out_str + '\n')

def train(train_dataloader,optim):
    model.train()
    for i, data in enumerate(train_dataloader):
        imgs, targets = data['img'][0].to(DEVICE), data['label'][0].to(DEVICE)
        x = Variable(imgs.to(DEVICE).float(), requires_grad=False).to(DEVICE)
        feats = feature_encoder(x).view(-1,2048).to(DEVICE)
        feats=feats.detach()
        logits = model(feats, attribute[train_mask])
        loss = F.cross_entropy(logits, targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if i % opts.print_freq == 0:
            print('loss: {:.2f}'.format(loss), flush=True)
    scheduler.step()

print(f'\n<=============== Starting training ===============>',flush=True)
# start_time = time()

attr_shape=500 if opts.attr=='w2v' else 1024
model = CNZSLModel(attr_shape, 1024, 2048).to(DEVICE)
# template = getattr(template, 'TEMPLATES_SIMPLE')[0]
# node_name = []
# for node in ordered_nodes:
#     synset = get_synsets(node)
#     name = synset.name().split('.')[0].replace('_', ' ')
#     name = template.format(name)
#     node_name.append(name)
# with torch.no_grad():
#     node_tokens = clip.tokenize(node_name).to(DEVICE)
if opts.load:
    model.load_state_dict(torch.load(opts.file_path))
feature_encoder = Res50().to(DEVICE)
feature_encoder.eval()
if opts.train:
    optim = torch.optim.Adam(model.model.parameters(), lr=0.0001, weight_decay=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optim, gamma=0.1, step_size=25)
    data = DataManager(opts=opts, split=opts.data_split_train, node_set=ordered_nodes,
                           candidates=splits[opts.data_train], resolution=224)
    train_dataloader = data.get_data_loader()
    for epoch in range(opts.epoch):
        print('epoch:{}'.format(epoch))
        train(train_dataloader,optim)
        file_path=opts.file_path+'_{}'.format(epoch)
        if epoch>5:
            torch.save(model.state_dict(), file_path)
else:
    # print('from epoch{}'.format(opts.from_epoch),flush=True)
    # out_str="from epoch{}".format(opts.from_epoch)+'\n'
    # with open('cnzsl/cnzsl.txt','a') as f:
    #     f.writelines(out_str)
    print('attr:{},cn:{},init:{}'.format(opts.attr,opts.cn,opts.init),flush=True)
    test()

# print(f'Training is done! Took time: {(time() - start_time): .1f} seconds',flush=True)
