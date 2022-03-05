#######################
#author: Shiming Chen
#FREE
#######################
from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
import numpy as np
import math
import sys
from sklearn import preprocessing
import csv
#import functions
import model
import util
import classifier as classifier_zero

from config import opt
import time
from center_loss import TripCenterLoss_min_margin,TripCenterLoss_margin
from dataset import ImageNet, DataManager, DataManager_test
from torch.utils.data import DataLoader
import json
import networkx as nx
import copy
from collections import defaultdict
from torchvision import models

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)
cudnn.benchmark = True
if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")
# load data
# data = util.DATA_LOADER(opt)
# print("# of training samples: ", data.ntrain)

netE = model.Encoder(opt)
netG = model.Generator(opt)
netD = model.Discriminator(opt)

center_criterion = TripCenterLoss_margin(num_classes=opt.nclass_seen, feat_dim=opt.attSize, use_gpu=opt.cuda)

 
netFR = model.FR(opt, opt.attSize)
 
###########
# Init Tensors
input_res = torch.FloatTensor(opt.batch_size, opt.resSize)
input_att = torch.FloatTensor(opt.batch_size, opt.attSize) #attSize class-embedding size
noise = torch.FloatTensor(opt.batch_size, opt.nz)
input_label = torch.LongTensor(opt.batch_size)
#one = torch.FloatTensor([1])
one = torch.tensor(1, dtype=torch.float)
mone = one * -1
beta=0
##########
# Cuda
if opt.cuda:
    netD.cuda()
    netE.cuda()
    netG.cuda()

    netFR.cuda()
    input_res = input_res.cuda()
    noise, input_att = noise.cuda(), input_att.cuda()
    one = one.cuda()
    mone = mone.cuda()
    input_label=input_label.cuda()

def loss_fn(recon_x, x, mean, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(recon_x+1e-12, x.detach(),reduction='sum')
    BCE = BCE.sum()/ x.size(0)
    KLD = -0.5 * torch.sum(1 + log_var - mean.pow(2) - log_var.exp())/ x.size(0)
    #return (KLD)
    return (BCE + KLD)
           
def sample():
    batch_feature, batch_label, batch_att = data.next_seen_batch(opt.batch_size)
    input_res.copy_(batch_feature)
    input_att.copy_(batch_att)
    input_label.copy_(util.map_label(batch_label, data.seenclasses))

def WeightedL1(pred, gt):
    wt = (pred-gt).pow(2)
    wt /= wt.sum(1).sqrt().unsqueeze(1).expand(wt.size(0),wt.size(1))
    loss = wt * (pred-gt).abs()
    return loss.sum()/loss.size(0)
    



optimizer          = optim.Adam(netE.parameters(), lr=opt.lr)
optimizerD         = optim.Adam(netD.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerG         = optim.Adam(netG.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))
optimizerFR      = optim.Adam(netFR.parameters(), lr=opt.dec_lr, betas=(opt.beta1, 0.999))
optimizer_center   = optim.Adam(center_criterion.parameters(), lr=opt.lr,betas=(opt.beta1, 0.999))

def calc_gradient_penalty(netD,real_data, fake_data, input_att):
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(input_att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty

def calc_gradient_penalty_FR(netFR, real_data, fake_data):
    #print real_data.size()
    alpha = torch.rand(opt.batch_size, 1)
    alpha = alpha.expand(real_data.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * real_data + ((1 - alpha) * fake_data)
    if opt.cuda:
        interpolates = interpolates.cuda()

    interpolates = Variable(interpolates, requires_grad=True)
    _,_,disc_interpolates,_ ,_, _ = netFR(interpolates)
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


def MI_loss(mus, sigmas, i_c, alpha=1e-8):
    kl_divergence = (0.5 * torch.sum((mus ** 2) + (sigmas ** 2)
                                  - torch.log((sigmas ** 2) + alpha) - 1, dim=1))

    MI_loss = (torch.mean(kl_divergence) - i_c)

    return MI_loss

def optimize_beta(beta, MI_loss,alpha2=1e-6):
    beta_new = max(0, beta + (alpha2 * MI_loss))

    # return the updated beta value:
    return beta_new



if not os.path.exists(opt.result_root):
    os.makedirs(opt.result_root)

class Res50(nn.Module):
    def __init__(self):
        super(Res50, self).__init__()
        self.net = models.resnet50(pretrained=True)
 
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
def map_label(names, all_names, batch=False):
    if batch:
        return [all_names.index(name) for name in names]
    else:
        return all_names.index(names)

def gen_tree(train_classes):
    graph_edges = json.load(open('data/process_results/graph_edges_cls.json', 'r'))

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

splits = json.load(open('data/process_results/splits_for_tree.json', 'r'))
train_classes=splits['train']
p2c,c2p,d2n,ordered_nodes,start_up = gen_tree(train_classes)
train_mask=np.arange(0,len(train_classes))
candidates_test=splits['rest']
test_index = torch.tensor([ordered_nodes.index(item) for item in candidates_test])
if opt.attSize==1024:
    text_feats=json.load(open('text_feats.json','r'))
    attribute=torch.cat([torch.tensor(feat) for feat in text_feats],0)
else:
    test_index = torch.tensor([ordered_nodes.index(item) for item in candidates_test]).cuda()
    attribute=get_attribute(ordered_nodes).cuda()
attribute /= attribute.pow(2).sum(1).sqrt().unsqueeze(1).expand(attribute.size(0),attribute.size(1))
feature_encoder = Res50()
data = ImageNet(split='train', node_set=ordered_nodes, candidates=splits['train'], resolution=224)
train_dataloader = DataLoader(data, batch_size=512, shuffle=True)
# data = DataManager(split='train', node_set=ordered_nodes, candidates=splits['train'], resolution=224)
# train_dataloader = data.get_data_loader()

best_zsl_acc = 0
best_zsl_epoch=0

netG.load_state_dict(torch.load(opt.load_G)['netG'])
netFR.load_state_dict(torch.load(opt.load_F)['netFR'])

zsl_cls = classifier_zero.CLASSIFIER(netG, feature_encoder,p2c,c2p,d2n,ordered_nodes,test_index, attribute, train_dataloader, 18278, opt.cuda, opt.classifier_lr, 0.5,20, netFR=netFR, dec_size=opt.attSize, dec_hidden_size=(opt.latensize*2))

zsl_cls.fit_zsl()
torch.save(zsl_cls.model.state_dict(), 'cls_{}'.format(opt.attSize))
data = DataManager_test(split='zsl_test', node_set=ordered_nodes, candidates=splits['rest'], resolution=224)
test_dataloader = data.get_data_loader()
zsl_cls.test(test_dataloader)

