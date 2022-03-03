import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import clip
from utils import map_label, get_synsets
from nltk.corpus import wordnet as wn
import ipdb
from tqdm import tqdm
from functools import partial
from utils import gen_tree, map_label
import copy
import data.templates as template
import gc
import random
import math
import os

class tree_model(nn.Module):
    def __init__(self, opts, candidates_train, candidates_test):
        super(tree_model, self).__init__()
        self.opts = opts
        self.device = opts.device
        self.save_path = '{}/{}/{}_{}_{}/'.format(opts.folder, opts.exp_name, opts.weights, opts.out_ratio, opts.in_ratio)
        self.file_path=self.save_path+'clip_{}'.format(opts.from_epoch)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Semantic structure
        self.p2c, self.c2p, self.d2n, self.nodes, self.start_up = gen_tree(self.opts)
        self.nodes_id = list(range(len(self.nodes)))

        # CLIP model
        self.clip_model, _ = clip.load(name=opts.arch, device=self.device, download_root='pretrained')

        if opts.fetch:
            self.clip_model.load_state_dict(torch.load(opts.fetch_path))
        if opts.load:
            if opts.load_path=='none':
                self.clip_model.load_state_dict(torch.load(self.file_path))
            else:
                self.clip_model.load_state_dict(torch.load(opts.load_path))
            print("successfully loaded")
            

        self.clip_model.eval()
        for params in self.clip_model.parameters():
            params.requires_grad_(True)
        self.loss = nn.CrossEntropyLoss()

        # Get tokens
        self.template = getattr(template, 'TEMPLATES_SIMPLE')[0]
        node_name = []
        for node in self.nodes:
            synset = get_synsets(node)
            name = synset.name().split('.')[0].replace('_', ' ')
            name = self.template.format(name)
            node_name.append(name)
        with torch.no_grad():
            self.node_tokens = clip.tokenize(node_name).to(self.device)

        # Misc
        self.resolution = self.clip_model.visual.input_resolution
        self.candidates_train = candidates_train
        self.candidates_test = candidates_test
        self.train_index = torch.tensor([self.nodes.index(item) for item in self.candidates_train]).to(self.device)
        self.test_index = torch.tensor([self.nodes.index(item) for item in self.candidates_test]).to(self.device)
        self.max_depth = max(self.d2n.keys())

        if self.opts.weights == 'adaptive':
            # self.layer_weight = nn.Parameter(torch.ones(self.max_depth + 1))
            num_layer=[len(self.d2n[layer]) for layer in self.d2n.keys()]
            weights=1.0/torch.tensor(num_layer)
            self.layer_weight = nn.Parameter(weights)*self.opts.scale

    def save(self, opts, epoch):
        file_path=self.save_path+'clip_{}'.format(epoch)
        torch.save(self.clip_model.state_dict(), file_path)

    def get_contra(self, method, target, batch_size, depth=None, parents=None):
        if method == 'random':
            compare_idx = random.sample(self.train_index.tolist(), self.opts.num_compare)
            if target not in compare_idx:
                compare_idx.append(target)

            compare_idx = torch.tensor(compare_idx).to(self.device)
            targets = (compare_idx == target).nonzero()[0].repeat(batch_size)

            return compare_idx, targets

        elif method == 'simi':
            with torch.no_grad():
                self_name = self.node_tokens[target].unsqueeze(0)
                other = self.node_tokens[self.train_index]
                candi = set(self.train_index)
                children=self.p2c[target]
                other = list(candi - set(parents)-set(children))
                text_feats_self = self.clip_model.encode_text(self_name)
                text_feats_other = self.clip_model.encode_text(other)
                text_feats_self /= text_feats_self.norm(dim=-1, keepdim=True)
                text_feats_other /= text_feats_other.norm(dim=-1, keepdim=True)
                simi = text_feats_self @ text_feats_other.T
                compare_idx = simi.argsort(dim=-1).flip(dims=[-1])[:self.opts.num_compare]
                compare_idx = self.train_index[compare_idx].tolist()

                if not target in compare_idx:
                    compare_idx.append(target)
                compare_idx = torch.tensor(compare_idx).to(self.device)
                targets = (compare_idx == target).nonzero()[0].repeat(batch_size)

                torch.cuda.empty_cache()
                gc.collect()

            return compare_idx, targets

        elif method == 'topk':
            low = min(self.d2n.keys())
            high = max(self.d2n.keys())

            if depth - self.opts.k > low:
                low = depth - self.opts.k
            if depth + self.opts.k < high:
                high = depth + self.opts.k

            candi = []
            for d in range(low, depth):
                candi.extend(self.d2n[d])
            if depth==0:
                candi.extend(self.d2n[depth])
            candi = set(candi)
            compare_idx = candi - set(parents)
            compare_idx = list(compare_idx)
            if len(compare_idx) > self.opts.num_compare:
                compare_idx = random.sample(compare_idx, self.opts.num_compare)
            if not target in compare_idx:
                compare_idx.append(target)
            compare_idx = torch.tensor(compare_idx).to(self.device)

            targets = (compare_idx == target).nonzero()[0].repeat(batch_size)

            return compare_idx, targets
        
        elif method == 'near_simi':
            low = min(self.d2n.keys())
            high = max(self.d2n.keys())

            if depth - self.opts.k > low:
                low = depth - self.opts.k
            if depth + self.opts.k < high:
                high = depth + self.opts.k

            self_name = self.node_tokens[target].unsqueeze(0)
            text_feats_self = self.clip_model.encode_text(self_name)
            text_feats_self /= text_feats_self.norm(dim=-1, keepdim=True)

            candi = []
            for d in range(low, high + 1):
                candi.extend(self.d2n[d])
            candi = set(candi)
            children=self.p2c[target]
            index = list(candi - set(parents)-set(children))
            index=torch.tensor(index)
            num_compare = min(self.opts.num_compare, len(index))

            other = self.node_tokens[index]
            text_feats_other = self.clip_model.encode_text(other)
            text_feats_other /= text_feats_other.norm(dim=-1, keepdim=True)

            simi = text_feats_self @ text_feats_other.T
            compare_idx = simi.argsort(dim=-1).flip(dims=[-1])[:num_compare]

            compare_idx = index[compare_idx].tolist()

            if not target in compare_idx:
                compare_idx.append(target)
            compare_idx = torch.tensor(compare_idx).to(self.device)
            targets = (compare_idx == target).nonzero()[0].repeat(batch_size)
            return compare_idx, targets

        elif method == 'brothers':

            if len(parents) > 1 and depth > 0:
                parent = parents[depth-1]
                compare_idx = copy.copy(self.p2c[parent])
            else:
                compare_idx = copy.copy(self.start_up)

            if len(compare_idx) > self.opts.num_compare:
                compare_idx = random.sample(compare_idx, self.opts.num_compare)
            if not target in compare_idx:
                compare_idx.append(target)

            compare_idx = torch.tensor(compare_idx).to(self.device)
            targets = (compare_idx == target).nonzero()[0].repeat(batch_size)

            return compare_idx, targets

    def get_weights(self, method, max_depth=None):
        if method == "equal":
            weights = (torch.ones(max_depth) / max_depth).to(self.device)
        elif method == "decreasing":
            weights = torch.arange(start=max_depth, end=0, step=-1).to(self.device)
            weights = weights / weights.sum()
        elif method == "increasing":
            weights = torch.arange(start=1, end=max_depth + 1).to(self.device)
            weights = weights / weights.sum()
        elif method == 'adaptive':
            weights = F.softmax(100**self.layer_weight[:max_depth], dim=0)
            # num_layer=[len(self.d2n[layer]) for layer in self.d2n.keys()]
            # weights=1.0/torch.tensor(num_layer)[:max_depth]
            # weights= weights / weights.sum()
        elif method == 'nl_increasing':
            weights = (torch.arange(start=1, end=max_depth + 1) ** 3).to(self.device)
            weights = weights / weights.sum()
        elif method == 'nl_decreasing':
            weights = (torch.arange(start=max_depth, end=0, step=-1) ** 3).to(self.device)
            weights = weights / weights.sum()

        return weights


    def train_batch(self, inputs, targets, training_method, sample_strategy):
        if training_method == "OM":
            img_feats = self.clip_model.encode_image(inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            img_feats_ = img_feats.detach().clone().requires_grad_(True)

            target = targets[0].item()
            parents = copy.copy(self.c2p[target])
            parents.append(target)

            p_loop_out = copy.copy(parents)
            len_out = len(p_loop_out)
            k = math.ceil(self.opts.out_ratio * len_out)
            if k == 0:
                k += 1
            p_loop_out = p_loop_out[::-1][:k]

            loss_layer = []

            for k_loop, p_out in enumerate(p_loop_out):

                parents_in = copy.copy(self.c2p[p_out])
                parents_in.append(p_out)

                p_loop_in = copy.copy(parents_in)
                len_in = len(p_loop_in)
                m = math.ceil(self.opts.in_ratio * len_in)
                if m == 0:
                    m += 1
                p_loop_in = p_loop_in[::-1][:m]

                # weights_out = self.get_weights(method=self.opts.weights, max_depth=len(p_loop_out))

                for m_loop, p_in in enumerate(p_loop_in):
                    depth = parents_in.index(p_in)
                    compare_idx, labels = self.get_contra(method=self.opts.sample_strategy, target=p_out,
                                                          batch_size=len(targets), depth=depth,
                                                          parents=parents_in)

                    text_feats = self.clip_model.encode_text(self.node_tokens[compare_idx])
                    text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                    logits = (img_feats_ @ text_feats.t()) * self.clip_model.logit_scale.exp()

                    if self.opts.weighting=='out':
                        weights_in = self.get_weights(method="equal", max_depth=len(p_loop_in))
                        weights_out = self.get_weights(method=self.opts.weights, max_depth=len(p_loop_out))
                    elif self.opts.weighting=='in':
                        weights_in = self.get_weights(method=self.opts.weights, max_depth=len(p_loop_in))
                        weights_out = self.get_weights(method='equal', max_depth=len(p_loop_out))
                    else:
                        weights_in = self.get_weights(method=self.opts.weights, max_depth=len(p_loop_in))
                        weights_out = self.get_weights(method=self.opts.weights, max_depth=len(p_loop_out))

                    loss_j = self.loss(logits, labels) * weights_in[m_loop] * weights_out[k_loop]
                    loss_j.backward()
                    loss_layer.append(loss_j.item())

            loss = sum(loss_layer)
            img_feats.backward(img_feats_.grad)
            return loss

        if training_method == 'hierarchical':
            img_feats = self.clip_model.encode_image(inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            img_feats_ = img_feats.detach().clone().requires_grad_(True)

            # avai = self.start_up
            parents = copy.copy(self.c2p[targets[0].item()])
            parents.append(targets[0].item())
            loss_layer = []

            max_layer = len(parents)

            for j, p in enumerate(parents):
                t_input = targets[0].item()

                compare_idx, labels = self.get_contra(method=self.opts.sample_strategy, target=t_input,
                                                      batch_size=len(targets), depth=j,
                                                      parents=parents)

                text_feats = self.clip_model.encode_text(self.node_tokens[compare_idx])
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
                logits = (img_feats_ @ text_feats.t()) * self.clip_model.logit_scale.exp()
                weights = self.get_weights(method=self.opts.weights, max_depth=max_layer)
                loss_j = self.loss(logits, labels) * weights[j]
                loss_j.backward()
                loss_layer.append(loss_j.item())

                # avai = self.p2c[p]

            loss = sum(loss_layer)

            img_feats.backward(img_feats_.grad)

            return loss

    def update_classifier(self):
        with torch.no_grad():
            text_feats_1 = self.clip_model.encode_text(self.node_tokens[:len(self.nodes)//2])
            text_feats_2 = self.clip_model.encode_text(self.node_tokens[len(self.nodes)//2:])
            text_feats = torch.cat([text_feats_1, text_feats_2])
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        self.zsl_weights = text_feats


    def forward(self, inputs, targets):
        feats = self.clip_model.encode_image(inputs)
        feats /= feats.norm(dim=-1, keepdim=True)
        logits = (feats @ self.zsl_weights.T)

        return logits