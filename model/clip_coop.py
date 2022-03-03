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
import model.CoOp as CoOp
import numpy as np

class CustomCLIP(nn.Module):
    def __init__(self, classnames, clip_model):
        super().__init__()
        self.prompt_learner = CoOp.PromptLearner(classnames, clip_model)
        self.tokenized_prompts = self.prompt_learner.tokenized_prompts
        self.token_embedding=clip_model.token_embedding
        self.image_encoder = clip_model.visual
        self.text_encoder = CoOp.TextEncoder(clip_model)
        self.logit_scale = clip_model.logit_scale
        self.dtype = clip_model.dtype
        self.prompt_learner.ctx_dim = clip_model.ln_final.weight.shape[0]



class tree_coop(nn.Module):
    def __init__(self, opts, candidates_train, candidates_test):
        super(tree_coop, self).__init__()
        self.opts = opts
        self.device = opts.device
        self.save_path = '{}/{}/{}_{}_{}/'.format(opts.folder, opts.exp_name, opts.weights, opts.out_ratio, opts.in_ratio)
        self.file_path=self.save_path+'clip_{}'.format(opts.from_epoch)
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

        # Semantic structure
        self.p2c, self.c2p, self.d2n, self.nodes, self.start_up = gen_tree(self.opts)

        # Get tokens
        # self.template = getattr(template, 'TEMPLATES_SIMPLE')[0]
        self.node_name = []
        for node in self.nodes:
            synset = get_synsets(node)
            name = synset.name().split('.')[0].replace('_', ' ')
        #     name = self.template.format(name)
            self.node_name.append(name)

        clip_model, _ = clip.load(name=opts.arch, device=self.device, download_root='pretrained')
        if opts.fetch:
            clip_model.load_state_dict(torch.load(opts.fetch_path))
            print(opts.fetch_path)
            print("successfully fetch")
        
        self.clip_model = CustomCLIP(self.node_name,clip_model)
        del clip_model

        if opts.load:
            self.clip_model.load_state_dict(torch.load(opts.load_path),strict=False)
            print(opts.load_path)
            print("successfully loaded")

        self.clip_model.eval()
        for name, param in self.clip_model.named_parameters():
            if "prompt_learner" in name:
                param.requires_grad_(False)
            else:
                param.requires_grad_(True)
        self.clip_model.to(self.device)
        self.loss = nn.CrossEntropyLoss()

        # Misc
        self.resolution = 224
        self.train_index = torch.tensor([self.nodes.index(item) for item in candidates_train]).to(self.device)
        self.test_index = torch.tensor([self.nodes.index(item) for item in candidates_test]).to(self.device)
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
        if method == 'topk':
            low = min(self.d2n.keys())
            high = max(self.d2n.keys())

            if depth - self.opts.k > low:
                low = depth - self.opts.k
            if depth + self.opts.k < high:
                high = depth + self.opts.k

            candi = []
            for d in range(low, high+1):
                candi.extend(self.d2n[d])
            candi = set(candi)
            compare_idx = candi - set(parents)
            if self.opts.exclu_bro:
                if len(parents) > 1 and depth > 0:
                    parent = parents[depth-1]
                    brothers = set(self.p2c[parent]) - set([target])
                    compare_idx = compare_idx - brothers
            compare_idx = list(compare_idx)
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
            img_feats = self.clip_model.image_encoder(inputs)
            img_feats = img_feats / img_feats.norm(dim=-1, keepdim=True)
            img_feats_ = img_feats.detach().clone().requires_grad_(True)
                # img_feats_ = img_feats.detach()

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
                    if p_in not in compare_idx:
                        compare_idx.append(p_in)
                    compare_idx = torch.tensor(compare_idx).to(self.device)

                    labels = (compare_idx == p_in).nonzero()[0].repeat(len(targets))
                    compare_idx, labels = self.get_contra(method=self.opts.sample_strategy, target=p_out,
                                                          batch_size=len(targets), depth=depth,
                                                          parents=parents_in)

                    text_feats = self.clip_model.text_encoder(self.clip_model.prompt_learner(compare_idx), self.clip_model.prompt_learner.tokenized_prompts[compare_idx])
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

    def update_classifier(self):
        with torch.no_grad():
            index=np.arange(0,10000)
            text_feats_1 = self.clip_model.text_encoder(self.clip_model.prompt_learner(index), self.clip_model.prompt_learner.tokenized_prompts[:10000])
            index=np.arange(10000,18278)
            text_feats_2 = self.clip_model.text_encoder(self.clip_model.prompt_learner(index), self.clip_model.prompt_learner.tokenized_prompts[10000:])
            text_feats = torch.cat([text_feats_1, text_feats_2])
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

        self.zsl_weights = text_feats


    def forward(self, inputs, targets):
        feats = self.clip_model.image_encoder(inputs)
        feats /= feats.norm(dim=-1, keepdim=True)

        logits = (feats @ self.zsl_weights.T)

        return logits