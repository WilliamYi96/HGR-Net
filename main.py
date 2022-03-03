import torch
import numpy as np
import argparse
from model import tree_model
from dataset import ImageNet, DataManager, DataManager_test
from torch.utils.data import DataLoader
import ipdb
from utils import cosine_lr, convert_models_to_fp32, convert_weights, accuracy, count_acc
import clip
import json
import gc
import copy

parser = argparse.ArgumentParser(description='HGR')

parser.add_argument('--exp_name', default='HGR', type=str)
parser.add_argument('--folder', default='adaptive', type=str)
parser.add_argument('--device', default=0, type=int)
parser.add_argument('--print_freq', default=1000, type=int)
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--test_after_train', default=False, action="store_true")

# Model
parser.add_argument('--arch', default='RN50', type=str)

# imagenet
parser.add_argument('--template', default='TEMPLATES_STANDARD', type=str)
parser.add_argument('--model_train', default='all', type=str)
parser.add_argument('--model_test', default='rest', type=str)
parser.add_argument('--data_train', default='train', type=str)
parser.add_argument('--data_test', default='rest', type=str)

# data
parser.add_argument('--graph_path', default='data/process_results/graph_edges_cls.json', type=str)
parser.add_argument('--split_path', default='data/process_results/splits_for_tree.json', type=str)
parser.add_argument('--num_workers', default=12, type=int)
parser.add_argument('--batch_size', default=256, type=int)
parser.add_argument('--test_batch_size', default=512, type=int)
parser.add_argument('--k_shots', default=-1, type=int)
parser.add_argument('--serial_batches', type=eval, default=True, choices=[True, False])
parser.add_argument('--n_episodes', default=-1, type=int)
parser.add_argument('--data_split_train', default='train', type=str, help="train, ls_train")
parser.add_argument('--data_split_test', default='zsl_test', type=str, help="val, ls_test, zsl_test")

# train
parser.add_argument('--open_eval', type=eval, default=True, choices=[True, False])
parser.add_argument('--train', default=True, type=eval, choices=[True, False])
parser.add_argument('--lr', default=3e-7, type=float)
parser.add_argument('--w_lr', default=1e-4, type=float)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('--wd', default=0.0, type=float)
parser.add_argument('--warmup_length', default=0, type=int)
parser.add_argument('--num_compare', default=256, type=int)
parser.add_argument('--weights', default="adaptive", type=str, help="equal, increasing, decreasing, adaptive, nl_increasing, nl_decreasing")
parser.add_argument('--training_method', default='OM', type=str, help="flat, hierarchical, OM")
parser.add_argument('--sample_strategy', default='topk', type=str, help="random, simi, topk, brothers")
parser.add_argument('--k', default=1, type=int)
parser.add_argument('--out_ratio', default=0.25, type=float, help="0.0, 0.25, 0.5, 0.75, 1.0")
parser.add_argument('--in_ratio', default=0.5, type=float, help="0.0, 0.25, 0.5, 0.75, 1.0")
parser.add_argument('--weighting', default="both", type=str, help="in,out")
parser.add_argument('--scale', default=1.0, type=float)

# resume
parser.add_argument('--fetch', default=False, action='store_true')
parser.add_argument('--fetch_path', type=str)
parser.add_argument('--load', default=False, action='store_true')
parser.add_argument('--load_path', default='none', type=str)
parser.add_argument('--from_epoch', default=-1, type=int)

opts = parser.parse_args()

def train(opts, epoch, model, train_loader, num_batches, optimizer, optimizer2, scheduler, device):
    torch.cuda.empty_cache()
    gc.collect()

    if not opts.open_eval:
        model.train()

    for i, data in enumerate(train_loader):

        step = i + epoch * num_batches
        scheduler(step)

        imgs, targets = data['img'][0].to(device), data['label'][0].to(device)

        loss = model.train_batch(imgs, targets, opts.training_method, opts.sample_strategy)
        params = [p for name, p in model.named_parameters() if p.requires_grad and name != "layer_weight"]
        torch.nn.utils.clip_grad_norm_(params, 1.0)

        convert_models_to_fp32(model)
        optimizer.step()
        if opts.weights=='adaptive':
            optimizer2.step()
        convert_weights(model)

        if i % opts.print_freq == 0:
            print('loss: {:.2f}, {}/{}'.format(loss, i, num_batches), flush=True)
            out_str='loss: {:.2f}, {}/{}'.format(loss, i, num_batches)
            log = model.save_path + 'arugements.log'
            with open(log, 'a') as f:
                f.writelines(out_str + '\n')


def test(opts, model, device, splits):
    print('out',opts.out_ratio)
    print('in',opts.in_ratio)
    model.eval()
    model.update_classifier()
    print('Loading datasets', flush=True)

    data = DataManager_test(opts=opts, split=opts.data_split_test, node_set=model.nodes,
                            candidates=splits[opts.data_test], resolution=model.resolution)
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

        log = model.save_path + 'arugements.log'

        for i, data in enumerate(loader_test):

            imgs, targets = data['img'].to(device)[0], data['label'].to(device)[0]  # imgs [batch,3,224,224]

            logits = model(imgs, targets)
            logits_ = logits[:, model.test_index]  # [batch_size,len(test)]
            maxk = max(topk)
            _, pred = logits_.topk(maxk, 1, True, True)
            pred = model.test_index[pred]
            pred = pred.t()
            correct = pred.eq(targets.reshape(1, -1).expand_as(pred))

            logits_ = logits[:, model.train_index] # [batch_size,len(all)]

            for k in topk:
                num_correct_k = correct[:k].reshape(-1).float().sum()
                hits_dict[k] += num_correct_k
            num_sample += len(targets)
            path_all_count += len(targets)


            target = targets[0].item()
            parents = copy.copy(model.c2p[target])
            parents.append(target)
            parent=torch.tensor(parents).expand(len(targets),len(parents))
            _, pred = logits_.topk(1, 1, True, True)
            pred = model.train_index[pred].cpu()
            pred = pred.expand(len(targets),len(parents))
            correct=pred.eq(parent).reshape(-1).float().sum()
            hits_all += correct

            dict_path = torch.zeros(len(targets), len(parents))
            for k, p in enumerate(parents):
                level = len(model.c2p[p])
                same_l = copy.copy(model.d2n[level])
                if p not in same_l:
                    same_l.append(p)
                    print("{} not in its level".format(p))
                rest = torch.tensor(list(set(list(range(len(model.nodes)))) - set(same_l))).cuda()
                logit_k = logits.detach().clone()
                logit_k = logit_k.index_fill(1, rest, -1)
                logit_k = logit_k[:, model.train_index]
                _, pred = logit_k.topk(1, 1, True, True)
                pred = model.train_index[pred]
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

        with open(log, 'a') as f:
            f.writelines(out_str + '\n')
        log_all = '{}.txt'.format(opts.weights)
        with open(log_all, 'a') as f:
            method = '{},{},{}:'.format(opts.weights, opts.out_ratio, opts.in_ratio)
            f.writelines(method + '\n' + out_str + '\n')


def main():
    device = 'cuda:{}'.format(opts.device)
    splits = json.load(open(opts.split_path, 'r'))
    print('Creating models')
    model = tree_model(opts, candidates_train=splits[opts.model_train], candidates_test=splits[opts.model_test])

    if opts.train:
        log = model.save_path + 'arugements.log'
        args = parser.parse_args()
        argsDict = args.__dict__
        with open(log, 'a') as f:
            for eachArg, value in argsDict.items():
                f.writelines(eachArg + ' : ' + str(value) + '\n')
        print('Training.')
        print('Loading datasets')
        data = DataManager(opts=opts, split=opts.data_split_train, node_set=model.nodes,
                           candidates=splits[opts.data_train], resolution=model.resolution)
        loader_train = data.get_data_loader()
        num_batches = data.n_episodes

        print('Creating optimizers')
        params = [p for name, p in model.named_parameters() if p.requires_grad and name != "layer_weight"]
        optimizer = torch.optim.AdamW(params, lr=opts.lr, weight_decay=opts.wd)
        if opts.weights=='adaptive':
            optimizer2 = torch.optim.SGD([model.layer_weight], lr=opts.w_lr)
        else:
            optimizer2 = None

        scheduler = cosine_lr(optimizer, opts.lr, opts.warmup_length, opts.epochs * num_batches)

        print('Running.')
        for epoch in range(opts.from_epoch + 1, opts.epochs):
            train(opts, epoch, model, loader_train, num_batches, optimizer, optimizer2, scheduler, device)
            model.save(opts, epoch)
            print('Model saved.')

            if opts.test_after_train:
                test(opts, model, device, splits)


    else:
        print('Direct testing.')
        test(opts, model, device, splits)


if __name__ == '__main__':
    main()