#######################
#author: Shiming Chen
#FREE
#######################
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import numpy as np
import util
from sklearn.preprocessing import MinMaxScaler 
import sys
import copy
import pdb
from sklearn.decomposition import PCA
from config import opt
import random

from sklearn.neighbors import KNeighborsClassifier

def generate_syn_feature(generator,classes, attribute,num=100):
    nclass = classes.size(0)
    syn_feature = torch.FloatTensor(nclass*num, opt.resSize)
    syn_label = torch.LongTensor(nclass*num) 
    syn_att = torch.FloatTensor(num, opt.attSize)
    syn_noise = torch.FloatTensor(num, opt.nz)
    if opt.cuda:
        syn_att = syn_att.cuda()
        syn_noise = syn_noise.cuda()
    for i in range(nclass):
        iclass = classes[i]
        iclass_att = attribute[iclass]
        syn_att.copy_(iclass_att.repeat(num, 1))
        syn_noise.normal_(0, 1)
        with torch.no_grad():
            syn_noisev = Variable(syn_noise)
            syn_attv = Variable(syn_att)
        fake = generator(syn_noisev,c=syn_attv)
        output = fake.cuda()
        syn_feature.narrow(0, i*num, num).copy_(output.data.cpu())
        syn_label.narrow(0, i*num, num).fill_(iclass)

    return syn_feature, syn_label

class CLASSIFIER:
    # train_Y is interger 
    def __init__(self, netG, feature_encoder,p2c,c2p,d2n,ordered_nodes, test_index, attribute, data_loader, _nclass, _cuda, _lr=0.001, _beta1=0.5, _nepoch=20, _batch_size=1512, netFR=None, dec_size=4096, dec_hidden_size=4096):
        self.feature_encoder=feature_encoder.cuda()
        self.p2c,self.c2p,self.d2n,self.ordered_nodes = p2c,c2p,d2n,ordered_nodes
        # self.test_seen_feature = data_loader.test_seen_feature.clone()
        # self.test_seen_label = data_loader.test_seen_label 
        # self.test_unseen_feature = data_loader.test_unseen_feature.clone()
        # self.test_unseen_label = data_loader.test_unseen_label 
        # self.seenclasses = data_loader.seenclasses
        # self.unseenclasses = data_loader.unseenclasses
        self.test_index = test_index
        self.attribute = attribute
        self.data_loader=data_loader
        self.netG=netG.cuda()
        self.batch_size = _batch_size
        self.nepoch = _nepoch
        self.nclass = _nclass
        self.input_dim = 2048
        self.cuda = _cuda
        self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass)
        self.netFR = netFR
        if netFR is not None:
            self.netFR = netFR.cuda()
        if self.netFR:
            self.netFR.eval()
            # self.input_dim = self.input_dim
            # self.input_dim = self.input_dim + dec_hidden_size
            self.input_dim = self.input_dim + dec_hidden_size + dec_size
            self.model =  LINEAR_LOGSOFTMAX_CLASSIFIER(self.input_dim, self.nclass).cuda()
            # self.train_X = self.compute_fear_out(self.train_X, self.input_dim)
            
            # self.test_unseen_feature = self.compute_fear_out(self.test_unseen_feature, self.input_dim)
            # self.test_seen_feature = self.compute_fear_out(self.test_seen_feature, self.input_dim)

        self.model.apply(util.weights_init)
        self.criterion = nn.NLLLoss()
        self.input = torch.FloatTensor(_batch_size, self.input_dim) 
        self.label = torch.LongTensor(_batch_size) 
        self.lr = _lr
        self.beta1 = _beta1
        self.optimizer = optim.Adam(self.model.parameters(), lr=_lr, betas=(_beta1, 0.999))
        if self.cuda:
            self.model.cuda()
            self.criterion.cuda()
            self.input = self.input.cuda()
            self.label = self.label.cuda()
        self.index_in_epoch = 0
        self.epochs_completed = 0
        self.syn_feature, self.syn_label = generate_syn_feature(self.netG, self.test_index, self.attribute)
        self.ntrain = self.syn_feature.size()[0]
        # self.best_acc, self.best_model = self.fit_zsl()
            
    def fit_zsl(self):
        best_acc = 0
        mean_loss = 0
        last_loss_epoch = 1e8
        for epoch in range(self.nepoch):
            print(epoch,flush=True)
            for i, data in enumerate(self.data_loader):
                if i<len(self.data_loader)-1:
                    imgs, input_label = data['img'].cuda(), data['label'].cuda()
                    with torch.no_grad():
                        feats = self.feature_encoder(imgs).view(-1,2048).cuda()
                    self.model.zero_grad()
                    batch_input, batch_label = self.next_batch(feats,input_label, syn_batch_size = 1000)
                    self.input.copy_(batch_input)
                    self.label.copy_(batch_label)
                    
                    inputv = Variable(self.input).cuda()
                    labelv = Variable(self.label).cuda()
                    output = self.model(inputv)
                    loss = self.criterion(output, labelv)
                    #mean_loss += loss.data[0]
                    mean_loss += loss.item()
                    loss.backward()
                    self.optimizer.step()
                    #print('Training classifier loss= ', loss.data[0])
                       
    def next_batch(self, train_feature, train_label, syn_batch_size=1000):
        start = self.index_in_epoch
        # shuffle the data at the first epoch
        if self.epochs_completed == 0 and start == 0:
            perm = torch.randperm(self.ntrain)
            self.syn_feature = self.syn_feature[perm]
            self.syn_label = self.syn_label[perm]
        # the last batch
        if start + syn_batch_size > self.ntrain:
            self.epochs_completed += 1
            rest_num_examples = self.ntrain - start
            if rest_num_examples > 0:
                X_rest_part = self.syn_feature[start:self.ntrain]
                Y_rest_part = self.syn_label[start:self.ntrain]
            # shuffle the data
            perm = torch.randperm(self.ntrain)
            self.syn_feature = self.syn_feature[perm]
            self.syn_label = self.syn_label[perm]
            # start next epoch
            start = 0
            self.index_in_epoch = syn_batch_size - rest_num_examples
            end = self.index_in_epoch
            X_new_part = self.syn_feature[start:end]
            Y_new_part = self.syn_label[start:end]
            #print(start, end)
            if rest_num_examples > 0:
                syn_feature ,syn_label = torch.cat((X_rest_part, X_new_part), 0) , torch.cat((Y_rest_part, Y_new_part), 0)
            else:
                syn_feature ,syn_label = X_new_part, Y_new_part
        else:
            self.index_in_epoch += syn_batch_size
            end = self.index_in_epoch
            #print(start, end)
            # from index start to index end-1
            syn_feature ,syn_label = self.syn_feature[start:end], self.syn_label[start:end]
        # syn_feature, syn_label = generate_syn_feature(self.netG, self.test_index, self.attribute, syn_batch_size)
        syn_feature = syn_feature.cuda()
        syn_label = syn_label.cuda()
        train_x = torch.cat((train_feature, syn_feature), 0)
        train_Y = torch.cat((train_label, syn_label), 0)
        train_X = self.compute_fear_out(train_x, self.input_dim)
        return train_X,train_Y




    def val_gzsl(self, test_X, test_label, target_classes): 
        start = 0
        ntest = test_X.size()[0]
        predicted_label = torch.LongTensor(test_label.size())
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])
            output = self.model(inputX)
            _, predicted_label[start:end] = torch.max(output.data, 1)
            start = end

        acc = self.compute_per_class_acc_gzsl(test_label, predicted_label, target_classes)
        return acc

    def compute_per_class_acc_gzsl(self, test_label, predicted_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (test_label == i)
            acc_per_class += torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx)
        acc_per_class /= target_classes.size(0)
        return acc_per_class
    
    def count_acc(self,hits_dict, num_tot):
        out_str = ""
        for key, value in hits_dict.items():
            acc = value / num_tot * 100.0
            out_str += "Top@{}(%):{:.2f}".format(key, acc)
            if key != list(hits_dict.keys())[-1]:
                out_str += ", "
            else:
                out_str += '.'

        return out_str

    def test(self,test_dataloader):
        topk = (1, 2, 5, 10, 20)
        hits_dict = dict(zip(topk, [0]*len(topk)))
        maxk = max(topk)
        num_sample=0
        hits_all=0
        len_parents_all=0
        path_all=0
        point_all=0
        path_all_count=0
        with torch.no_grad():
            for i, data in enumerate(test_dataloader):
                imgs, targets = data['img'][0].cuda(), data['label'][0].cuda()
                feats = self.feature_encoder(imgs).view(-1,2048).cuda()
                _,_,_,_, _, feat2 = self.netFR(feats)
                feat1 = self.netFR.getLayersOutDet()
                new_test_X = torch.cat([feats,feat1,feat2],dim=1).data.cuda()
                logits = self.model(new_test_X)
                logits_ = logits[:, self.test_index]
                # print(logits_.shape,flush=True)
                _, pred = logits_.topk(maxk, 1, True, True)
                pred = self.test_index[pred]
                pred = pred.t().cuda()
                correct = pred.eq(targets.reshape(1, -1).expand_as(pred))
                for k in topk:
                    num_correct_k = correct[:k].reshape(-1).float().sum()
                    hits_dict[k] += num_correct_k
                num_sample += len(targets)
                path_all_count += len(targets)

                target = targets[0].item()
                parents = copy.copy(self.c2p[target])
                parents.append(target)
                parent=torch.tensor(parents).expand(len(targets),len(parents))
                _, pred = logits_.topk(1, 1, True, True)
                pred = self.test_index[pred].cpu()
                pred = pred.expand(len(targets),len(parents))
                correct=pred.eq(parent).reshape(-1).float().sum()
                hits_all += correct

                dict_path = torch.zeros(len(targets), len(parents))
                for k, p in enumerate(parents):
                    level = len(self.c2p[p])
                    same_l = copy.copy(self.d2n[level])
                    if p not in same_l:
                        same_l.append(p)
                        print("{} not in its level".format(p))
                    rest = torch.tensor(list(set(list(range(len(self.ordered_nodes)))) - set(same_l))).cuda()
                    logit_k = logits.detach().clone()
                    logit_k = logit_k.index_fill(1, rest, -1)
                    logit_k = logit_k[:, self.test_index]
                    _, pred = logit_k.topk(1, 1, True, True)
                    pred = self.test_index[pred]
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

                if i % 1 == 0:
                    out_str = "\n"
                    tmp_str = self.count_acc(hits_dict, num_sample)
                    out_str += tmp_str
                    hit_ratio = hits_all / num_sample * 100.0
                    out_str += ' hit_ratio(%):{:.2f}'.format(hit_ratio)
                    path_ratio = path_all / path_all_count * 100.0
                    out_str += ' path_ratio(%):{:.2f}'.format(path_ratio)
                    point_ratio = point_all / num_sample * 100.0
                    out_str += ' point_ratio(%):{:.2f}'.format(point_ratio)
                    print(out_str, flush=True)
                    with open('log.txt', 'a') as f:
                        f.writelines(out_str + '\n')
            print('End of testing.')
            out_str = "\n"
            tmp_str = self.count_acc(hits_dict, num_sample)
            out_str += tmp_str
            hit_ratio = hits_all / num_sample * 100.0
            out_str += ' hit_ratio(%):{:.2f}'.format(hit_ratio)
            path_ratio = path_all / path_all_count * 100.0
            out_str += ' path_ratio(%):{:.2f}'.format(path_ratio)
            point_ratio = point_all / num_sample * 100.0
            out_str += ' point_ratio(%):{:.2f}'.format(point_ratio)
            print(out_str, flush=True)


    

    def compute_per_class_acc(self, test_label, predicted_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            acc_per_class[i] = torch.sum(test_label[idx]==predicted_label[idx]).float() / torch.sum(idx)
        return acc_per_class.mean() 


    def compute_fear_out(self, test_X, new_size):
        start = 0
        ntest = test_X.size()[0]
        new_test_X = torch.zeros(ntest,new_size)
        for i in range(0, ntest, self.batch_size):
            end = min(ntest, start+self.batch_size)
            if self.cuda:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end].cuda())
            else:
                with torch.no_grad():
                    inputX = Variable(test_X[start:end])
            _,_,_,_, _, feat2 = self.netFR(inputX)
            feat1 = self.netFR.getLayersOutDet()
            # new_test_X[start:end] = inputX.data.cpu()
            # new_test_X[start:end] = torch.cat([inputX,feat1],dim=1).data.cpu()
            new_test_X[start:end] = torch.cat([inputX,feat1,feat2],dim=1).data.cpu()
            
            start = end
            # pca = PCA(n_components=4096,whiten=False)
            # fit = pca.fit(new_test_X)
            # features = pca.fit_transform(new_test_X)
            # features = torch.from_numpy(features)
            # fnorm = torch.norm(features, p=2, dim=1, keepdim=True)
            # new_test_X = features.div(fnorm.expand_as(features))
        return new_test_X

    def compute_per_class_acc_gzsl_knn( self,  predicted_label, test_label, target_classes):
        acc_per_class = 0
        for i in target_classes:
            idx = (predicted_label == i)
            if torch.sum(idx)==0:
                acc_per_class +=0
            else:
                acc_per_class += float(torch.sum(predicted_label[idx] == test_label[idx])) / float(torch.sum(idx))
        acc_per_class /= float(target_classes.size(0))
        return acc_per_class
        
    def compute_per_class_acc_knn(self, predicted_label, test_label, nclass):
        acc_per_class = torch.FloatTensor(nclass).fill_(0)
        for i in range(nclass):
            idx = (test_label == i)
            if torch.sum(idx)==0:
                acc_per_class +=0
            else:
                acc_per_class += torch.sum(predicted_label[idx]==test_label[idx]).float() / torch.sum(idx)
        acc_per_class /= float(nclass)
        return acc_per_class.mean() 
    
class LINEAR_LOGSOFTMAX_CLASSIFIER(nn.Module):
    def __init__(self, input_dim, nclass):
        super(LINEAR_LOGSOFTMAX_CLASSIFIER, self).__init__()
        self.fc = nn.Linear(input_dim, nclass)
        self.logic = nn.LogSoftmax(dim=1)
    def forward(self, x): 
        o = self.logic(self.fc(x))
        return o