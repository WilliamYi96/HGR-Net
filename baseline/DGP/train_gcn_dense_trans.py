import argparse
import json
import random
import os.path as osp
import clip
from nltk.corpus import wordnet as wn
import torch
import torch.nn.functional as F
import templates as template

from utils import ensure_path, set_gpu, l2_loss
from models.gcn_dense_att import GCN_Dense_Att


def save_checkpoint(name):
    torch.save(gcn.state_dict(), osp.join(name + '.pth'))
    torch.save(pred_obj, osp.join(name + '.pred'))


def mask_l2_loss(a, b, mask):
    return l2_loss(a[mask], b[mask])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--max-epoch', type=int, default=3000)
    parser.add_argument('--trainval', default='10,0')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight-decay', type=float, default=0.0005)
    parser.add_argument('--save_epoch', type=int, default=1000)
    parser.add_argument('--save_path', default='result/tran')

    parser.add_argument('--gpu', default='0')

    parser.add_argument('--no-pred', action='store_true')
    args = parser.parse_args()

    set_gpu(args.gpu)

    save_path = args.save_path
    # ensure_path(save_path)

    graph = json.load(open('materials/imagenet-dense-grouped-graph_1.json', 'r'))
    wnids = graph['wnids']
    n = len(wnids) # 32324
    print(n)

    edges_set = graph['edges_set']
    print('edges_set', [len(l) for l in edges_set])

    lim = 4
    for i in range(lim + 1, len(edges_set)):
        edges_set[lim].extend(edges_set[i])
    edges_set = edges_set[:lim + 1]
    print('edges_set', [len(l) for l in edges_set])
    
    fcfile = json.load(open('materials/fc-weights.json', 'r'))
    test_sets = json.load(open('data/process_results/splits_for_tree.json', 'r'))
    train_wnids = test_sets['train']
    test_wnids = test_sets['rest']
    # train_wnids = [x[0] for x in fcfile]
    fc_vector = [x[1] for x in fcfile]
    fc_vectors=[]
    for train_wnid in train_wnids:
        index=wnids.index(train_wnid)
        fc_vectors.append(fc_vector[index])
    fc_vectors = torch.tensor(fc_vectors).cuda()
    fc_vectors = F.normalize(fc_vectors) # [1000,2049]

    text_feats=json.load(open('text_feats_1.json','r'))
    pred_vectors=torch.cat([torch.tensor(feat) for feat in text_feats],0)
    ones=torch.ones((18278,1))
    word_vectors=torch.cat((pred_vectors,pred_vectors,ones),1)
    # word_vectors = text_feats / text_feats.norm(dim=-1, keepdim=True)

    # word_vectors = torch.tensor(graph['vectors']).cuda() # [32324, 300]
    word_vectors = F.normalize(word_vectors).cuda()

    hidden_layers = 'd2048,d'
    gcn = GCN_Dense_Att(n, edges_set,
                        word_vectors.shape[1], fc_vectors.shape[1], hidden_layers).cuda()

    print('word vectors:', word_vectors.shape)
    print('fc vectors:', fc_vectors.shape)
    print('hidden layers:', hidden_layers)

    optimizer = torch.optim.Adam(gcn.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # optimizer2= torch.optim.AdamW(clip_model.parameters(), lr=3e-7, weight_decay=1e-4)

    v_train, v_val = map(float, args.trainval.split(','))
    n_trainval = len(fc_vectors)
    n_train = round(n_trainval * (v_train / (v_train + v_val)))
    print('num train: {}, num val: {}'.format(n_train, n_trainval - n_train),flush=True)
    tlist = list(range(len(fc_vectors)))
    random.shuffle(tlist)

    min_loss = 1e18

    trlog = {}
    trlog['train_loss'] = []
    trlog['val_loss'] = []
    trlog['min_loss'] = 0

    for epoch in range(1, args.max_epoch + 1):
        gcn.train()
        output_vectors = gcn(word_vectors)
        loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        gcn.eval()
        output_vectors = gcn(word_vectors)
        train_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[:n_train]).item()
        if v_val > 0:
            val_loss = mask_l2_loss(output_vectors, fc_vectors, tlist[n_train:]).item()
            loss = val_loss
        else:
            val_loss = 0
            loss = train_loss
        print('epoch {}, train_loss={:.4f}, val_loss={:.4f}'.format(epoch, train_loss, val_loss),flush=True)

        trlog['train_loss'].append(train_loss)
        trlog['val_loss'].append(val_loss)
        trlog['min_loss'] = min_loss
        torch.save(trlog, osp.join(save_path, 'trlog.txt'))

        if (epoch % args.save_epoch == 0):
            if args.no_pred:
                pred_obj = None
            else:
                pred_obj = {
                    'wnids': wnids,
                    'pred': output_vectors
                }

        if epoch % args.save_epoch == 0:
            save_checkpoint('epoch_{}'.format(epoch))

        pred_obj = None

