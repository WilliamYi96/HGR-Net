import json
import sys
sys.path.append('../')
from utils import listdir_nohidden
from collections import defaultdict
import random
import ipdb
import os

root_1k = "/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg"
root_21k = "/ibex/ai/reference/CV/ILSVR/2021/data/raw/train"

classes = json.load(open('process_results/splits_for_tree.json', 'r'))

ls_train = defaultdict(list)
ls_test = defaultdict(list)
ls_val = defaultdict(list)

len_train=0
len_val=0
len_test=0

# 1k train/val
print('1k')
folders = listdir_nohidden(os.path.join(root_1k, "train"), sort=True)
folders = [f for f in folders if f in classes['train']]
for f in folders:

    imnames_train = listdir_nohidden(os.path.join(root_1k, "train", f))
    imnames_val = listdir_nohidden(os.path.join(root_1k, "val", f))

    imnames_train = [os.path.join(root_1k, "train", f, name) for name in imnames_train]
    imnames_val = [os.path.join(root_1k, "val", f, name) for name in imnames_val]

    ls_train[f] = imnames_train
    ls_val[f]=imnames_val
    ls_test[f] = imnames_val

    len_train+=len(imnames_train)
    len_val+=len(imnames_val)
    len_test+=len(imnames_val)

print('seen classes, train: {},val: {},test: {}'.format(len_train,len_val,len_test))
len_train=0
len_val=0
len_test=0

# rest train/val
print('rest')
folders = listdir_nohidden(root_21k, sort=True)
folders = [f for f in folders if f in classes['rest']]
for f in folders:
    imnames = listdir_nohidden(os.path.join(root_21k, f))
    imnames = [os.path.join(root_21k, f, name) for name in imnames]
    num_samples = len(imnames)
    if num_samples >= 10:
        imnames_train = random.sample(imnames, 10)
    else:
        imnames_train = imnames
    rest=list(set(imnames)-set(imnames_train))
    num_samples = len(rest)
    if num_samples > 50:
        imnames_val = random.sample(rest, 50)
    else:
        imnames_val = rest

    # if num_samples < 10:
    #     print('{} contains less than 10 files'.format(f))

    assert len(imnames_train) + len(rest) == len(imnames)

    ls_train[f] = imnames_train
    ls_val[f] = imnames_val
    ls_test[f] = rest

    len_train+=len(imnames_train)
    len_val+=len(imnames_val)
    len_test+=len(rest)
    print(len(rest))

print('unseen classes, train: {},val: {},test: {}'.format(len_train,len_val,len_test))

json.dump(ls_train, open('ls_train_split.json', 'w'))
json.dump(ls_val, open('ls_val_split.json', 'w'))
json.dump(ls_test, open('ls_test_split.json', 'w'))

'''
seen classes, train: 1259303,val: 49150,test: 49150
unseen classes, train: 166566,val: 776438,test: 11171023
'''
