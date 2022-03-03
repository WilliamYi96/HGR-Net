import json
import sys
sys.path.append('../')
from utils import listdir_nohidden
from collections import defaultdict
import random
import ipdb
import os
import torch

root_1k = "/ibex/ai/reference/CV/ILSVR/classification-localization/data/jpeg"
root_21k = "/ibex/ai/reference/CV/ILSVR/2021/data/raw/train"

imagenet21kp = torch.load("official/imagenet21k_miil_tree.pth")['class_list']
classes = json.load(open('process_results/splits_for_tree.json', 'r'))
train_21kp=list(set(classes['train']).intersection(set(imagenet21kp))) # 975
rest_21kp=list(set(classes['rest']).intersection(set(imagenet21kp))) # 9046

target = {}
target['train'] = train_21kp
target['rest'] = rest_21kp
target['all'] = train_21kp+rest_21kp


json.dump(target, open('process_results/imagenet21kp_split.json', 'w'))

train = defaultdict(list)
val = defaultdict(list)
zsl_test = defaultdict(list)

len_train=0
len_val=0
len_test=0

# 1k train/val
print('1k')
folders = listdir_nohidden(os.path.join(root_1k, "train"), sort=True)
folders = [f for f in folders if f in train_21kp]
for f in folders:
    # print(f)

    imnames_train = listdir_nohidden(os.path.join(root_1k, "train", f))
    imnames_val = listdir_nohidden(os.path.join(root_1k, "val", f))

    imnames_train = [os.path.join(root_1k, "train", f, name) for name in imnames_train]
    imnames_val = [os.path.join(root_1k, "val", f, name) for name in imnames_val]

    train[f] = imnames_train
    val[f] = imnames_val
    zsl_test[f] = imnames_val

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
folders = [f for f in folders if f in rest_21kp]
for f in folders:
#     print(f)
    imnames = listdir_nohidden(os.path.join(root_21k, f))
    imnames = [os.path.join(root_21k, f, name) for name in imnames]
    num_samples = len(imnames)
    if num_samples > 50:
        imnames_val = random.sample(imnames, 50)
    else:
        imnames_val = imnames
    imnames_train = []
    for item in imnames:
        if item not in imnames_val:
            imnames_train.append(item)

    if num_samples <= 50:
        print('{} contains less than 50 files'.format(f))

    assert len(imnames_val) + len(imnames_train) == len(imnames)

    train[f] = imnames_train
    val[f] = imnames_val
    zsl_test[f] = imnames

    len_train+=len(imnames_train)
    len_val+=len(imnames_val)
    len_test+=len(imnames)

print('unseen classes, train: {},val: {},test: {}'.format(len_train,len_val,len_test))

json.dump(train, open('21kp_train_split.json', 'w'))
json.dump(val, open('21kp_val_split.json', 'w'))
json.dump(zsl_test, open('21kp_test_split.json', 'w'))

'''
1k
seen classes 975, train: 1252157,val: 48750,test: 48750
rest
unseen classes 9046, train: 9394816,val: 452300,test: 9847116
'''





