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

train = defaultdict(list)
val = defaultdict(list)
zsl_test = defaultdict(list)

len_train=0
len_val=0
len_test=0

# 1k train/val
print('1k')
folders = listdir_nohidden(os.path.join(root_1k, "train"), sort=True)
folders = [f for f in folders if f in classes['train']]
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
folders = [f for f in folders if f in classes['rest']]

for f in folders:
    # print(f)
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

    assert len(imnames_val) + len(imnames_train) == len(imnames)

    train[f] = imnames_train
    val[f] = imnames_val
    zsl_test[f] = imnames


    len_train+=len(imnames_train)
    len_val+=len(imnames_val)
    len_test+=len(imnames)

print('unseen classes, train: {},val: {},test: {}'.format(len_train,len_val,len_test))

json.dump(train, open('train_split.json', 'w'))
json.dump(val, open('val_split.json', 'w'))
json.dump(zsl_test, open('zsl_test_split.json', 'w'))

'''
seen classes, train: 1259303,val: 49150,test: 49150
unseen classes, train: 10545079,val: 792510,test: 11337589
'''