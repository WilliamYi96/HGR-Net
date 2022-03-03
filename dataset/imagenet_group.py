import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from collections import defaultdict
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import json
import random
import ipdb


from utils import listdir_nohidden
import PIL

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

def _convert_image_to_rgb(image):
    return image.convert("RGB")

def _transform(n_px):
    return Compose([
        Resize(n_px, interpolation=BICUBIC),
        CenterCrop(n_px),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class DataManager(object):
    def __init__(self, opts, split, node_set, candidates=None, resolution=224):
        super(DataManager, self).__init__()

        self.split = split
        self.node_set = node_set
        self.transform = _transform(resolution)

        if candidates is None:
            self.candidates = self.node_set
        else:
            self.candidates = candidates

        self.batch_size = opts.batch_size
        self.serial_batches = opts.serial_batches
        self.k_shots = opts.k_shots
        # if self.k_shots > 0:
        #     assert (self.k_shots % self.batch_size == 0)

        self.data_grouped = self.read_data()

        counts = [len(group) for group in self.data_grouped.values()]
        self.num_data = sum(counts)

        if opts.n_episodes > 0:
            self.n_episodes = opts.n_episodes
        else:
            self.n_episodes = self.num_data // self.batch_size + 1
        

    def read_data(self):
        data_grouped = defaultdict(list)
        data = json.load(open('data/{}_split.json'.format(self.split)))
        num_items = 0
        num_classes = 0
        for cls in self.candidates:
            data_grouped[cls] = data[cls]
            num_items += len(data[cls])
            num_classes += 1

        print('Done reading data, number of classes: {}, images: {}'.format(num_classes, num_items))

        if self.k_shots > 0:
            unseen=json.load(open('/data/process_results/splits_for_tree.json'))['rest']
            num_items_seen = 0
            num_items_unseen = 0
            for cls_label, cls_group in data_grouped.items():
                if cls_label in unseen:
                    if len(cls_group) > self.k_shots:
                        data_grouped[cls_label] = random.sample(cls_group, self.k_shots)
                        num_items_unseen += self.k_shots
                    else:
                        data_grouped[cls_label] = cls_group
                        num_items_unseen += len(cls_group)
                else:
                    num_items_seen += len(cls_group)
            print('Done preparing {}-shot datasets, number of seen images: {}, number of unseen images: {}'.format(self.k_shots,num_items_seen,num_items_unseen))

        return data_grouped


    def get_data_loader(self):

        dataset = ImageDataset(self.data_grouped, self.node_set, self.batch_size, self.transform, self.serial_batches)
        sampler = GroupBatchSampler(self.n_episodes, len(dataset))
        data_loader = DataLoader(
            dataset=dataset,
            batch_sampler=sampler,
            num_workers=12,
            pin_memory=True
        )

        return data_loader


class ImageDataset(object):
    def __init__(self, data_grouped, node_set, batch_size, transform, serial_batches):
        self.data_grouped = data_grouped
        self.batch_size = batch_size
        self.transform = transform
        self.serial_batches = serial_batches
        self.node_set = node_set

        self.group_loaders = []
        group_loader_params = dict(
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=False,
            drop_last=False
        )
        for cls_name, cls_group in self.data_grouped.items():
            if len(cls_group) > 0:
                label = self.node_set.index(cls_name)
                group_dataset = GroupDataset(cls_group, label, transform=self.transform)
                group_loader = DataLoader(dataset=group_dataset, **group_loader_params)
                self.group_loaders.append(iter(group_loader))

    def __getitem__(self, i):

        if self.serial_batches:
            try:
                data = next(self.group_loaders[i])
            except StopIteration:
                data = next(iter(self.group_loaders[i]))
        else:
            data = next(iter(self.group_loaders[i]))

        return data

    def __len__(self):
        return len(self.group_loaders)


class GroupDataset(Dataset):
    def __init__(self, img_paths, label, transform):
        self.img_paths = img_paths
        self.label = label
        self.transform = transform

    def __getitem__(self, i):
        try:
            img = Image.open(self.img_paths[i]).convert("RGB")
        except PIL.UnidentifiedImageError:
            img = Image.open(self.img_paths[0]).convert("RGB")

        img = self.transform(img)

        return {'img': img, 'label': self.label, 'path': self.img_paths[i]}

    def __len__(self):
        return len(self.img_paths)


class GroupBatchSampler(object):
    def __init__(self, n_episodes, n_groups):
        self.n_episodes = n_episodes
        self.n_groups = n_groups

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for i in range(self.n_episodes // self.n_groups + 1):
            seq = list(range(self.n_groups))
            random.shuffle(seq)
            for g in seq:
                yield [g]
