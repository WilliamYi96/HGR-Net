import os
import os.path as osp
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import json
import ipdb
import PIL

from utils import listdir_nohidden

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

class Datum:
    """Data instance which defines the basic attributes.
    Args:
        impath (str): image path.
        label (int): class label.
        classname (str): class name.
    """

    def __init__(self, impath="", label=0, classname=""):
        assert isinstance(impath, str)
        assert osp.isfile(impath)



        self._impath = impath
        self._label = label
        self._classname = classname

    @property
    def impath(self):
        return self._impath

    @property
    def label(self):
        return self._label

    @property
    def classname(self):
        return self._classname

class ImageNet(Dataset):
    def __init__(self, split, node_set, candidates=None, resolution=224):
        super(ImageNet, self).__init__()

        self.split = split
        self.resolution = resolution
        self.preprocess = _transform(self.resolution)
        self.node_set = node_set

        if candidates is None:
            self.candidates = self.node_set
        else:
            self.candidates = candidates

        # self.data = self.read_data()
        self.read_data()

    def read_data(self):
        classes = []
        self.data = []
        data = json.load(open('data/{}_split.json'.format(self.split)))
        for cls in self.candidates:
            classes.append(cls)
            label = self.node_set.index(cls)
            for impath in data[cls]:
                self.data.append((impath, label))

        classes = np.unique(classes)
        print('Done reading data, number of classes: {}, images: {}'.format(len(classes), len(self.data)), flush=True)

        # data = json.load(open('data/{}_split.json'.format(self.split)))
        # items = []
        # classes = []
        # for cls in self.candidates:
        #     classes.append(cls)
        #     for impath in data[cls]:
        #         item = Datum(
        #             impath=impath,
        #             label=self.node_set.index(cls),
        #             classname=cls
        #         )
        #         items.append(item)
        #
        # classes = np.unique(classes)
        # print('Done reading data, number of classes: {}, images: {}'.format(len(classes), len(items)))
        # return items

        # folders = listdir_nohidden(os.path.join(self.dataroot, self.split), sort=True)
        # folders = [f for f in folders if (f in self.node_set and f in self.candidates)]
        # items = []
        # classes = []
        #
        # for i, folder in enumerate(folders):
        #     imnames = listdir_nohidden(os.path.join(self.dataroot, self.split, folder))
        #     classname = folder
        #     label = self.node_set.index(classname)
        #     classes.append(label)
        #     for imname in imnames:
        #         impath = os.path.join(self.dataroot, self.split, folder, imname)
        #
        #         # try:
        #         #     img = Image.open(impath)
        #         # except PIL.UnidentifiedImageError:
        #         #     continue
        #
        #         item = Datum(
        #             impath=impath,
        #             label=label,
        #             classname=classname
        #         )
        #         items.append(item)
        #
        #         if self.debug:
        #             break
        #
        # classes = np.unique(classes)
        # print('Done reading data, number of classes: {}, images: {}'.format(len(classes), len(items)))
        # return items

    def __getitem__(self, item):
        item = self.data[item]
        output = {
            "label": item[1],
            "impath": item[0],
        }

        try:
            img = Image.open(item[0]).convert("RGB")
        except PIL.UnidentifiedImageError:
            img = Image.open(self.data[0][0]).convert("RGB")

        img = self.preprocess(img)

        output['img'] = img

        return output

    def __len__(self):
        return len(self.data)

