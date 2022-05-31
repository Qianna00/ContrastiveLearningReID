from __future__ import absolute_import
import os
import os.path as osp
from torch.utils.data import DataLoader, Dataset
import numpy as np
import random
import math
from PIL import Image
import torch


class Preprocessor(Dataset):
    def __init__(self, dataset, root=None, transform=None, camstyle=False, num_cams=6):
        super(Preprocessor, self).__init__()
        self.dataset = dataset
        self.root = root
        self.transform = transform
        self.camstyle = camstyle
        self.num_cams = num_cams

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, indices):
        return self._get_single_item(indices)

    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)
        if self.camstyle:
            sel_cam = self._random_camid(camid+1)
            fpath = fpath.replace('bounding_box_train', 'bounding_box_train_camstyle')[:-4] + '_fake_' + str(camid+1) + 'to' + str(sel_cam) + '.jpg'

        img = Image.open(fpath).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        return img, fname, pid, camid, index

    def _random_camid(self, camid):
        while True:
            rand_camid = torch.randperm(self.num_cams)[0].item() + 1
            if rand_camid != camid:
                return rand_camid


class ContrastivePreprocessor(Preprocessor):
    def _get_single_item(self, index):
        fname, pid, camid = self.dataset[index]
        fpath = fname
        if self.root is not None:
            fpath = osp.join(self.root, fname)

        img = Image.open(fpath).convert('RGB')

        assert self.transform is not None

        img1 = self.transform(img)
        img2 = self.transform(img)

        return img1, img2, fname, pid, camid, index