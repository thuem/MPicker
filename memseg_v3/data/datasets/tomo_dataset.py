# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Simple dataset class that wraps a list of path names
"""
import numpy as np
import torch
from .abstract import AbstractDataset
import time
from .utils import get_patch_crop_coords

import pdb


class TomogramTrain(AbstractDataset):
    def __init__(self, image_lists, transforms=None, is_label=False):
        self.image_lists = image_lists
        self.transforms = transforms
        self.is_label = is_label
        # self.co_train = co_train

    def __getitem__(self, item):
        # start_time = time.time()
        img_path = self.image_lists[item]
        img = np.load(img_path)

        if self.is_label:
            seg_path = img_path.replace('img', 'seg')
            box_path = img_path.replace('img', 'box')
            class_path = img_path.replace('img', 'class')

            seg = np.load(seg_path)
            box_coord = np.load(box_path)
            box_class = np.load(class_path) + 1
        else:
            seg = np.int16(np.zeros(img.shape))
            box_coord = np.array([])
            box_class = np.array([])

        # read_time = time.time()
        if self.transforms is not None:
            img, seg, box_coord, box_class = self.transforms(img, seg, box_coord, box_class)
            return img, seg, box_coord, box_class
            # if self.co_train:
            #     img_weak, img, seg, box_coord, box_class = self.transforms(img, seg, box_coord, box_class)
            #     return img_weak, img, seg, box_coord, box_class
            # else:
            #     img, seg, box_coord, box_class = self.transforms(img, seg, box_coord, box_class)
            #     return img, seg, box_coord, box_class
                
    def __len__(self):
        return len(self.image_lists)




class TomogramTest(AbstractDataset):
    def __init__(self, img_path, patch_size, min_overlap):
        self.img = np.transpose(np.load(img_path), [2, 1, 0])
        self.coords_list = get_patch_crop_coords(patch_size, [1], self.img.shape, min_overlap)
        self.img_path = img_path
        
    def __getitem__(self, item):
        coords = np.int32(self.coords_list[item])
        
        pad_min = np.maximum(0, -np.int32(coords[:3]))
        pad_max = np.maximum(0, np.int32(coords[3:] - self.img.shape))

        crop_min = np.maximum(0, np.int32(coords[:3]))
        crop_max = np.minimum(self.img.shape, np.int32(coords[3:]))

        img_patch = self.img[crop_min[0]:crop_max[0], crop_min[1]:crop_max[1], crop_min[2]:crop_max[2]]
        img_patch = np.pad(img_patch, (
                           (pad_min[0], pad_max[0]), (pad_min[1], pad_max[1]), (pad_min[2], pad_max[2])),
                           'reflect')

        return img_patch, coords

    def __len__(self):
        return len(self.coords_list)

