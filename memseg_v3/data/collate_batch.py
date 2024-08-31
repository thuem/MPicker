# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import numpy as np


class TrainBatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))

        if len(transposed_batch) == 4:
            img = np.stack(transposed_batch[0])
            seg = np.stack(transposed_batch[1])
            box_coord = np.array(transposed_batch[2])
            box_class = np.array(transposed_batch[3])
            return img, seg, box_coord, box_class
        else:
            img_weak = np.stack(transposed_batch[0])
            img = np.stack(transposed_batch[1])
            seg = np.stack(transposed_batch[2])
            box_coord = np.array(transposed_batch[3])
            box_class = np.array(transposed_batch[4])
            return img_weak, img, seg, box_coord, box_class


class TestBatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
        
        img = np.stack(transposed_batch[0])
        coords = np.stack(transposed_batch[1])

        return img, coords

