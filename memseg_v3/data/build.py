# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
from glob import glob
import random
import numpy as np

import torch.utils.data
from utils.comm import get_world_size

from .datasets import TomogramTrain, TomogramTest
from . import samplers

from .collate_batch import TrainBatchCollator, TestBatchCollator
from .transforms import build_train_transforms, build_test_transforms
from .prepare_data import generate_dataset

import pdb

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        pdb.set_trace()
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(dataset, sampler, images_per_batch, num_iters=None, start_iter=0):

    batch_sampler = torch.utils.data.sampler.BatchSampler(
        sampler, images_per_batch, drop_last=False
    )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_train_loader(cfg, is_distributed=False, start_iter=0, is_for_period=False):
    # get train and valid data path list
    data_list = glob(cfg.DATASETS.TRAIN + "/*/*" + cfg.DATASETS.TRAIN_SUFFIX + "*")
    random.shuffle(data_list)

    if cfg.DATASETS.IS_VALID:
        train_data_list = data_list
        valid_data_list = glob(cfg.DATASETS.VALID)
    else:
        val_num = max(int(cfg.DATASETS.SPLIT * len(data_list)), 1)
        train_data_list = data_list
        valid_data_list = []
    
    train_data_list = generate_dataset(train_data_list, output_dir=cfg.DATASETS.TRAIN_SAVE, dataset_name='train')
    valid_data_list = generate_dataset(valid_data_list, output_dir=cfg.DATASETS.VALID_SAVE, dataset_name='valid', crop_size=cfg.DATASETS.VAL_CROP)
        
    # If bbox aug is enabled in testing, simply set transforms to None and we will apply transforms later
    train_transforms = build_train_transforms(cfg)
    if cfg.DATASETS.VALID_AUG:
        valid_transforms = build_train_transforms(cfg, is_train=False)
    else:
        valid_transforms = build_test_transforms(cfg)
    

    if not isinstance(train_data_list, (list, tuple)):
        raise RuntimeError("dataset_list should be a list of strings, got {}".format(train_data_list))

    train_dataset = TomogramTrain(train_data_list, transforms=train_transforms, is_label=True)
    valid_dataset = TomogramTrain(valid_data_list, transforms=valid_transforms, is_label=cfg.DATASETS.IS_LABEL)
    datasets = [train_dataset, valid_dataset]

    images_per_batch = cfg.SOLVER.IMS_PER_BATCH
    images_per_gpu = images_per_batch

    valid_iters = cfg.SOLVER.MAX_EPOCH
    
    shuffle = True
    train_iters = cfg.SOLVER.MAX_EPOCH * cfg.SOLVER.ITER_PER_EPOCH
    
    data_loaders = []

    for ind, dataset in enumerate(datasets):
        
        if len(dataset) == 0:
            data_loaders.append(None)
            continue
        num_workers = cfg.DATALOADER.NUM_WORKERS
        if ind == 1:
            num_iters = valid_iters
            shuffle = False
        else:
            num_iters = train_iters
            shuffle = True
        
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, images_per_gpu, num_iters, start_iter
            )
        collator = TrainBatchCollator()
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=True,
        )
        data_loaders.append(data_loader)

    return data_loaders


def make_test_loader(cfg, is_distributed=False, start_iter=0, is_for_period=False):
    num_gpus = get_world_size()
    
    images_per_batch = cfg.TEST.IMS_PER_BATCH
    images_per_gpu = images_per_batch // num_gpus
    
    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter "
            "an out-of-memory (OOM) error if your GPU does not have "
            "sufficient memory. If this happens, you can reduce "
            "SOLVER.IMS_PER_BATCH (for training) or "
            "TEST.IMS_PER_BATCH (for inference). For training, you must "
            "also adjust the learning rate and schedule length according "
            "to the linear scaling rule. "
        )

    # get test data path list
    test_data_list = glob(cfg.DATASETS.TEST)
    test_data_list = generate_dataset(test_data_list, output_dir=cfg.DATASETS.TEST_SAVE, dataset_name='test', crop_patch=cfg.DATASETS.VAL_CROP)
    
    datasets = []
    for test_data_path in test_data_list:
        test_dataset = TomogramTest(test_data_path, cfg.INPUT.PATCH_SIZE, cfg.TEST.MIN_OVERLAP)
        datasets.append(test_dataset)

    data_loaders = []
    for ind, dataset in enumerate(datasets):
        test_iters = np.ceil(len(dataset) / cfg.TEST.IMS_PER_BATCH)

        num_workers = cfg.TEST.IMS_PER_BATCH
        shuffle = False
        
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler, images_per_gpu, test_iters, start_iter
            )
        collator = TestBatchCollator()
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
            pin_memory=True
        )
        data_loaders.append(data_loader)

    return data_loaders

