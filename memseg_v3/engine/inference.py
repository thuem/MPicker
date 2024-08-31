# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from calendar import c
import logging
import time
import os

import torch
import numpy as np
from tqdm import tqdm
import math

import mrcfile

import pdb

from utils.clustering import weighted_box_clustering, compute_box_factors
# from utils.plotting import plot_valid_box

def save_mrc(img, path):
    with mrcfile.new(path, overwrite=True) as tomo:
        tomo.set_data(np.float32(img))


def inference(
    cfg,
    model,
    data_loader_test,
    device,
    output_folder,
    logger
):

    # writer = SummaryWriter('./output/tensorboard')
    #logger = logging.getLogger("maskrcnn_benchmark.trainer")
    logger.info("Start inferencing")
    model.eval()
    start_training_time = time.time()
    end = time.time()

    seg_result = np.zeros(data_loader_test.dataset.img.shape)
    seg_count = np.zeros(seg_result.shape)
    coord_bound = np.concatenate([seg_result.shape, seg_result.shape]) - 1
    overlap = np.array(cfg.TEST.MIN_OVERLAP) // 2

    # for box_coord, box_score, box_class, , overlap
    detection_result = []
    start_time = time.time()

    with tqdm(total=math.ceil(len(data_loader_test.dataset) / cfg.TEST.IMS_PER_BATCH), desc='',postfix='') as bar:

        for iteration, (img, coords) in enumerate(data_loader_test):

            iteration = iteration + 1
            with torch.no_grad():
                img = torch.Tensor(img)
                img = img.unsqueeze(dim=1)
                img = img.to(device)

                img_shape = img.shape[2:]
                patch_size = np.array(img_shape) / 2

                seg_predictions = model(img)
                for i in range(3):
                    seg_predict = model(torch.flip(img, [i + 2]))
                    seg_predict = torch.flip(seg_predict, [i + 2])
                    
                    seg_predictions += seg_predict
                seg_predictions /= 4


            for idx in range(seg_predictions.shape[0]):
                crop_flag =  (coords[idx, 3:] < seg_result.shape)
                coord = coords[idx, :] + np.array([overlap[0], overlap[1], overlap[2],
                    -overlap[0] * crop_flag[0],
                    -overlap[1] * crop_flag[1],
                    -overlap[2] * crop_flag[2]])
                seg_predict_patch = np.array(seg_predictions[idx, 1, :].cpu())
                img_patch = np.array(img[idx].cpu())
                # with mrcfile.new(output_folder.replace(".mrc", "{}_{}_patch.mrc".format(iteration, idx)), overwrite=True) as tomo:
                #     tomo.set_data(np.float32(seg_predict_patch))
                # with mrcfile.new(output_folder.replace(".mrc", "{}_{}_img.mrc".format(iteration, idx)), overwrite=True) as tomo:
                #     tomo.set_data(np.float32(img_patch))

                seg_predict_crop = seg_predict_patch[overlap[0]:overlap[0] + coord[3] - coord[0],
                                                    overlap[1]:overlap[1] + coord[4] - coord[1],
                                                    overlap[2]:overlap[2] + coord[5] - coord[2]]

                crop_min = np.maximum(0, np.int32(coord[:3]))
                crop_max = np.minimum(seg_result.shape, np.int32(coord[3:]))

                pad_min = np.maximum(0, -np.int32(coord[:3]))
                pad_max = np.maximum(0, np.int32(coord[3:] - seg_result.shape))
                patch_shape = seg_predict_crop.shape

                seg_result[crop_min[0]:crop_max[0], crop_min[1]:crop_max[1], crop_min[2]:crop_max[2]] += seg_predict_crop[
                        pad_min[0]: patch_shape[0] - pad_max[0],
                        pad_min[1]: patch_shape[1] - pad_max[1],
                        pad_min[2]: patch_shape[2] - pad_max[2]]
                seg_count[crop_min[0]:crop_max[0], crop_min[1]:crop_max[1], crop_min[2]:crop_max[2]] += 1

            bar.update(1)

        with mrcfile.new(output_folder, overwrite=True) as tomo:
            tomo.set_data(np.transpose(np.float32(seg_result), [2, 1, 0]))
