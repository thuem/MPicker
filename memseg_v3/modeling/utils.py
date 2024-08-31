# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import numpy as np
import torch
import pdb

def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (x1, y1, z1, x2, y2, z2))
    For better performance, pass the largest set first and the smaller second.
    """
    # Areas of anchors and GT boxes
    volume1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    volume2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = np.zeros((boxes1.shape[0], boxes2.shape[0]))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]  # this is the gt box
        overlaps[:, i] = compute_iou_3D(box2, boxes1, volume2[i], volume1)
    return overlaps

def compute_iou_3D(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [x1, y1, z1, x2, y2, z2] (typically gt box)
    boxes: [boxes_count, (x1, y1, z1, x2, y2, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    z1 = np.maximum(box[2], boxes[:, 2])
    x2 = np.minimum(box[3], boxes[:, 3])
    y2 = np.minimum(box[4], boxes[:, 4])
    z2 = np.minimum(box[5], boxes[:, 5])
    intersection = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0) * np.maximum(z2 - z1, 0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / union

    return iou

# def compute_combined_overlaps(anchors, targets, device):
#     anchors_gpu = anchors.to(device)
#     volume1 = (anchors_gpu[:, 3] - anchors_gpu[:, 0]) * (anchors_gpu[:, 4] - anchors_gpu[:, 1]) * (anchors_gpu[:, 5] - anchors_gpu[:, 2])

#     overlaps = []
#     for idx in range(targets.shape[0]):
#         targets_gpu = torch.from_numpy(targets[idx]).to(device)
#         volume2 = (targets_gpu[:, 3] - targets_gpu[:, 0]) * (targets_gpu[:, 4] - targets_gpu[:, 1]) * (targets_gpu[:, 5] - targets_gpu[:, 2])
#         overlaps_gpu = torch.zeros((anchors_gpu.shape[0], targets_gpu.shape[0]), device=device)
#         for i in range(overlaps_gpu.shape[1]):
#             box2 = targets_gpu[i]  # this is the gt box
#             overlaps_gpu[:, i] = compute_iou_3D_gpu(box2, anchors_gpu, volume2[i], volume1)
#         overlaps.append(overlaps_gpu)
#     return overlaps

def compute_combined_overlaps(anchors, targets, device):
    anchors_gpu = anchors.to(device)
    volume1 = (anchors_gpu[:, 3] - anchors_gpu[:, 0]) * (anchors_gpu[:, 4] - anchors_gpu[:, 1]) * (anchors_gpu[:, 5] - anchors_gpu[:, 2])

    overlaps = []
    for idx in range(targets.shape[0]):
        if targets[idx].shape[0] is not 0:
            targets_gpu = torch.from_numpy(targets[idx]).to(device)
            volume2 = (targets_gpu[:, 3] - targets_gpu[:, 0]) * (targets_gpu[:, 4] - targets_gpu[:, 1]) * (targets_gpu[:, 5] - targets_gpu[:, 2])
            overlaps_gpu = torch.zeros((anchors_gpu.shape[0], targets_gpu.shape[0]), device=device)
        else:
            overlaps_gpu = torch.zeros((anchors_gpu.shape[0], targets[idx].shape[0]), device=device)
        for i in range(overlaps_gpu.shape[1]):
            box2 = targets_gpu[i]  # this is the gt box
            overlaps_gpu[:, i] = compute_iou_3D_gpu(box2, anchors_gpu, volume2[i], volume1)
        overlaps.append(overlaps_gpu)
    return overlaps


def compute_overlaps_gpu(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: [N, (x1, y1, z1, x2, y2, z2))
    For better performance, pass the largest set first and the smaller second.
    """
    device = boxes1.device
    # Areas of anchors and GT boxes
    volume1 = (boxes1[:, 3] - boxes1[:, 0]) * (boxes1[:, 4] - boxes1[:, 1]) * (boxes1[:, 5] - boxes1[:, 2])
    volume2 = (boxes2[:, 3] - boxes2[:, 0]) * (boxes2[:, 4] - boxes2[:, 1]) * (boxes2[:, 5] - boxes2[:, 2])
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.
    overlaps = torch.zeros((boxes1.shape[0], boxes2.shape[0]), device=device)
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]  # this is the gt box
        overlaps[:, i] = compute_iou_3D_gpu(box2, boxes1, volume2[i], volume1)
    return overlaps

def compute_iou_3D_gpu(box, boxes, box_volume, boxes_volume):
    """Calculates IoU of the given box with the array of the given boxes.
    box: 1D vector [x1, y1, z1, x2, y2, z2] (typically gt box)
    boxes: [boxes_count, (x1, y1, z1, x2, y2, z2)]
    box_area: float. the area of 'box'
    boxes_area: array of length boxes_count.

    Note: the areas are passed in rather than calculated here for
          efficency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    z1 = torch.maximum(box[2], boxes[:, 2])
    x2 = torch.minimum(box[3], boxes[:, 3])
    y2 = torch.minimum(box[4], boxes[:, 4])
    z2 = torch.minimum(box[5], boxes[:, 5])

    intersection = torch.clamp(x2 - x1, min=0) * torch.clamp(y2 - y1, min=0) * torch.clamp(z2 - z1, min=0)
    union = box_volume + boxes_volume[:] - intersection[:]
    iou = intersection / union

    return iou

def sum_tensor(input, axes, keepdim=False):
    axes = np.unique(axes)
    if keepdim:
        for ax in axes:
            input = input.sum(ax, keepdim=True)
    else:
        for ax in sorted(axes, reverse=True):
            input = input.sum(int(ax))
    return input

def batch_dice(pred, y, false_positive_weight=1.0, smooth=1e-6):
    '''
    compute soft dice over batch. this is a differentiable score and can be used as a loss function.
    only dice scores of foreground classes are returned, since training typically
    does not benefit from explicit background optimization. Pixels of the entire batch are considered a pseudo-volume to compute dice scores of.
    This way, single patches with missing foreground classes can not produce faulty gradients.
    :param pred: (b, c, x, y, z), softmax probabilities (network output). (c==classes)
    :param y: (b, c, x, y, z), one-hot-encoded segmentation mask.
    :param false_positive_weight: float [0,1]. For weighting of imbalanced classes,
    reduces the penalty for false-positive pixels. Can be beneficial sometimes in data with heavy fg/bg imbalances.
    :return: soft dice score (float). This function discards the background score and returns the mean of foreground scores.
    '''
    axes = (0, 2, 3, 4)
    intersect = sum_tensor(pred * y.unsqueeze(1), axes, keepdim=False)
    denom = sum_tensor(false_positive_weight * pred + y.unsqueeze(1), axes, keepdim=False)

    return torch.mean(((2*intersect + smooth) / (denom + smooth))[1:]) # only fg dice here.