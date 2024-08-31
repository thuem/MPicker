# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from torch import nn
import pdb

from ._utils import log2
from cuda_functions.crop_and_resize import CropAndResizeFunction as ra3D

def pyramid_roi_align(features, proposals, pool_size, pyramid_levels):
    """
    Implements ROI Pooling on multiple levels of the feature pyramid.
    :param features: list of feature maps, each of shape (b, c, y, x , z)
    :param proposals: proposals (normalized coords.) as returned by RPN. contain info about original batch element allocation.
    (b, n_proposals, x1, y1, z1, x2, y2, z2)
    :param pool_size: list of poolsizes in dims: [x, y, z]
    :param pyramid_levels: list. [0, 1, 2, ...]
    :return: pooled: pooled feature map rois (b, n_proposals, c, poolsize_y, poolsize_x, poolsize_z)

    Output:
    Pooled regions in the shape: [num_boxes, height, width, channels].
    The width and height are those specific in the pool_shape in the layer
    constructor.
    """
    device = proposals.device
    batch_idx = torch.arange(proposals.shape[0]).unsqueeze(1).repeat(1,proposals.shape[1]).view(-1)
    boxes = proposals.view(-1, 6)


    # Assign each ROI to a level in the pyramid based on the ROI area.
    x1, y1, z1, x2, y2, z2 = boxes.chunk(6, dim=1)

    w = x2 - x1
    h = y2 - y1

    # Equation 1 in https://arxiv.org/abs/1612.03144. Account for
    # the fact that our coordinates are normalized here.
    # divide sqrt(h*w) by 1 instead image_area.
    roi_level = (4 + log2(torch.sqrt(h * w))).round().int().clamp(pyramid_levels[0], pyramid_levels[-1])

    # if Pyramid contains additional level P6, adapt the roi_level assignemnt accordingly.
    if len(pyramid_levels) == 5:
        roi_level[h * w > 0.65] = 5

    # Loop through levels and apply ROI pooling to each.
    pooled = []
    box_to_level = []

    for level_ix, level in enumerate(pyramid_levels):
        ix = roi_level == level
        if not ix.any():
            continue
        ix = torch.nonzero(ix)[:, 0]
        level_boxes = boxes[ix, :]
        # re-assign rois to feature map of original batch element.
        ind = batch_idx[ix].int().to(device)

        # Keep track of which box is mapped to which level
        box_to_level.append(ix)

        # Stop gradient propogation to ROI proposals
        level_boxes = level_boxes.detach()
        level_boxes = level_boxes[:, [0, 1, 3, 4, 2, 5]]

        # Crop and Resize
        # From Mask R-CNN paper: "We sample four regular locations, so
        # that we can evaluate either max or average pooling. In fact,
        # interpolating only a single value at each bin center (without
        # pooling) is nearly as effective."
        pooled_features = ra3D.apply(features[level_ix], level_boxes, ind, pool_size[0], pool_size[1], pool_size[2], 0)
        pooled.append(pooled_features)
    # Pack pooled features into one tensor
    pooled = torch.cat(pooled, dim=0)

    # Pack box_to_level mapping into one array and add another
    # column representing the order of pooled boxes
    box_to_level = torch.cat(box_to_level, dim=0)

    # Rearrange pooled features to match the order of the original boxes
    _, box_to_level = torch.sort(box_to_level)
    pooled = pooled[box_to_level, :, :]

    return pooled
