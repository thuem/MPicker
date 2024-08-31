# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

import torch
from torch import nn
import torch.nn.functional as F

import time

from ..backbone import build_backbone
from ..make_layers import make_conv
from ..utils import batch_dice

import pdb


class MaskRCNN3D(nn.Module):
    """
    Main class for Generalized R-CNN. Currently supports boxes and masks.
    It consists of three main parts:
    - backbone
    - rpn
    - heads: takes the features + the proposals from the RPN and computes
        detections / masks from it.
    """

    def __init__(self, cfg):
        super(MaskRCNN3D, self).__init__()

        self.backbone = build_backbone(cfg)
        self.conv_mask = make_conv(cfg.MODEL.RESNETS.OUT_CHANNELS, cfg.MODEL.OUTPUT.SEG_CLASS, ks=1, pad=0, norm=cfg.MODEL.FPN.USE_NORM, relu=None)
        self.max_pool = torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=(1, 1, 1))
        self.backbone_type = cfg.MODEL.BACKBONE

    def forward(self, img, mask=None, box_coord=None, box_class=None):
        seg_logits = self.backbone(img)

        if self.training:
            seg_loss_dice = 1 - batch_dice(F.softmax(seg_logits, dim=1), mask)
            seg_loss_ce = F.cross_entropy(seg_logits, mask.long())
            seg_losses = (seg_loss_dice + seg_loss_ce) / 2
            
            return seg_losses
        else:
            seg_predict = F.softmax(seg_logits, dim=1)
            return seg_predict