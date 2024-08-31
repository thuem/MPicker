# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .fpn import FPN
from .unet import UNet3D

def build_backbone(cfg):
    if cfg.MODEL.BACKBONE == "fpn":
        backbone = FPN(cfg, operate_stride1=True)
    elif cfg.MODEL.BACKBONE == "unet":
        backbone = UNet3D(1, 2)
    else:
        raise ValueError("Model backbone should be in fpn or unet")
    return backbone
