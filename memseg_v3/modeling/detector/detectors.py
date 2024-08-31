# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .generalized_rcnn import MaskRCNN3D

_DETECTION_META_ARCHITECTURES = {"MaskRCNN3D": MaskRCNN3D}

def build_detection_model(cfg):
    meta_arch = _DETECTION_META_ARCHITECTURES[cfg.MODEL.META_ARCHITECTURE]
    return meta_arch(cfg)
