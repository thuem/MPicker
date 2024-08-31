# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Miscellaneous utility functions
"""

import torch
from torch import nn
from torch.nn import functional as F
from config import cfg
#from layers import Conv2d


def get_group_gn(dim, dim_per_gp, num_groups):
    """get number of groups used by GroupNorm, based on number of channels."""
    assert dim_per_gp == -1 or num_groups == -1, \
        "GroupNorm: can only specify G or C/G."

    if dim_per_gp > 0:
        assert dim % dim_per_gp == 0, \
            "dim: {}, dim_per_gp: {}".format(dim, dim_per_gp)
        group_gn = dim // dim_per_gp
    else:
        assert dim % num_groups == 0, \
            "dim: {}, num_groups: {}".format(dim, num_groups)
        group_gn = num_groups

    return group_gn


def group_norm(out_channels, affine=True, divisor=1):
    out_channels = out_channels // divisor
    dim_per_gp = cfg.MODEL.GROUP_NORM.DIM_PER_GP // divisor
    num_groups = cfg.MODEL.GROUP_NORM.NUM_GROUPS // divisor
    eps = cfg.MODEL.GROUP_NORM.EPSILON # default: 1e-5
    return torch.nn.GroupNorm(
        get_group_gn(out_channels, dim_per_gp, num_groups),
        out_channels,
        eps,
        affine
    )


def make_conv(
    c_in,
    c_out,
    ks,
    dim=3,
    pad=0,
    stride=1,
    norm=None,
    relu=None,
    kaiming_init=False
):
    if dim == 2:
        conv = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=ks,
            stride=stride,
            padding=pad
        )
        if norm == "instance_norm":
            norm_layer = nn.InstanceNorm2d(c_out)
        elif norm == 'batch_norm':
            norm_layer = nn.BatchNorm2d(c_out)
        else:
            norm_layer = None
    elif dim == 3:
        conv = nn.Conv3d(
            c_in,
            c_out,
            kernel_size=ks,
            stride=stride,
            padding=pad
        )
        if norm == "instance_norm":
            norm_layer = nn.InstanceNorm3d(c_out)
        elif norm == 'batch_norm':
            norm_layer = nn.BatchNorm3d(c_out)
        else:
            norm_layer = None
    else:
        raise ValueError('Input dimension as specified in configs is not implemented... {}'.format(dim))
        
    if kaiming_init:
        if relu == "leakyrelu":
            nn.init.kaiming_normal_(
                conv.weight, mode="fan_out", nonlinearity="leaky_relu"
            )
        else:
            nn.init.kaiming_normal_(
                conv.weight, mode="fan_out", nonlinearity="relu"
            )
    else:
        torch.nn.init.normal_(conv.weight, std=0.01)
        torch.nn.init.constant_(conv.bias, 0)
    
    if norm_layer is not None:
        conv = nn.Sequential(conv, norm_layer)

    if relu == "relu":
        relu_layer = nn.ReLU(inplace=True)
        conv = nn.Sequential(conv, relu_layer)
    elif relu == "leakyrelu":
        relu_layer = nn.LeakyReLU(negative_slope=0.1, inplace=True)
        conv = nn.Sequential(conv, relu_layer)

    return conv

def make_fc(dim_in, hidden_dim):
    '''
        Caffe2 implementation uses XavierFill, which in fact
        corresponds to kaiming_uniform_ in PyTorch
    '''
    fc = nn.Linear(dim_in, hidden_dim)
    nn.init.kaiming_uniform_(fc.weight, a=1)
    nn.init.constant_(fc.bias, 0)
    return fc

def conv_with_kaiming_uniform(use_gn=False, use_relu=False):
    def make_conv(
        in_channels, out_channels, kernel_size, stride=1, dilation=1
    ):
        conv = Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=dilation * (kernel_size - 1) // 2,
            dilation=dilation,
            bias=False if use_gn else True
        )
        # Caffe2 implementation uses XavierFill, which in fact
        # corresponds to kaiming_uniform_ in PyTorch
        nn.init.kaiming_uniform_(conv.weight, a=1)
        if not use_gn:
            nn.init.constant_(conv.bias, 0)
        module = [conv,]
        if use_gn:
            module.append(group_norm(out_channels))
        if use_relu:
            module.append(nn.ReLU(inplace=True))
        if len(module) > 1:
            return nn.Sequential(*module)
        return conv

    return make_conv
