# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torch.nn.functional as F
from torch import nn

from .net_utils import ResBlock, Interpolate
from ..make_layers import make_conv

import pdb

class FPN(nn.Module):
    """
    Feature Pyramid Network from https://arxiv.org/pdf/1612.03144.pdf with options for modifications.
    by default is constructed with Pyramid levels P2, P3, P4, P5.
    """
    def __init__(self, cfg, operate_stride1=False):
        """
        from configs:
        :param input_channels: number of channel dimensions in input data.
        :param in_channels:  number of feature_maps in first layer. rest is scaled accordingly.
        :param out_channels: number of feature_maps for output_layers of all levels in decoder.
        :param conv: instance of custom conv class containing the dimension info.
        :param res_architecture: string deciding whether to use "resnet50" or "resnet101".
        :param operate_stride1: boolean flag. enables adding of Pyramid levels P1 (output stride 2) and P0 (output stride 1).
        :param norm: string specifying type of feature map normalization. If None, no normalization is applied.
        :param relu: string specifying type of nonlinearity. If None, no nonlinearity is applied.
        :param sixth_pooling: boolean flag. enables adding of Pyramid level P6.
        """
        super(FPN, self).__init__()

        self.in_channels = cfg.MODEL.RESNETS.IN_CHANNELS
        self.out_channels = cfg.MODEL.RESNETS.OUT_CHANNELS
        self.n_blocks = [3, 4, {"resnet50": 6, "resnet101": 23}[cfg.MODEL.RESNETS.ARCHITECTURE], 3]
        self.block = ResBlock
        self.block_expansion = cfg.MODEL.RESNETS.BLOCK_EXPANSION
        self.sixth_pooling = cfg.MODEL.RESNETS.SIXTH_POOLING
        self.operate_stride1 = operate_stride1
        
        self.norm = cfg.MODEL.FPN.USE_NORM
        self.relu = cfg.MODEL.FPN.USE_RELU

        if self.operate_stride1:
            self.C0 = nn.Sequential(make_conv(1, self.in_channels, ks=3, pad=1, norm=self.norm, relu=self.relu),
                                    make_conv(self.in_channels, self.in_channels, ks=3, pad=1, norm=self.norm, relu=self.relu))

            self.C1 = make_conv(self.in_channels, self.in_channels, ks=7, stride=(2, 2, 1), pad=3, norm=self.norm, relu=self.relu)

        else:
            self.C1 = make_conv(1, self.in_channels, ks=7, stride=(2, 2, 1), pad=3, norm=self.norm, relu=self.relu)

        expansion_channels = self.in_channels * self.block_expansion

        C2_layers = []
        C2_downsample = make_conv(self.in_channels, expansion_channels, ks=1, stride=1, norm=self.norm, relu=self.relu)
        C2_layers.append(nn.MaxPool3d(kernel_size=3, stride=(2, 2, 1), padding=1))
        C2_layers.append(self.block(self.in_channels, self.in_channels, stride=1, norm=self.norm, relu=self.relu, 
                         downsample=C2_downsample))
        
        for i in range(1, self.n_blocks[0]):
            C2_layers.append(self.block(expansion_channels, self.in_channels, norm=self.norm, relu=self.relu))
        self.C2 = nn.Sequential(*C2_layers)

        C3_layers = []
        C3_downsample = make_conv(expansion_channels, expansion_channels * 2, ks=1, stride=2, norm=self.norm, relu=self.relu)
        C3_layers.append(self.block(expansion_channels, self.in_channels * 2, stride=2, norm=self.norm, relu=self.relu,
                                    downsample=C3_downsample))
        
        for i in range(1, self.n_blocks[1]):
            C3_layers.append(self.block(expansion_channels * 2, self.in_channels * 2, norm=self.norm, relu=self.relu))
        self.C3 = nn.Sequential(*C3_layers)

        C4_layers = []
        C4_downsample = make_conv(expansion_channels * 2, expansion_channels * 4, ks=1, stride=2, norm=self.norm, relu=self.relu)
        C4_layers.append(self.block(
            expansion_channels * 2, self.in_channels * 4, stride=2, norm=self.norm, relu=self.relu, downsample=C4_downsample))
        
        for i in range(1, self.n_blocks[2]):
            C4_layers.append(self.block(expansion_channels * 4, self.in_channels * 4, norm=self.norm, relu=self.relu))
        self.C4 = nn.Sequential(*C4_layers)

        C5_layers = []
        C5_downsample = make_conv(expansion_channels * 4, expansion_channels * 8, ks=1, stride=2, norm=self.norm, relu=self.relu)
        C5_layers.append(self.block(
            expansion_channels * 4, self.in_channels * 8, stride=2, norm=self.norm, relu=self.relu, downsample=C5_downsample))
        
        for i in range(1, self.n_blocks[3]):
            C5_layers.append(self.block(expansion_channels * 8, self.in_channels * 8, norm=self.norm, relu=self.relu))
        self.C5 = nn.Sequential(*C5_layers)

        if self.sixth_pooling:
            C6_layers = []
            C6_downsample = make_conv(expansion_channels * 8, expansion_channels * 16, ks=1, stride=2, norm=self.norm, relu=self.relu)
            C6_layers.append(self.block(
                expansion_channels * 8, self.in_channels * 16, stride=2, norm=self.norm, relu=self.relu, downsample=C6_downsample))
           
            for i in range(1, self.n_blocks[3]):
                C6_layers.append(self.block(expansion_channels * 16, self.in_channels * 16, norm=self.norm, relu=self.relu))
            self.C6 = nn.Sequential(*C6_layers)


        self.P1_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')
        self.P2_upsample = Interpolate(scale_factor=(2, 2, 1), mode='trilinear')

        self.P5_conv1 = make_conv(self.in_channels * 32 + cfg.MODEL.RESNETS.N_LATENT_DIMS, self.out_channels, ks=1, stride=1, relu=None) #
        self.P4_conv1 = make_conv(self.in_channels * 16, self.out_channels, ks=1, stride=1, relu=None)
        self.P3_conv1 = make_conv(self.in_channels * 8, self.out_channels, ks=1, stride=1, relu=None)
        self.P2_conv1 = make_conv(self.in_channels * 4, self.out_channels, ks=1, stride=1, relu=None)
        self.P1_conv1 = make_conv(self.in_channels, self.out_channels, ks=1, stride=1, relu=None)

        if operate_stride1:
            self.P0_conv1 = make_conv(self.in_channels, self.out_channels, ks=1, stride=1, relu=None)
            self.P0_conv2 = make_conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        self.P1_conv2 = make_conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P2_conv2 = make_conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P3_conv2 = make_conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P4_conv2 = make_conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)
        self.P5_conv2 = make_conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)

        if self.sixth_pooling:
            self.P6_conv1 = make_conv(self.in_channels * 64, self.out_channels, ks=1, stride=1, relu=None)
            self.P6_conv2 = make_conv(self.out_channels, self.out_channels, ks=3, stride=1, pad=1, relu=None)


    def forward(self, x):
        """
        :param x: input image of shape (b, c, x, y, z)
        :return: list of output feature maps per pyramid level, each with shape (b, c, x, y, z).
        """
        if self.operate_stride1:
            c0_out = self.C0(x)
        else:
            c0_out = x

        c1_out = self.C1(c0_out)
        c2_out = self.C2(c1_out)
        c3_out = self.C3(c2_out)
        c4_out = self.C4(c3_out)
        c5_out = self.C5(c4_out)
        
        
        if self.sixth_pooling:
            c6_out = self.C6(c5_out)
            p6_pre_out = self.P6_conv1(c6_out)
            p5_pre_out = self.P5_conv1(c5_out) + F.interpolate(p6_pre_out, scale_factor=2)
        else:
            p5_pre_out = self.P5_conv1(c5_out)

        p4_pre_out = self.P4_conv1(c4_out) + F.interpolate(p5_pre_out, scale_factor=2)
        p3_pre_out = self.P3_conv1(c3_out) + F.interpolate(p4_pre_out, scale_factor=2)
        p2_pre_out = self.P2_conv1(c2_out) + F.interpolate(p3_pre_out, scale_factor=2)

        p2_out = self.P2_conv2(p2_pre_out)
        p3_out = self.P3_conv2(p3_pre_out)
        p4_out = self.P4_conv2(p4_pre_out)
        p5_out = self.P5_conv2(p5_pre_out)
        out_list = [p2_out, p3_out, p4_out, p5_out]

        if self.sixth_pooling:
            p6_out = self.P6_conv2(p6_pre_out)
            out_list.append(p6_out)

        if self.operate_stride1:
            p1_pre_out = self.P1_conv1(c1_out) + self.P2_upsample(p2_pre_out)
            p0_pre_out = self.P0_conv1(c0_out) + self.P1_upsample(p1_pre_out)
            # p1_out = self.P1_conv2(p1_pre_out) # usually not needed.
            p0_out = self.P0_conv2(p0_pre_out)
            out_list = [p0_out] + out_list

        return out_list