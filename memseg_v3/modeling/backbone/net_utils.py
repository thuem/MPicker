import torch.nn as nn
from ..make_layers import make_conv

# -----------------------------------------------------------------------------
# Standard ResNet block models
# -----------------------------------------------------------------------------
class ResBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None, norm=None, relu=None):
        super(ResBlock, self).__init__()
        self.conv1 = make_conv(inplanes, planes, ks=1, stride=stride, norm=norm, relu=relu)
        self.conv2 = make_conv(planes, planes, ks=3, pad=1, norm=norm, relu=relu)
        self.conv3 = make_conv(planes, planes*4, ks=1, norm=norm, relu=None)

        if relu is not None:
            self.relu = nn.ReLU(inplace=True) if relu == "relu" else nn.LeakyReLU(inplace=True)
        else:
            self.relu = None
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        
        if self.relu is not None:
            out = self.relu(out)
        return out


class Interpolate(nn.Module):
    def __init__(self, scale_factor, mode):
        super(Interpolate, self).__init__()
        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        x = self.interp(x, scale_factor=self.scale_factor, mode=self.mode, align_corners=False)
        return x