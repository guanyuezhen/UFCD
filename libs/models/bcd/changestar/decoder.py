import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import ConvBnAct, PyramidPoolingModule


class UPerNet(nn.Module):
    def __init__(self, in_channels: list = None, out_channel: int = 128, pool_scales=None):
        super(UPerNet, self).__init__()
        if in_channels is None:
            in_channels = [64, 128, 256, 512]
        if pool_scales is None:
            pool_scales = [1, 2, 3, 6]
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.pool_scales = pool_scales
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.psp_module = PyramidPoolingModule(in_channel=self.in_channels[-1], out_channel=self.out_channel,
                                               pool_scales=self.pool_scales)
        for in_channel in self.in_channels[:-1]:
            lateral_conv = nn.Conv2d(in_channel, self.out_channel, kernel_size=1)
            fpn_conv = ConvBnAct(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        self.lateral_convs.append(self.psp_module)

        self.fusion_conv = ConvBnAct(self.out_channel * len(self.in_channels), self.out_channel,
                                     kernel_size=3, stride=1, padding=1)

    @staticmethod
    def up_add(a, b):
        out = F.interpolate(a, b.size()[2:], mode='bilinear', align_corners=True) + b
        return out

    def forward(self, inputs: list):
        pyramid_features = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        for i in range(len(inputs) - 2, -1, -1):
            pyramid_features[i] = self.up_add(pyramid_features[i + 1], pyramid_features[i])

        for i in range(len(inputs) - 2, -1, -1):
            pyramid_features[i] = self.fpn_convs[i](pyramid_features[i])

        for i in range(len(inputs) - 1, 0, -1):
            pyramid_features[i] = F.interpolate(
                pyramid_features[i], pyramid_features[0].size()[2:], mode='bilinear', align_corners=True)

        p_out = self.fusion_conv(torch.cat(pyramid_features, dim=1))

        return p_out
