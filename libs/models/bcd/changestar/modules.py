import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnAct(nn.Module):
    def __init__(self, in_channel, out_channel,
                 kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True, act=True):
        super(ConvBnAct, self).__init__()
        if act:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, groups=groups, bias=bias),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(inplace=True)
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=stride, padding=padding,
                          dilation=dilation, groups=groups, bias=bias),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        x = self.conv(x)
        return x


class PyramidPoolingModule(nn.Module):
    def __init__(self, in_channel: int = 512, out_channel: int = 64, pool_scales: list = None):
        super(PyramidPoolingModule, self).__init__()
        if pool_scales is None:
            pool_scales = [1, 2, 3, 6]
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.pool_scales = pool_scales
        self.hid_channel = in_channel // 4
        self.pool_convs = nn.ModuleList()
        for pool_scale in self.pool_scales:
            pool_conv = nn.Sequential(
                nn.AdaptiveAvgPool2d(pool_scale),
                ConvBnAct(self.in_channel, self.hid_channel, kernel_size=1)
            )
            self.pool_convs.append(pool_conv)
        self.fusion_conv = ConvBnAct(self.in_channel + self.hid_channel * 4, self.out_channel,
                                     kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        pp_x = [x]
        for idx, pool_conv in enumerate(self.pool_convs):
            pool_x = pool_conv(x)
            pool_x = F.interpolate(pool_x, x.size()[2:], mode='bilinear', align_corners=True)
            pp_x.append(pool_x)
        ppm_out = self.fusion_conv(torch.cat(pp_x, dim=1))

        return ppm_out


class ChangeMixin(nn.Module):
    def __init__(self, in_channel=256, inner_channel=16, num_class=2):
        super(ChangeMixin, self).__init__()
        self.in_channel = in_channel
        self.inner_channel = inner_channel
        self.num_class = num_class
        self.fusion_layer = nn.Sequential(
            ConvBnAct(self.in_channel * 2, self.inner_channel, kernel_size=3, stride=1, padding=1),
            ConvBnAct(self.inner_channel, self.inner_channel, kernel_size=3, stride=1, padding=1),
            ConvBnAct(self.inner_channel, self.inner_channel, kernel_size=3, stride=1, padding=1),
            ConvBnAct(self.inner_channel, self.inner_channel, kernel_size=3, stride=1, padding=1),
        )
        self.cls_layer = nn.Conv2d(self.inner_channel, self.num_class, 3, 1, 1)

    def forward(self, t1, t2):
        change_t1vt2 = self.cls_layer(self.fusion_layer(torch.cat([t1, t2], dim=1)))
        change_t2vt1 = self.cls_layer(self.fusion_layer(torch.cat([t2, t1], dim=1)))

        return [change_t1vt2, change_t2vt1]
