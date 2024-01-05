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


class BottleNeck(nn.Module):
    def __init__(self, channel: int = 64, factor: int = 2):
        super(BottleNeck, self).__init__()
        self.channel = channel
        self.factor = factor
        self.hid_channel = self.channel // self.factor
        self.conv1 = ConvBnAct(self.channel, self.hid_channel, kernel_size=1)
        self.conv2 = ConvBnAct(self.hid_channel, self.hid_channel, kernel_size=3, stride=1, padding=1)
        self.conv3 = ConvBnAct(self.hid_channel, self.channel, kernel_size=1, act=False)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x_identity = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.act(x + x_identity)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, channel: int, reduction: int = 8):
        super(ChannelAttention, self).__init__()
        self.channel = channel
        self.reduction = reduction
        hid_channel = self.channel // self.reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(self.channel * 4, hid_channel, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channel, self.channel * 4, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        ca_max = self.mlp(self.max_pool(x))
        ca_avg = self.mlp(self.avg_pool(x))
        x = x * self.act(ca_max + ca_avg)

        return x


class GuideRefinementBlock(nn.Module):
    def __init__(self, channel: int):
        super(GuideRefinementBlock, self).__init__()
        self.channel = channel
        self.channel_attention = ChannelAttention(self.channel)
        self.fusion_conv = ConvBnAct(self.channel * 4, self.channel, kernel_size=3, stride=1, padding=1)
        self.pools_sizes = [0, 2, 4, 8]
        self.pool_convs = nn.ModuleList()
        self.refine_convs = nn.ModuleList()
        for pool_size in self.pools_sizes:
            pool_conv = nn.Sequential(
                nn.AvgPool2d(kernel_size=pool_size, stride=pool_size) if pool_size > 0 else nn.Identity(),
                nn.Conv2d(self.channel, self.channel, kernel_size=1)
            )
            refine_conv = ConvBnAct(self.channel, self.channel, kernel_size=3, stride=1, padding=1)
            self.pool_convs.append(pool_conv)
            self.refine_convs.append(refine_conv)

    def forward(self, inputs: list):
        inputs_up = inputs.copy()
        for i in range(len(inputs_up) - 1, 0, -1):
            inputs_up[i] = F.interpolate(inputs_up[i], inputs_up[0].size()[2:], mode='bilinear', align_corners=True)

        inputs_agg = self.channel_attention(torch.cat(inputs_up, dim=1))
        inputs_agg = self.fusion_conv(inputs_agg)
        outputs = []
        for idx, pool_conv in enumerate(self.pool_convs):
            x_pool = pool_conv(inputs_agg)
            x_output = self.refine_convs[idx](x_pool + inputs[idx])
            outputs.append(x_output)

        return outputs


class FeaturePyramidNetworks(nn.Module):
    def __init__(self, in_channels: list = None,
                 out_channel: int = 128,
                 num_grbs=2,
                 pool_scales=None,
                 ):
        super(FeaturePyramidNetworks, self).__init__()
        if in_channels is None:
            in_channels = [64, 128, 256, 512]
        if pool_scales is None:
            pool_scales = [1, 2, 3, 6]
        self.in_channels = in_channels
        self.out_channel = out_channel
        self.pool_scales = pool_scales
        self.num_grbs = num_grbs
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.psp_module = PyramidPoolingModule(in_channel=self.in_channels[-1], out_channel=self.out_channel,
                                               pool_scales=self.pool_scales)
        for in_channel in self.in_channels[:-1]:
            lateral_conv = ConvBnAct(in_channel, self.out_channel, kernel_size=1)
            fpn_conv = ConvBnAct(self.out_channel, self.out_channel, kernel_size=3, stride=1, padding=1)
            self.lateral_convs.append(lateral_conv)
            self.fpn_convs.append(fpn_conv)
        self.lateral_convs.append(self.psp_module)

        self.guide_refinement_blocks = nn.ModuleList(
            [GuideRefinementBlock(self.out_channel) for i in range(self.num_grbs)]
        )

        self.fusion_conv = ConvBnAct(self.out_channel * len(self.in_channels), self.out_channel,
                                     kernel_size=3, stride=1, padding=1)

    @staticmethod
    def up_add(a, b):
        out = F.interpolate(a, b.size()[2:], mode='bilinear', align_corners=True) + b

        return out

    def forward(self, inputs: list):
        pyramid_features = [lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)]

        for idx, grb in enumerate(self.guide_refinement_blocks):
            pyramid_features = grb(pyramid_features)

        for i in range(len(inputs) - 2, -1, -1):
            pyramid_features[i] = self.up_add(pyramid_features[i+1], pyramid_features[i])
            pyramid_features[i] = self.fpn_convs[i](pyramid_features[i])

        for i in range(len(inputs) - 1, 0, -1):
            pyramid_features[i] = F.interpolate(
                pyramid_features[i], pyramid_features[0].size()[2:], mode='bilinear', align_corners=True)

        p_out = self.fusion_conv(torch.cat(pyramid_features, dim=1))

        return p_out


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
