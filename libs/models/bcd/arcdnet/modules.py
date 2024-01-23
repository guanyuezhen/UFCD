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


class FeatureFusionModule(nn.Module):
    def __init__(self, channel):
        super(FeatureFusionModule, self).__init__()
        self.channel = channel
        self.semantic_attention_conv = nn.Sequential(
            ConvBnAct(self.channel, self.channel, kernel_size=3, stride=1, padding=1, act=False),
            nn.Sigmoid()
        )
        self.low_level_gated_context_conv = ConvBnAct(self.channel, self.channel, kernel_size=3, stride=1, padding=1)
        self.fusion_conv = ConvBnAct(self.channel * 2, self.channel, kernel_size=3, stride=1, padding=1)

    def forward(self, low_level_feature, high_level_feature):
        high_level_feature_up = F.interpolate(
            high_level_feature, low_level_feature.size()[2:], mode='bilinear', align_corners=True)
        semantic_attention = self.semantic_attention_conv(high_level_feature_up)
        low_level_feature = self.low_level_gated_context_conv(low_level_feature * (1 + semantic_attention))
        out_feature = self.fusion_conv(torch.cat([low_level_feature, high_level_feature_up], dim=1))

        return out_feature


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


class ChannelAttention(nn.Module):
    def __init__(self, channel: int, reduction: int = 8):
        super(ChannelAttention, self).__init__()
        self.channel = channel
        self.reduction = reduction
        hid_channel = self.channel // self.reduction
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(self.channel, hid_channel, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channel, self.channel, 1, bias=False),
        )
        self.act = nn.Sigmoid()

    def forward(self, x):
        ca_max = self.mlp(self.max_pool(x))
        ca_avg = self.mlp(self.avg_pool(x))
        x = x * self.act(ca_max + ca_avg)

        return x


class FeatureEnhancementUnit(nn.Module):
    def __init__(self, channel, num_class=None):
        super(FeatureEnhancementUnit, self).__init__()
        self.channel = channel
        self.num_class = num_class
        if num_class is not None:
            self.attention_conv = nn.Sequential(
                nn.Conv2d(self.num_class, self.channel, kernel_size=1),
                nn.BatchNorm2d(self.channel),
                nn.Sigmoid()
            )
        self.conv1 = ConvBnAct(self.channel * 2, self.channel, kernel_size=1)
        self.channel_attention = ChannelAttention(self.channel)
        self.conv2 = ConvBnAct(self.channel, self.channel, kernel_size=3, stride=1, padding=1, act=False)

    def forward(self, feature, mask=None):
        feature = self.conv1(feature)
        if mask is not None and self.num_class is not None:
            attention = self.attention_conv(mask)
            feature = feature * (1 + attention)
        feature = self.channel_attention(feature)
        feature = self.conv2(feature)

        return feature


class KnowledgeReviewModule(nn.Module):
    def __init__(self, channel, num_class):
        super(KnowledgeReviewModule, self).__init__()
        self.channel = channel
        self.num_class = num_class
        self.feature_enhancement_unit_1 = FeatureEnhancementUnit(channel=self.channel,
                                                                 num_class=None)
        self.feature_enhancement_unit_2 = FeatureEnhancementUnit(channel=self.channel,
                                                                 num_class=self.num_class)
        self.feature_enhancement_unit_3 = FeatureEnhancementUnit(channel=self.channel,
                                                                 num_class=self.num_class)
        self.act = nn.ReLU(inplace=True)
        self.fusion_conv = ConvBnAct(self.channel + self.num_class, channel, kernel_size=3, stride=1, padding=1)

    def forward(self, feature, fine_mask, coarse_mask):
        softmax_fine_mask = torch.softmax(fine_mask, dim=1)
        softmax_coarse_mask = torch.softmax(coarse_mask, dim=1)
        # without attention
        context_1 = self.feature_enhancement_unit_1(feature)
        # reverse attention
        reverse_attention = self.num_class - 1 - softmax_coarse_mask
        context_2 = self.feature_enhancement_unit_2(feature, reverse_attention)
        # uncertainty attention
        uncertainty_attention = torch.abs(softmax_coarse_mask - softmax_fine_mask)
        context_3 = self.feature_enhancement_unit_3(feature, uncertainty_attention)
        # fusion
        feature = self.act(context_1 + context_2 + context_3)
        #
        krm_out = self.fusion_conv(torch.cat([feature, fine_mask], dim=1))

        return krm_out


class TemporalDifferenceModule(nn.Module):
    def __init__(self, channel: int, num_temporal: int = 2):
        super(TemporalDifferenceModule, self).__init__()
        self.channel = channel
        self.num_temporal = num_temporal
        self.temporal_convs = nn.ModuleList()
        self.act = nn.ReLU(inplace=True)
        for i in range(self.num_temporal):
            temporal_conv = nn.Sequential(
                nn.Conv3d(channel, channel, kernel_size=[2, 3, 3], stride=[2, 1, 1], padding=[0, 1, 1]),
                nn.BatchNorm3d(channel)
            )
            self.temporal_convs.append(temporal_conv)

    def forward(self, temporal: list):
        B, C, H, W = temporal[0].size()
        temporal = [temp.view(B, C, 1, H, W) for temp in temporal]
        aggregated_temporal = temporal.copy()
        aggregated_temporal[0] = torch.cat(temporal, dim=2)
        aggregated_temporal[1] = torch.cat(temporal[::-1], dim=2)

        for idx, temporal_conv in enumerate(self.temporal_convs):
            aggregated_temporal[idx] = temporal_conv(aggregated_temporal[idx])

        out_feature = self.act(sum(aggregated_temporal))
        out_feature = out_feature.view(B, C, H, W)

        return out_feature
