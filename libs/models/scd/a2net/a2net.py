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


class NeighborFeatureAggregation(nn.Module):
    def __init__(self, in_channels=None, channel=64):
        super(NeighborFeatureAggregation, self).__init__()
        if in_channels is None:
            in_channels = [24, 32, 96, 320]
        self.in_channels = in_channels
        self.channel = channel
        self.hid_chanel = channel // 2
        self.fuse_modules = nn.ModuleList()
        for idx, in_channel in enumerate(self.in_channels):
            if idx == 0:
                fuse_module = FeatureFusionModule(in_channel, self.in_channels[idx:idx + 2], self.hid_chanel,
                                                  self.channel)
            elif idx == (len(self.in_channels) - 1):
                fuse_module = FeatureFusionModule(in_channel, self.in_channels[idx - 1:idx + 1], self.hid_chanel,
                                                  self.channel)
            else:
                fuse_module = FeatureFusionModule(in_channel, self.in_channels[idx - 1:idx + 2], self.hid_chanel,
                                                  self.channel)
            self.fuse_modules.append(fuse_module)

    def forward(self, inputs: list):
        outputs = []
        for idx, fuse_module in enumerate(self.fuse_modules):
            if idx == 0:
                aggregated_feature = fuse_module(inputs[idx: idx + 2], inputs[idx])
            elif idx == (len(self.in_channels) - 1):
                aggregated_feature = fuse_module(inputs[idx - 1: idx + 1], inputs[idx])
            else:
                aggregated_feature = fuse_module(inputs[idx - 1: idx + 2], inputs[idx])
            outputs.append(aggregated_feature)

        return outputs


class FeatureFusionModule(nn.Module):
    def __init__(self, identity_feature_channel, in_channels, hid_channel, out_channel):
        super(FeatureFusionModule, self).__init__()
        self.identity_feature_channel = identity_feature_channel
        self.in_channels = in_channels
        self.hid_channel = hid_channel
        self.out_channel = out_channel
        self.convs = nn.ModuleList()
        for in_channel in self.in_channels:
            conv = ConvBnAct(in_channel, self.hid_channel, kernel_size=3, stride=1, padding=1)
            self.convs.append(conv)

        self.fuse_conv = nn.Sequential(
            ConvBnAct(self.hid_channel * len(self.in_channels), self.out_channel, kernel_size=3, stride=1, padding=1),
            ConvBnAct(out_channel, self.out_channel, kernel_size=3, stride=1, padding=1, act=False)
        )
        self.identity_conv = nn.Conv2d(self.identity_feature_channel, self.out_channel, kernel_size=1)
        self.act = nn.ReLU(inplace=True)

    @staticmethod
    def up(a, b):
        out = F.interpolate(a, b.size()[2:], mode='bilinear', align_corners=True)
        return out

    @staticmethod
    def down(a, b):
        out = F.adaptive_max_pool2d(a, b.size()[2:])
        return out

    def forward(self, feature_list: list, identity_feature):
        for idx, conv in enumerate(self.convs):
            if feature_list[idx].size()[2] > identity_feature.size()[2]:
                feature_list[idx] = self.down(feature_list[idx], identity_feature)
                feature_list[idx] = conv(feature_list[idx])
            elif feature_list[idx].size()[2] < identity_feature.size()[2]:
                feature_list[idx] = conv(feature_list[idx])
                feature_list[idx] = self.up(feature_list[idx], identity_feature)
            else:
                feature_list[idx] = conv(feature_list[idx])

        fused_feature = self.fuse_conv(torch.cat(feature_list, dim=1))
        out_feature = self.act(fused_feature + self.identity_conv(identity_feature))

        return out_feature


class ProgressiveChangeIdentifyModule(nn.Module):
    def __init__(self, channel, dilation_sizes=None):
        super(ProgressiveChangeIdentifyModule, self).__init__()
        if dilation_sizes is None:
            dilation_sizes = [7, 5, 3, 1]
        self.channel = channel
        self.dilation_sizes = dilation_sizes
        self.act = nn.ReLU(inplace=True)

        self.identity_convs = nn.ModuleList()
        self.dilation_convs = nn.ModuleList()
        for dilation_size in self.dilation_sizes:
            dilation_conv = ConvBnAct(self.channel, self.channel, kernel_size=3, stride=1,
                                      padding=dilation_size, dilation=dilation_size, act=False)
            identity_conv = nn.Conv2d(self.channel, self.channel, kernel_size=1)
            self.identity_convs.append(identity_conv)
            self.dilation_convs.append(dilation_conv)

    def forward(self, x):
        output = x
        for idx, dilation_conv in enumerate(self.dilation_convs):
            output = self.act(
                self.identity_convs[idx](x) + dilation_conv(output)
            )

        return output


class TemporalFusionModule(nn.Module):
    def __init__(self, channel, num_features, dilation_sizes):
        super(TemporalFusionModule, self).__init__()
        self.channel = channel
        self.num_features = num_features
        self.dilation_sizes = dilation_sizes
        self.temporal_difference_convs = nn.ModuleList()
        for i in range(self.num_features):
            td_conv = ProgressiveChangeIdentifyModule(self.channel, dilation_sizes=self.dilation_sizes)
            self.temporal_difference_convs.append(td_conv)

    def forward(self, x1_features, x2_features):
        td_features = []
        for idx, td_conv in enumerate(self.temporal_difference_convs):
            td_feature = torch.abs(x1_features[idx] - x2_features[idx])
            td_feature = td_conv(td_feature)
            td_features.append(td_feature)

        return td_features


class SupervisedAttentionModule(nn.Module):
    def __init__(self, channel, num_bcd_class):
        super(SupervisedAttentionModule, self).__init__()
        self.channel = channel
        self.num_bcd_class = num_bcd_class
        self.cls = nn.Conv2d(self.channel, self.num_bcd_class, kernel_size=1)
        self.conv_context = nn.Sequential(
            nn.Conv2d(self.num_bcd_class + 1, self.channel, kernel_size=1),
            nn.BatchNorm2d(self.channel),
            nn.Sigmoid()
        )
        self.conv2 = ConvBnAct(self.channel, self.channel, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        mask = self.cls(x)
        mask_f = torch.sigmoid(mask)
        mask_b = 1 - mask_f
        context = torch.cat([mask_f, mask_b], dim=1)
        context = self.conv_context(context)
        x = x * context
        x_out = self.conv2(x)

        return x_out, mask


class Decoder(nn.Module):
    def __init__(self, channel, num_bcd_class, num_features):
        super(Decoder, self).__init__()
        self.channel = channel
        self.num_bcd_class = num_bcd_class
        self.num_features = num_features
        self.sa_modules = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()
        for i in range(0, self.num_features - 1):
            fusion_conv = ConvBnAct(self.channel, self.channel, kernel_size=3, stride=1, padding=1)
            sa_module = SupervisedAttentionModule(self.channel, self.num_bcd_class)
            self.fusion_convs.append(fusion_conv)
            self.sa_modules.append(sa_module)

        self.cls = nn.Conv2d(self.channel, self.num_bcd_class, kernel_size=1)

    @staticmethod
    def up_add(a, b):
        out = F.interpolate(a, b.size()[2:], mode='bilinear', align_corners=True) + b
        return out

    def forward(self, inputs: list):
        reversed_inputs = inputs[::-1]
        masks = []
        for i in range(len(reversed_inputs) - 1):
            reversed_inputs[i], mask_i = self.sa_modules[i](reversed_inputs[i])
            masks.append(mask_i)
            reversed_inputs[i + 1] = self.up_add(reversed_inputs[i], reversed_inputs[i + 1])
            reversed_inputs[i + 1] = self.fusion_convs[i](reversed_inputs[i + 1])

        mask_i = self.cls(reversed_inputs[-1])
        masks.append(mask_i)
        reversed_masks = masks[::-1]

        return reversed_masks


class SegmentationDecoder(nn.Module):
    def __init__(self, channel, num_scd_class, num_features):
        super(SegmentationDecoder, self).__init__()
        self.channel = channel
        self.num_scd_class = num_scd_class
        self.num_features = num_features
        self.sa_modules = nn.ModuleList()
        self.fusion_convs = nn.ModuleList()
        for i in range(0, self.num_features - 1):
            fusion_conv = ConvBnAct(self.channel, self.channel, kernel_size=3, stride=1, padding=1)
            self.fusion_convs.append(fusion_conv)

        self.cls = nn.Conv2d(self.channel, self.num_scd_class, kernel_size=1)

    @staticmethod
    def up_add(a, b):
        out = F.interpolate(a, b.size()[2:], mode='bilinear', align_corners=True) + b
        return out

    def forward(self, inputs: list):
        reversed_inputs = inputs[::-1]
        for i in range(len(reversed_inputs) - 1):
            reversed_inputs[i + 1] = self.up_add(reversed_inputs[i], reversed_inputs[i + 1])
            reversed_inputs[i + 1] = self.fusion_convs[i](reversed_inputs[i + 1])

        mask = self.cls(reversed_inputs[-1])

        return mask


class A2Net(nn.Module):
    def __init__(self,
                 context_encoder,
                 in_channels=None,
                 channel=64,
                 dilation_sizes=None,
                 num_bcd_class=1,
                 num_scd_class=7,
                 ):
        super(A2Net, self).__init__()
        if dilation_sizes is None:
            dilation_sizes = [7, 5, 3, 1]
        if in_channels is None:
            in_channels = [24, 32, 96, 320]
        if len(in_channels) > 4:
            in_channels = in_channels[1:]
        self.context_encoder = context_encoder
        self.in_channels = in_channels
        self.channel = channel
        self.dilation_sizes = dilation_sizes
        self.num_bcd_class = num_bcd_class
        self.num_scd_class = num_scd_class
        self.swa = NeighborFeatureAggregation(self.in_channels, self.channel)
        self.tfm = TemporalFusionModule(self.channel, len(self.in_channels), self.dilation_sizes)
        self.decoder = Decoder(self.channel, self.num_bcd_class, len(self.in_channels))
        self.segmentation_decoder = SegmentationDecoder(self.channel, self.num_scd_class, len(self.in_channels))

    def forward(self, x1, x2):
        # forward backbone resnet
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.context_encoder(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.context_encoder(x2)
        # aggregation
        [x1_2, x1_3, x1_4, x1_5] = self.swa([x1_2, x1_3, x1_4, x1_5])
        [x2_2, x2_3, x2_4, x2_5] = self.swa([x2_2, x2_3, x2_4, x2_5])
        # temporal fusion
        [c2, c3, c4, c5] = self.tfm([x1_2, x1_3, x1_4, x1_5], [x2_2, x2_3, x2_4, x2_5])
        # fpn for change detection
        masks = self.decoder([c2, c3, c4, c5])
        seg_mask_t1 = self.segmentation_decoder([x1_2, x1_3, x1_4, x1_5])
        seg_mask_t2 = self.segmentation_decoder([x2_2, x2_3, x2_4, x2_5])
        seg_mask_t1 = F.interpolate(seg_mask_t1, size=x1.size()[2:], mode='bilinear', align_corners=True)
        seg_mask_t2 = F.interpolate(seg_mask_t2, size=x1.size()[2:], mode='bilinear', align_corners=True)
        for i in range(len(masks)):
            masks[i] = F.interpolate(masks[i], size=x1.size()[2:], mode='bilinear', align_corners=True)
        prediction = {
            'change_mask': masks,
            'pre_mask': [seg_mask_t1],
            'post_mask': [seg_mask_t2],
        }

        return prediction
