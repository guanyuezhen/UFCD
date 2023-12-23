import torch
import torch.nn as nn
import torch.nn.functional as F
from .CSWin_Transformer import mit


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class ResBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(ResBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels_high, in_channels_low, out_channels, scale_ratio=1):
        super(_DecoderBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels_high, in_channels_high, kernel_size=2, stride=2)
        in_channels = in_channels_high + in_channels_low // scale_ratio
        self.transit = nn.Sequential(
            conv1x1(in_channels_low, in_channels_low // scale_ratio),
            nn.BatchNorm2d(in_channels_low // scale_ratio),
            nn.ReLU(inplace=True))
        self.decode = nn.Sequential(
            conv3x3(in_channels, out_channels),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x, low_feat):
        x = self.up(x)
        low_feat = self.transit(low_feat)
        x = torch.cat([x, low_feat], dim=1)
        x = self.decode(x)
        return x


class SCanNet(nn.Module):
    def __init__(self, context_encoder, in_channels=None, is_scannet=False,
                 num_bc_class=1, num_sc_class=7, input_size=512):
        super(SCanNet, self).__init__()
        if in_channels is None:
            in_channels = [64, 64, 128, 256, 512]
        de_channel_c2 = 64
        de_channel_c5 = 128
        feat_size = input_size // 4
        self.context_encoder = context_encoder
        self.is_scannet = is_scannet
        self.feature_extraction_c5 = nn.Sequential(
            nn.Conv2d(in_channels[4], de_channel_c5, kernel_size=1, bias=False),
            nn.BatchNorm2d(de_channel_c5),
            nn.ReLU()
        )
        if is_scannet:
            self.transformer = mit(img_size=feat_size, in_chans=de_channel_c5 * 3, embed_dim=de_channel_c5 * 3)

        self.Dec1 = _DecoderBlock(de_channel_c5, de_channel_c2, de_channel_c5)
        self.Dec2 = _DecoderBlock(de_channel_c5, de_channel_c2, de_channel_c5)
        self.classifier1 = nn.Conv2d(de_channel_c5, num_sc_class, kernel_size=1)
        self.classifier2 = nn.Conv2d(de_channel_c5, num_sc_class, kernel_size=1)
        self.resCD = self._make_layer(ResBlock, de_channel_c5 * 2, de_channel_c5, 6, stride=1)
        self.DecCD = _DecoderBlock(de_channel_c5, de_channel_c5, de_channel_c5, scale_ratio=2)
        self.classifierCD = nn.Sequential(
            nn.Conv2d(de_channel_c5, de_channel_c5 // 2, kernel_size=1),
            nn.BatchNorm2d(de_channel_c5 // 2),
            nn.ReLU(),
            nn.Conv2d(de_channel_c5 // 2, num_bc_class, kernel_size=1)
        )

        # for param in self.context_encoder.parameters():
        #     param.requires_grad = False

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, t1, t2):
        t1_c1, t1_c2, t1_c3, t1_c4, t1_c5 = self.context_encoder(t1)
        t2_c1, t2_c2, t2_c3, t2_c4, t2_c5 = self.context_encoder(t2)
        #
        t1_c5 = self.feature_extraction_c5(t1_c5)
        t2_c5 = self.feature_extraction_c5(t2_c5)
        t1_ = self.Dec1(t1_c5, t1_c2)
        t2_ = self.Dec2(t2_c5, t2_c2)
        #
        td_c5 = self.resCD(torch.cat([t1_c5, t2_c5], dim=1))
        td_c2 = torch.cat([t1_c2, t2_c2], dim=1)
        td_ = self.DecCD(td_c5, td_c2)
        #
        if self.is_scannet:
            x = torch.cat([t1_, t2_, td_], dim=1)
            x = self.transformer(x)
            t1_, t2_, td_ = torch.chunk(x, 3, dim=1)

        #
        mask_bc = self.classifierCD(td_)
        mask_t1 = self.classifier1(t1_)
        mask_t2 = self.classifier2(t2_)

        mask_bc = F.interpolate(mask_bc, size=t1.size()[2:], mode='bilinear', align_corners=True)
        mask_t1 = F.interpolate(mask_t1, size=t1.size()[2:], mode='bilinear', align_corners=True)
        mask_t2 = F.interpolate(mask_t2, size=t1.size()[2:], mode='bilinear', align_corners=True)

        return mask_t1, mask_t2, mask_bc


def get_model(context_encoder, in_channels=None, is_scannet=False, num_bc_class=1, num_sc_class=7, input_size=512):
    model = SCanNet(context_encoder=context_encoder, in_channels=in_channels, is_scannet=is_scannet,
                    num_bc_class=num_bc_class, num_sc_class=num_sc_class, input_size=input_size)

    return model
