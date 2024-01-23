import torch.nn as nn
import torch.nn.functional as F
from .modules import ChangeMixin
from .decoder import UPerNet


class ChangeStarUperNet(nn.Module):
    def __init__(self,
                 context_encoder,
                 in_channels=None,
                 channel=256,
                 inner_channel=16,
                 num_bcd_class=2,
                 ):
        super(ChangeStarUperNet, self).__init__()
        if in_channels is None:
            in_channels = [64, 128, 256, 512]
        if len(in_channels) > 4:
            in_channels = in_channels[1:]

        self.context_encoder = context_encoder
        self.in_channels = in_channels
        self.channel = channel
        self.inner_channel = inner_channel
        self.num_bcd_class = num_bcd_class
        self.shared_decoder = UPerNet(self.in_channels, self.channel)
        self.change_mixin = ChangeMixin(self.channel, self.inner_channel, self.num_bcd_class)

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.context_encoder(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.context_encoder(x2)

        x1_feature = self.shared_decoder([x1_2, x1_3, x1_4, x1_5])
        x2_feature = self.shared_decoder([x2_2, x2_3, x2_4, x2_5])

        change_mask = self.change_mixin(x1_feature, x2_feature)

        change_mask = [F.interpolate(mask, size=x1.size()[2:], mode='bilinear', align_corners=True)
                       for mask in change_mask]
        prediction = {
            'change_mask': change_mask,
        }

        return prediction
