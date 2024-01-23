import torch.nn as nn
import torch.nn.functional as F
from .modules import TemporalDifferenceModule
from .encoder import FeaturePyramidNetworks
from .decoder import KnowledgeReviewDecoder, UncertaintyEstimationDecoder


class ARCDNetBCD(nn.Module):
    def __init__(self,
                 context_encoder,
                 in_channels=None,
                 channel=64,
                 num_bcd_class=2,
                 ):
        super(ARCDNetBCD, self).__init__()
        if in_channels is None:
            in_channels = [64, 128, 256, 512]
        if len(in_channels) > 4:
            in_channels = in_channels[1:]

        self.context_encoder = context_encoder
        self.in_channels = in_channels
        self.channel = channel
        self.num_bcd_class = num_bcd_class
        self.temporal_difference_convs = nn.ModuleList()
        self.shared_fpn = FeaturePyramidNetworks(self.in_channels, self.channel)
        for i in range(len(self.in_channels)):
            td_conv = TemporalDifferenceModule(self.channel)
            self.temporal_difference_convs.append(td_conv)
        self.online_uncertainty_learning_decoder = UncertaintyEstimationDecoder(self.channel)
        self.knowledge_review_decoder = KnowledgeReviewDecoder(self.channel, len(self.in_channels), self.num_bcd_class)

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.context_encoder(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.context_encoder(x2)

        x1_features = self.shared_fpn([x1_2, x1_3, x1_4, x1_5])
        x2_features = self.shared_fpn([x2_2, x2_3, x2_4, x2_5])

        temporal_difference_features = []
        for idx, td_conv in enumerate(self.temporal_difference_convs):
            td_feature = td_conv([x1_features[idx], x2_features[idx]])
            temporal_difference_features.append(td_feature)

        uncertainty_mask, uncertainty_feature = self.online_uncertainty_learning_decoder(
            x1, x2, temporal_difference_features[0])

        change_mask = self.knowledge_review_decoder(temporal_difference_features, uncertainty_feature)

        uncertainty_mask = F.interpolate(uncertainty_mask, size=x1.size()[2:], mode='bilinear', align_corners=True)

        change_mask = [F.interpolate(mask, size=x1.size()[2:], mode='bilinear', align_corners=True)
                       for mask in change_mask]
        prediction = {
            'change_mask': change_mask,
            'uncertainty_mask': [uncertainty_mask],
        }

        return prediction
