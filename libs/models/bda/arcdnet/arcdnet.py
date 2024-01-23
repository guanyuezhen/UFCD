import torch.nn as nn
import torch.nn.functional as F
from .modules import TemporalDifferenceModule
from .encoder import FeaturePyramidNetworks
from .decoder import KnowledgeReviewDecoder, UncertaintyEstimationDecoder


class ARCDNet(nn.Module):
    def __init__(self,
                 context_encoder,
                 in_channels: list = None,
                 channel: int = 64,
                 num_bl_class: int = 2,
                 num_bda_class: int = 5
                 ):
        super(ARCDNet, self).__init__()
        if in_channels is None:
            in_channels = [64, 128, 256, 512]
        if len(in_channels) > 4:
            in_channels = in_channels[1:]

        self.context_encoder = context_encoder
        self.in_channels = in_channels
        self.channel = channel
        self.num_bl_class = num_bl_class
        self.num_bda_class = num_bda_class

        self.location_fpn = FeaturePyramidNetworks(self.in_channels, self.channel)
        self.damage_fpn = FeaturePyramidNetworks(self.in_channels, self.channel)

        self.temporal_difference_convs = nn.ModuleList()
        for i in range(len(self.in_channels)):
            td_conv = TemporalDifferenceModule(self.channel)
            self.temporal_difference_convs.append(td_conv)

        self.online_uncertainty_learning_decoder_loc = UncertaintyEstimationDecoder(self.channel)
        self.knowledge_review_decoder_loc = KnowledgeReviewDecoder(self.channel, len(self.in_channels),
                                                                   self.num_bl_class)

        self.online_uncertainty_learning_decoder_cls = UncertaintyEstimationDecoder(self.channel)
        self.knowledge_review_decoder_cls = KnowledgeReviewDecoder(self.channel, len(self.in_channels),
                                                                   self.num_bda_class)

    def forward(self, x1, x2):
        x1_1, x1_2, x1_3, x1_4, x1_5 = self.context_encoder(x1)
        x2_1, x2_2, x2_3, x2_4, x2_5 = self.context_encoder(x2)

        x1_features = self.location_fpn([x1_2, x1_3, x1_4, x1_5])
        x2_features = self.damage_fpn([x2_2, x2_3, x2_4, x2_5])

        temporal_difference_features = []
        for idx, td_conv in enumerate(self.temporal_difference_convs):
            td_feature = td_conv([x1_features[idx], x2_features[idx]])
            temporal_difference_features.append(td_feature)

        uncertainty_mask_loc, uncertainty_feature_loc = self.online_uncertainty_learning_decoder_loc(
            x1, x1_features[0])

        uncertainty_mask_cls, uncertainty_feature_cls = self.online_uncertainty_learning_decoder_cls(
            x2, temporal_difference_features[0])

        mask_loc = self.knowledge_review_decoder_loc(x1_features, uncertainty_feature_loc)
        mask_cls = self.knowledge_review_decoder_cls(temporal_difference_features, uncertainty_feature_cls)

        uncertainty_mask_loc = F.interpolate(uncertainty_mask_loc, size=x1.size()[2:], mode='bilinear',
                                             align_corners=True)
        uncertainty_mask_cls = F.interpolate(uncertainty_mask_cls, size=x1.size()[2:], mode='bilinear',
                                             align_corners=True)

        mask_loc = [F.interpolate(mask, size=x1.size()[2:], mode='bilinear', align_corners=True)
                    for mask in mask_loc]
        mask_cls = [F.interpolate(mask, size=x1.size()[2:], mode='bilinear', align_corners=True)
                    for mask in mask_cls]

        prediction = {
            'loc_mask': mask_loc,
            'cls_mask': mask_cls,
            'uncertainty_mask_loc': [uncertainty_mask_loc],
            'uncertainty_mask_cls': [uncertainty_mask_cls],
        }

        return prediction
