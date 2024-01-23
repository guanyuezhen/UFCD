import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import KnowledgeReviewModule, ConvBnAct, TemporalDifferenceModule, FeatureFusionModule


class KnowledgeReviewDecoder(nn.Module):
    def __init__(self, channel, num_features, num_class=2, num_features_for_krm=2):
        super(KnowledgeReviewDecoder, self).__init__()
        self.channel = channel
        self.num_features = num_features
        self.num_class = num_class
        self.num_features_for_krm = num_features_for_krm
        self.mask_generation_convs = nn.ModuleList()
        self.knowledge_review_modules = nn.ModuleList()
        for i in range(self.num_features_for_krm):
            mask_generation_conv = nn.Conv2d(self.channel, self.num_class, kernel_size=1)
            self.mask_generation_convs.append(mask_generation_conv)

        for i in range(self.num_features_for_krm - 1):
            knowledge_review_module = KnowledgeReviewModule(self.channel, self.num_class)
            self.knowledge_review_modules.append(knowledge_review_module)

        self.fusion_conv = ConvBnAct(self.channel * (self.num_features + 1), self.channel, kernel_size=3, stride=1,
                                     padding=1)
        self.final_mask_generation_conv = nn.Conv2d(self.channel, self.num_class, kernel_size=1)

    def forward(self, change_features: list, uncertainty_feature):
        for i in range(len(change_features) - 1, 0, -1):
            change_features[i] = F.interpolate(
                change_features[i], change_features[0].size()[2:], mode='bilinear', align_corners=True)
        mask = []

        features_for_knowledge_review = [change_features[0]] + change_features[-self.num_features_for_krm-1:]

        for i in range(self.num_features_for_krm):
            mask_i = self.mask_generation_convs[i](features_for_knowledge_review[i])
            mask.append(mask_i)

        for i in range(self.num_features_for_krm - 1):
            features_for_knowledge_review[i+1] = self.knowledge_review_modules[i](
                torch.cat([features_for_knowledge_review[0], features_for_knowledge_review[i+1]], dim=1),
                mask[0], mask[i + 1]
            )

        features_for_knowledge_review.append(uncertainty_feature)
        all_features = features_for_knowledge_review + change_features[1:-self.num_features_for_krm-1]

        out_feature = self.fusion_conv(torch.cat(all_features, dim=1))
        mask_final = self.final_mask_generation_conv(out_feature)
        mask.insert(0, mask_final)

        return mask


class UncertaintyEstimationDecoder(nn.Module):
    def __init__(self, channel):
        super(UncertaintyEstimationDecoder, self).__init__()
        self.channel = channel
        self.feature_extraction_conv = nn.Sequential(
            ConvBnAct(3, self.channel // 2, kernel_size=3, stride=2, padding=1),
            ConvBnAct(self.channel // 2, self.channel, kernel_size=3, stride=2, padding=1),
            ConvBnAct(self.channel, self.channel, kernel_size=3, stride=1, padding=1)
        )
        self.temporal_difference_module = TemporalDifferenceModule(self.channel)
        self.feature_fusion_module = FeatureFusionModule(self.channel)
        self.uncertainty_generation = nn.Conv2d(channel, 1, kernel_size=1)

    def forward(self, t1, t2, c2):
        t1_feature = self.feature_extraction_conv(t1)
        t2_feature = self.feature_extraction_conv(t2)

        temporal_difference = self.temporal_difference_module([t1_feature, t2_feature])

        uncertainty_feature = self.feature_fusion_module(temporal_difference, c2)

        uncertainty_mask = self.uncertainty_generation(uncertainty_feature)

        return uncertainty_mask, uncertainty_feature
