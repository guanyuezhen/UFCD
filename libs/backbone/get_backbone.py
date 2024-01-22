import timm
from timm.models.resnet import _cfg as resnet_cfg
from timm.models.efficientnet import _cfg as efficient_cfg


def get_backbone(backbone_name='resnet18d', output_stride=32):
    if backbone_name == 'resnet18d':
        encoder_config = resnet_cfg(url='', file='./libs/backbone/resnet18d_ra2-48a79e06.pth')
        context_encoder = timm.create_model('resnet18d', features_only=True,
                                            output_stride=output_stride, pretrained=True,
                                            pretrained_cfg=encoder_config)
        in_channels = [64, 64, 128, 256, 512]

        return context_encoder, in_channels

    elif backbone_name == 'resnet34d':
        encoder_config = resnet_cfg(url='', file='./libs/backbone/resnet34d_ra2-f8dcfcaf.pth')
        context_encoder = timm.create_model('resnet34d', features_only=True,
                                            output_stride=output_stride, pretrained=True,
                                            pretrained_cfg=encoder_config)
        in_channels = [64, 64, 128, 256, 512]

        return context_encoder, in_channels

    elif backbone_name == 'mobilenetv2':
        encoder_config = efficient_cfg(url='', file='./libs/backbone/mobilenetv2_100_ra-b33bc2c4.pth')
        context_encoder = timm.create_model('mobilenetv2_100', features_only=True, pretrained=True,
                                            pretrained_cfg=encoder_config)
        in_channels = [16, 24, 32, 96, 320]

        return context_encoder, in_channels

    elif backbone_name == 'mobilevitv2':
        encoder_config = efficient_cfg(url='', file='./libs/backbone/mobilenetv2_100_ra-b33bc2c4.pth')
        context_encoder = timm.create_model('mobilenetv2_100', features_only=True, pretrained=True,
                                            pretrained_cfg=encoder_config)
        in_channels = [16, 24, 32, 96, 320]

        return context_encoder, in_channels
