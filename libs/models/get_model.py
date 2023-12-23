import timm
from timm.models.resnet import _cfg as resnet_cfg
from timm.models.efficientnet import _cfg as efficient_cfg


def get_model_by_name(model_name, num_classes, in_width):
    if model_name == 'A2Net':
        from libs.models.A2Net.A2Net import get_model
        context_encoder, in_channels = get_backbone(backbone_name='mobilenetv2', output_stride=32)
        return get_model(context_encoder=context_encoder, in_channels=in_channels,
                         channel=32, num_bc_class=1, num_sc_class=num_classes)
    elif model_name == 'A2Net18':
        from libs.models.A2Net.A2Net import get_model
        context_encoder, in_channels = get_backbone(backbone_name='resnet18', output_stride=8)
        return get_model(context_encoder=context_encoder, in_channels=in_channels,
                         channel=64, num_bc_class=1, num_sc_class=num_classes)
    elif model_name == 'BiSRNet':
        from libs.models.BiSRNet.BiSRNet import get_model
        context_encoder, in_channels = get_backbone(backbone_name='resnet18', output_stride=8)
        return get_model(context_encoder, in_channels=None, is_bisrnet=True,
                         num_bc_class=1, num_sc_class=num_classes)
    elif model_name == 'SCanNet':
        from libs.models.SCanNet.SCanNet import get_model
        context_encoder, in_channels = get_backbone(backbone_name='resnet18', output_stride=8)
        return get_model(context_encoder=context_encoder, in_channels=in_channels, is_scannet=True,
                         num_bc_class=1, num_sc_class=num_classes, input_size=in_width)
    elif model_name == 'SSCDL':
        from libs.models.BiSRNet.BiSRNet import get_model
        context_encoder, in_channels = get_backbone(backbone_name='resnet18', output_stride=8)
        return get_model(context_encoder, in_channels=None, is_bisrnet=False,
                         num_bc_class=1, num_sc_class=num_classes)
    elif model_name == 'TED':
        from libs.models.SCanNet.SCanNet import get_model
        context_encoder, in_channels = get_backbone(backbone_name='resnet18', output_stride=8)
        return get_model(context_encoder=context_encoder, in_channels=in_channels, is_scannet=False,
                         num_bc_class=1, num_sc_class=num_classes, input_size=in_width)
    else:
        raise TypeError('%s has not defined' % model_name)


def get_backbone(backbone_name='resnet18', output_stride=32):
    if backbone_name == 'resnet18':
        encoder_config = resnet_cfg(url='', file='./libs/pretrain/resnet18d_ra2-48a79e06.pth')
        context_encoder = timm.create_model('resnet18d', features_only=True,
                                            output_stride=output_stride, pretrained=True,
                                            pretrained_cfg=encoder_config)
        in_channels = [64, 64, 128, 256, 512]

        return context_encoder, in_channels

    elif backbone_name == 'resnet34':
        encoder_config = resnet_cfg(url='', file='./libs/pretrain/resnet34d_ra2-f8dcfcaf.pth')
        context_encoder = timm.create_model('resnet34d', features_only=True,
                                            output_stride=output_stride, pretrained=True,
                                            pretrained_cfg=encoder_config)
        in_channels = [64, 64, 128, 256, 512]

        return context_encoder, in_channels

    elif backbone_name == 'mobilenetv2':
        encoder_config = efficient_cfg(url='', file='./libs/pretrain/mobilenetv2_100_ra-b33bc2c4.pth')
        context_encoder = timm.create_model('mobilenetv2_100', features_only=True, pretrained=True,
                                            pretrained_cfg=encoder_config)
        in_channels = [16, 24, 32, 96, 320]

        return context_encoder, in_channels
