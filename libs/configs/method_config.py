
A2Net_CFG = {
    'backbone_cfg': {
        'backbone_name': 'mobilenetv2',
        'output_stride': 32
    },
    'head_cfg': {
        'channel': 32,
    },
    'optimizer_cfg': {
        'lr': 5e-4,
        'max_epoch': 50,
        'power': 0.9,
        'min_lr': 0.0,
        'warm_up_iter': 1500,
        'warm_up_ratio': 1e-6,
        'lr_factor': 1.0,
    },
}

A2Net18_CFG = {
    'backbone_cfg': {
        'backbone_name': 'resnet18d',
        'output_stride': 32
    },
    'head_cfg': {
        'channel': 32,
    },
    'optimizer_cfg': {
        'lr': 5e-4,
        'max_epoch': 50,
        'power': 0.9,
        'min_lr': 0.0,
        'warm_up_iter': 1500,
        'warm_up_ratio': 1e-6,
        'lr_factor': 1.0,
    },
}

BISRNET_CFG = {
    'backbone_cfg': {
        'backbone_name': 'resnet18d',
        'output_stride': 8
    },
    'head_cfg': {
        'is_bisrnet': True,
        'de_channel_c5': 128,
    },
    'optimizer_cfg': {
        'lr': 5e-4,
        'max_epoch': 50,
        'power': 0.9,
        'min_lr': 0.0,
        'warm_up_iter': 1500,
        'warm_up_ratio': 1e-6,
        'lr_factor': 1.0,
    },
}

SSCDL_CFG = {
    'backbone_cfg': {
        'backbone_name': 'resnet18d',
        'output_stride': 8
    },
    'head_cfg': {
        'is_bisrnet': False,
        'de_channel_c5': 128,
    },
    'optimizer_cfg': {
        'lr': 5e-4,
        'max_epoch': 50,
        'power': 0.9,
        'min_lr': 0.0,
        'warm_up_iter': 1500,
        'warm_up_ratio': 1e-6,
        'lr_factor': 1.0,
    },
}

SCANNET_CFG = {
    'backbone_cfg': {
        'backbone_name': 'resnet18d',
        'output_stride': 8
    },
    'head_cfg': {
        'is_scannet': True,
        'image_size': [512, 512],
        'de_channel_c2': 64,
        'de_channel_c5': 128,
    },
    'optimizer_cfg': {
        'lr': 5e-4,
        'max_epoch': 50,
        'power': 0.9,
        'min_lr': 0.0,
        'warm_up_iter': 1500,
        'warm_up_ratio': 1e-6,
        'lr_factor': 1.0,
    },
}

TED_CFG = {
    'backbone_cfg': {
        'backbone_name': 'resnet18d',
        'output_stride': 8
    },
    'head_cfg': {
        'is_scannet': True,
        'image_size': [512, 512],
        'de_channel_c2': 64,
        'de_channel_c5': 128,
    },
    'optimizer_cfg': {
        'lr': 5e-4,
        'max_epoch': 50,
        'power': 0.9,
        'min_lr': 0.0,
        'warm_up_iter': 1500,
        'warm_up_ratio': 1e-6,
        'lr_factor': 1.0,
    },
}

CHANGEOS_CFG = {
    'backbone_cfg': {
        'backbone_name': 'resnet18d',
        'output_stride': 32
    },
    'head_cfg': {
        'decoder_channel': 64,
    },
    'optimizer_cfg': {
        'lr': 5e-4,
        'max_epoch': 50,
        'power': 0.9,
        'min_lr': 0.0,
        'warm_up_iter': 1500,
        'warm_up_ratio': 1e-6,
        'lr_factor': 1.0,
    },
}
