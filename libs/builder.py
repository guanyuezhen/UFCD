import timm
import torch
import torch.utils.data
from timm.models.resnet import _cfg as resnet_cfg
from timm.models.efficientnet import _cfg as efficient_cfg
from libs.models.SCD.A2Net.A2Net import A2Net
from libs.models.SCD.BiSRNet.BiSRNet import BiSRNet
from libs.models.SCD.SCanNet.SCanNet import SCanNet
from libs.models.BDA.ChangeOS.ChangeOS import ChangeOS
from libs.configs.method_config import A2Net_CFG, A2Net18_CFG, BISRNET_CFG, SSCDL_CFG, SCANNET_CFG, TED_CFG, \
    CHANGEOS_CFG
from libs.datasets.scd_dataset import SCDDataset
from libs.datasets.bda_dataset import BDADataset
from libs.configs.data_config import SECOND_CFG, LANDSAT_CFG, XBD_CFG
from libs.configs.dataloader_config import DATALOADER_CFG_BS_8, DATALOADER_CFG_BS_16, DATALOADER_CFG_BS_32
from libs.logger.bda_logger import BDALogger
from libs.logger.scd_logger import SCDLogger


def get_model_dataset_by_name(cmd_cfg):
    METHOD_CFG_SET = {
        'A2Net': A2Net_CFG,
        'A2Net18': A2Net18_CFG,
        'BiSRNet': BISRNET_CFG,
        'SCanNet': SCANNET_CFG,
        'SSCDL': SSCDL_CFG,
        'TED': TED_CFG,
        'ChangeOS': CHANGEOS_CFG,
    }
    METHOD_SET = {
        'A2Net': A2Net,
        'A2Net18': A2Net,
        'BiSRNet': BiSRNet,
        'SCanNet': SCanNet,
        'SSCDL': BiSRNet,
        'TED': SCanNet,
        'ChangeOS': ChangeOS,
    }
    DATA_CFG_SET = {
        'SECOND': SECOND_CFG,
        'LandsatSCD': LANDSAT_CFG,
        'xBD': XBD_CFG
    }
    DATA_SET = {
        'SECOND': SCDDataset,
        'LandsatSCD': SCDDataset,
        'xBD': BDADataset
    }
    DATALOADER_CFG_SET = {
        'bs_8': DATALOADER_CFG_BS_8,
        'bs_16': DATALOADER_CFG_BS_16,
        'bs_32': DATALOADER_CFG_BS_32
    }
    LOGGER_SET = {
        'bda': BDALogger,
        'scd': SCDLogger,
    }

    model_name = cmd_cfg.model_name
    data_name = cmd_cfg.data_name
    dataloader_name = cmd_cfg.dataloader_name
    is_train = cmd_cfg.is_train

    data_cfg = DATA_CFG_SET[data_name].copy()
    task_type = data_cfg['task_type']
    task_cfg = data_cfg['task_cfg']
    method_cfg = METHOD_CFG_SET[model_name].copy()
    dataloader_cfg = DATALOADER_CFG_SET[dataloader_name]
    optimizer_cfg = method_cfg['optimizer_cfg']
    if 'image_size' in method_cfg['head_cfg']:
        method_cfg['head_cfg']['image_size'] = data_cfg['data_cfg']['image_size']

    context_encoder, in_channels = get_backbone(**method_cfg['backbone_cfg'])
    model = METHOD_SET[model_name](
        context_encoder=context_encoder,
        in_channels=in_channels,
        **method_cfg['head_cfg'],
        **data_cfg['task_cfg'],
    )
    logger = LOGGER_SET[task_type](cmd_cfg=cmd_cfg)

    if is_train:
        train_data = DATA_SET[data_name](data_cfg=data_cfg['data_cfg'],
                                         dataset=data_cfg['data_cfg']['train_dataset'],
                                         transform=True)
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            **dataloader_cfg['train']
        )
        val_data = DATA_SET[data_name](data_cfg=data_cfg['data_cfg'],
                                       dataset=data_cfg['data_cfg']['val_dataset'],
                                       transform=False)
        val_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            **dataloader_cfg['val']
        )
        test_data = DATA_SET[data_name](data_cfg=data_cfg['data_cfg'],
                                        dataset=data_cfg['data_cfg']['test_dataset'],
                                        transform=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            **dataloader_cfg['test']
        )

        optimizer = torch.optim.AdamW(
            model.parameters(),
            method_cfg['optimizer_cfg']['lr'],
            (0.9, 0.999),
            weight_decay=1e-2
        )
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        return logger, model, train_loader, val_loader, test_loader, optimizer, scaler, optimizer_cfg, task_type, task_cfg

    else:
        test_data = DATA_SET[data_name](data_cfg=data_cfg['data_cfg'],
                                        dataset=data_cfg['data_cfg']['test_dataset'],
                                        transform=False)
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            **dataloader_cfg['test']
        )

        return logger, model, test_loader, task_type, task_cfg


def get_backbone(backbone_name='resnet18', output_stride=32):
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
