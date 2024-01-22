import torch
import numpy as np
import torch.utils.data
from libs.backbone.get_backbone import get_backbone
from libs.models.scd.a2net.a2net import A2Net
from libs.models.scd.bisrnet.bisrnet import BiSRNet
from libs.models.scd.scannet.scannet import SCanNet
from libs.models.bda.changeos.changeos import ChangeOS
from libs.models.bda.changeosgrm.changeosgrm import ChangeOSGRM
from libs.models.bcd.a2net.a2net import A2NetBCD
from libs.models.bcd.tfigr.tfigr import TFIGR
from libs.configs.method_config import A2Net_CFG, A2Net34_CFG, BISRNET_CFG, SSCDL_CFG, SCANNET_CFG, TED_CFG, \
    CHANGEOS_CFG, A2NetBCD_CFG, TFIGR_CFG, CHANGEOSGRM_CFG
from libs.datasets.scd_dataset import SCDDataset
from libs.datasets.bda_dataset import BDADataset
from libs.datasets.bcd_dataset import BCDDataset
from libs.configs.base.datasets.levir_config import DATASET_CFG_LEVIR
from libs.configs.base.datasets.levir_plus_config import DATASET_CFG_LEVIRP
from libs.configs.base.datasets.sysu_config import DATASET_CFG_SYSU
from libs.configs.base.datasets.second_config import DATASET_CFG_SECOND
from libs.configs.base.datasets.landsatcd_config import DATASET_CFG_LANDSAT
from libs.configs.base.datasets.xview2_config import DATASET_CFG_XVIEW2
from libs.configs.base.dataloaders.dataloader_config import DATALOADER_CFG_BS_8, DATALOADER_CFG_BS_16, \
    DATALOADER_CFG_BS_32
from libs.utils.logger.bda_logger import BDALogger
from libs.utils.logger.scd_logger import SCDLogger
from libs.utils.logger.bcd_logger import BCDLogger


def get_model_dataset_by_name(cmd_cfg):
    METHOD_CFG_SET = {
        'TFIGR': TFIGR_CFG,
        'A2NetBCD': A2NetBCD_CFG,
        'A2Net': A2Net_CFG,
        'A2Net34': A2Net34_CFG,
        'BiSRNet': BISRNET_CFG,
        'SCanNet': SCANNET_CFG,
        'SSCDL': SSCDL_CFG,
        'TED': TED_CFG,
        'ChangeOS': CHANGEOS_CFG,
        'ChangeOS-GRM': CHANGEOSGRM_CFG,
    }
    METHOD_SET = {
        'TFIGR': TFIGR,
        'A2NetBCD': A2NetBCD,
        'A2Net': A2Net,
        'A2Net34': A2Net,
        'BiSRNet': BiSRNet,
        'SCanNet': SCanNet,
        'SSCDL': BiSRNet,
        'TED': SCanNet,
        'ChangeOS': ChangeOS,
        'ChangeOS-GRM': ChangeOSGRM,
    }
    DATA_CFG_SET = {
        'SECOND': DATASET_CFG_SECOND,
        'LandsatSCD': DATASET_CFG_LANDSAT,
        'xview2': DATASET_CFG_XVIEW2,
        'LEVIR': DATASET_CFG_LEVIR,
        'LEVIR+': DATASET_CFG_LEVIRP,
        'SYSU': DATASET_CFG_SYSU,
    }
    DATA_SET = {
        'SECOND': SCDDataset,
        'LandsatSCD': SCDDataset,
        'xview2': BDADataset,
        'LEVIR': BCDDataset,
        'LEVIR+': BCDDataset,
        'SYSU': BCDDataset,
    }
    DATALOADER_CFG_SET = {
        'bs_8': DATALOADER_CFG_BS_8,
        'bs_16': DATALOADER_CFG_BS_16,
        'bs_32': DATALOADER_CFG_BS_32
    }
    LOGGER_SET = {
        'bda': BDALogger,
        'scd': SCDLogger,
        'bcd': BCDLogger,
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
                                         train_cfg=data_cfg['train_cfg']
                                         )
        train_loader = torch.utils.data.DataLoader(
            dataset=train_data,
            **dataloader_cfg['train']
        )
        val_data = DATA_SET[data_name](data_cfg=data_cfg['data_cfg'],
                                       train_cfg=data_cfg['val_cfg']
                                       )
        val_loader = torch.utils.data.DataLoader(
            dataset=val_data,
            **dataloader_cfg['val']
        )
        test_data = DATA_SET[data_name](data_cfg=data_cfg['data_cfg'],
                                        train_cfg=data_cfg['test_cfg']
                                        )
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

        optimizer_cfg['max_epoch'] = int(np.ceil(optimizer_cfg['max_iter'] / len(train_loader)))
        optimizer_cfg['max_iter'] = optimizer_cfg['max_epoch'] * len(train_loader)

        return logger, model, train_loader, val_loader, test_loader, optimizer, scaler, optimizer_cfg, task_type, task_cfg

    else:
        test_data = DATA_SET[data_name](data_cfg=data_cfg['data_cfg'],
                                        train_cfg=data_cfg['test_cfg']
                                        )
        test_loader = torch.utils.data.DataLoader(
            dataset=test_data,
            **dataloader_cfg['test']
        )

        return logger, model, test_loader, task_type, task_cfg
