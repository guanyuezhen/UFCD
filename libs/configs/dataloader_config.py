
DATALOADER_CFG_BS_8 = {
    'train': {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': True,
    },
    'val': {
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': False,
    },
    'test': {
        'batch_size': 8,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': False,
    }
}

DATALOADER_CFG_BS_16 = {
    'train': {
        'batch_size': 16,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': True,
    },
    'val': {
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': False,
    },
    'test': {
        'batch_size': 16,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': False,
    }
}

DATALOADER_CFG_BS_32 = {
    'train': {
        'batch_size': 32,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': True,
    },
    'val': {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': False,
    },
    'test': {
        'batch_size': 32,
        'shuffle': False,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': False,
    }
}