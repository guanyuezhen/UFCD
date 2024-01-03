import torch


def custom_collate_fn(batch):
    batched_images = {key: torch.stack([sample['image'][key] for sample in batch], dim=0) for key in batch[0]['image']}
    batched_labels = {key: torch.stack([sample['label'][key] for sample in batch], dim=0) for key in batch[0]['label']}
    batched_image_names = [sample['image_name'] for sample in batch]

    return {'image': batched_images, 'label': batched_labels, 'image_name': batched_image_names}


DATALOADER_CFG_BS_8 = {
    'train': {
        'batch_size': 8,
        'shuffle': True,
        'num_workers': 4,
        'pin_memory': False,
        'drop_last': True,
        'collate_fn': custom_collate_fn,
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
