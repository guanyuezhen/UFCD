import albumentations as A
from albumentations.pytorch import ToTensorV2

DATASET_CFG_LEVIRP = {
    'task_type': 'bcd',
    'task_cfg': {
        'num_bcd_class': 2,
    },
    'data_cfg': {
        'data_root': '/mnt/disk_d/Change Detection/UFCD-data/data/BCD/LEVIR+',
        'color_map': [[0, 0, 0], [255, 255, 255]],
        'classes': ['unchanged', 'change'],
        'image_size': [1024, 1024],
    },
    'train_cfg': {
        'sub_set': 'train',
        'transform': {
            'transforms_for_all': A.Compose(
                [
                    A.Flip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.Rotate(45, p=0.5),
                ],
                additional_targets={'image1': 'image'}
            ),
            'transforms_for_pre_image': A.Compose([
                A.ColorJitter(p=0.5)
            ]),
            'transforms_for_post_image': A.Compose([
                A.ColorJitter(p=0.5)
            ]),
            'normalize_image': A.Compose([
                A.Normalize()
            ], additional_targets={'image1': 'image'}),
            'to_tensor': A.Compose([
                ToTensorV2()
            ], additional_targets={'image1': 'image'}),
        },
    },
    'val_cfg': {
        'sub_set': 'val',
        'transform': {
            'transforms_for_all': None,
            'transforms_for_pre_image': None,
            'transforms_for_post_image': None,
            'normalize_image': A.Compose([
                A.Normalize()
            ], additional_targets={'image1': 'image'}),
            'to_tensor': A.Compose([
                ToTensorV2()
            ], additional_targets={'image1': 'image'}),
        },
    },
    'test_cfg': {
        'sub_set': 'test',
        'transform': {
            'transforms_for_all': None,
            'transforms_for_pre_image': None,
            'transforms_for_post_image': None,
            'normalize_image': A.Compose([
                A.Normalize()
            ], additional_targets={'image1': 'image'}),
            'to_tensor': A.Compose([
                ToTensorV2()
            ], additional_targets={'image1': 'image'}),
        },
    },
}
