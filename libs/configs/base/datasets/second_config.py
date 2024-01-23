import albumentations as A
from albumentations.pytorch import ToTensorV2

DATASET_CFG_SECOND = {
    'task_type': 'scd',
    'task_cfg': {
        'num_bcd_class': 2,
        'num_scd_class': 7,
    },
    'data_cfg': {
        'data_root': '/mnt/disk_d/Change Detection/UFCD-data/data/SCD/SECOND',
        'image_size': [512, 512],
        'color_map': [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0],
                      [255, 0, 0]],
        'classes': ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field'],
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
                additional_targets={'image1': 'image', 'mask1': 'mask'}
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
            ], additional_targets={'image1': 'image', 'mask1': 'mask'})
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
            ], additional_targets={'image1': 'image', 'mask1': 'mask'})
        },
    },
    'test_cfg': {
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
            ], additional_targets={'image1': 'image', 'mask1': 'mask'})
        },
    },
}
