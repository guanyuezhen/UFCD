import albumentations as A
from albumentations.pytorch import ToTensorV2

DATASET_CFG_XVIEW2 = {
    'task_type': 'bda',
    'task_cfg': {
        'num_bl_class': 2,
        'num_bda_class': 5,
    },
    'data_cfg': {
        'data_root': '/root/data1/data/BDA/xView2',
        'image_size': [1024, 1024],
        'color_map': [[0, 0, 0], [180, 169, 150], [244, 137, 36], [12, 185, 193], [248, 90, 64]],
        'classes': ['background', 'no-damage', 'minor-damage', 'major-damage', 'destroyed'],
    },
    'train_cfg': {
        'sub_set': 'train',
        'transform': {
            'transforms_for_all': A.Compose(
                [
                    A.Flip(p=0.5),
                    A.RandomRotate90(p=0.5),
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
            ], additional_targets={'image1': 'image', 'mask1': 'mask'})
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
            ], additional_targets={'image1': 'image', 'mask1': 'mask'})
        },
    },
}
