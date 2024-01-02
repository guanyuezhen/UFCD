

SECOND_CFG = {
    'data_cfg': {
        'data_root': '/mnt/disk_d/Change Detection/Datasets_SCD/SECOND_rgb',
        'train_dataset': 'train',
        'val_dataset': 'val',
        'test_dataset': 'val',
        'image_size': [512, 512],
        'color_map': [[255, 255, 255], [0, 0, 255], [128, 128, 128], [0, 128, 0], [0, 255, 0], [128, 0, 0],
                      [255, 0, 0]],
        'classes': ['unchanged', 'water', 'ground', 'low vegetation', 'tree', 'building', 'sports field'],
    },
    'task_type': 'scd',
    'task_cfg': {
        'num_bcd_class': 1,
        'num_scd_class': 7,
    }
}

LANDSAT_CFG = {
    'data_cfg': {
        'data_root': '/mnt/disk_d/Change Detection/Datasets_SCD/LandsatSCD_rgb',
        'train_dataset': 'train',
        'val_dataset': 'val',
        'test_dataset': 'test',
        'image_size': [416, 416],
        'color_map': [[255, 255, 255], [0, 155, 0], [255, 165, 0], [230, 30, 100], [0, 170, 240]],
        'classes': ['No change', 'Farmland', 'Desert', 'Building', 'Water'],
    },
    'task_type': 'scd',
    'task_cfg': {
        'num_bcd_class': 1,
        'num_scd_class': 7,
    }
}

XBD_CFG = {
    'data_cfg': {
        'data_root': '/mnt/disk_d/Change Detection/Datasets_SCD/xView2_rgb',
        'train_dataset': 'train',
        'val_dataset': 'test',
        'test_dataset': 'test',
        'image_size': [1024, 1024],
        'color_map': [[0, 0, 0], [180, 169, 150], [244, 137, 36], [12, 185, 193], [248, 90, 64]],
        'classes': ['background', 'no-damage', 'major-damage', 'destroyed'],
    },
    'task_type': 'bda',
    'task_cfg': {
        'num_bl_class': 1,
        'num_bda_class': 5,
    },
}
