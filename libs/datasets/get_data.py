import torch
import torch.utils.data
from .dataset import Dataset


def get_data_by_name(data_name='SECOND', batch_size=32, num_workers=4, is_train=True):

    if data_name == 'SECOND':
        data_root = '/mnt/disk_d/Change Detection/Datasets_SCD/SECOND_rgb'
        train_dataset = 'train'
        val_dataset = 'val'
        test_dataset = 'val'
        num_classes = 7
    elif data_name == 'LandsatSCD':
        data_root = '/mnt/disk_d/Change Detection/Datasets_SCD/LandsatSCD_rgb'
        train_dataset = 'train'
        val_dataset = 'val'
        test_dataset = 'test'
        num_classes = 5
    else:
        raise TypeError('%s has not defined' % data_name)

    if is_train:
        train_data = Dataset(dataset=train_dataset, file_name=data_name, data_root=data_root, transform=True)
        train_loader = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=False, drop_last=True
        )
        val_data = Dataset(dataset=val_dataset, file_name=data_name, data_root=data_root, transform=False)
        val_loader = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
        )
        test_data = Dataset(dataset=test_dataset, file_name=data_name, data_root=data_root, transform=False)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
        )
        return train_loader, val_loader, test_loader
    else:
        test_data = Dataset(dataset=test_dataset, file_name=data_name, data_root=data_root, transform=False)
        test_loader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=False
        )
        return test_loader
