import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from .base_dataset import BaseDataset


class BDADataset(BaseDataset):
    def __init__(self,
                 data_cfg=None,
                 dataset='train',
                 transform=None
                 ):
        super(BDADataset, self).__init__(data_cfg=data_cfg, transform=transform)
        self.data_root = data_cfg['data_root']
        self.dataset = dataset
        self.image_size = data_cfg['image_size']
        self.color_map = data_cfg['color_map']
        self.classes = data_cfg['classes']
        self.transform = transform

        self.color_map_to_label = np.zeros(256 ** 3)
        for i, cm in enumerate(self.color_map):
            self.color_map_to_label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

        self.file_list = open(self.data_root + '/' + self.dataset + '/' + 'image_list.txt').read().splitlines()
        if transform:
            self.train_transforms_all = A.Compose([
                A.Flip(p=0.5),
                A.Transpose(p=0.5),
                A.Rotate(45, p=0.3),
                A.ShiftScaleRotate(p=0.3),
                A.RandomSizedCrop(min_max_height=(self.image_size[0], self.image_size[1]),
                                  width=self.image_size[0], height=self.image_size[1], w2h_ratio=0.8, p=0.3),
            ], additional_targets={'image1': 'image', 'mask1': 'mask'})
            self.train_transforms_pre_image = A.Compose(
                [A.OneOf([
                    A.GaussNoise(p=1),
                    A.HueSaturationValue(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.Emboss(p=1),
                    A.MotionBlur(p=1),
                ], p=0.8)])
            self.train_transforms_post_image = A.Compose(
                [A.OneOf([
                    A.GaussNoise(p=1),
                    A.HueSaturationValue(p=1),
                    A.RandomBrightnessContrast(p=1),
                    A.RandomGamma(p=1),
                    A.Emboss(p=1),
                    A.MotionBlur(p=1),
                ], p=0.8)])
        self.normalize_image = A.Compose([
            A.Normalize()
        ], additional_targets={'image1': 'image'})
        self.to_tensor = A.Compose([
            ToTensorV2()
        ], additional_targets={'image1': 'image', 'mask1': 'mask'})

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        image_id = self.file_list[idx][:-17]
        pre_image_id = image_id + 'pre_disaster.png'
        post_image_id = image_id + 'post_disaster.png'
        pre_label_id = image_id + 'pre_disaster.png'
        post_label_id = image_id + 'post_disaster_rgb.png'
        pre_image_path = self.data_root + '/' + self.dataset + '/images/' + pre_image_id
        post_image_path = self.data_root + '/' + self.dataset + '/images/' + post_image_id
        pre_label_path = self.data_root + '/' + self.dataset + '/masks/' + pre_label_id
        post_label_path = self.data_root + '/' + self.dataset + '/masks/' + post_label_id
        #
        pre_image = self.load(pre_image_path, file_type='rbg_image')
        post_image = self.load(post_image_path, file_type='rbg_image')
        pre_label = self.load(pre_label_path, file_type='binary_label')
        post_label = self.load(post_label_path, file_type='rbg_label')
        #
        if self.transform:
            sample = self.train_transforms_all(image=pre_image, image1=post_image, mask=pre_label, mask1=post_label)
            pre_image, post_image, pre_label, post_label = (
                sample['image'], sample['image1'], sample['mask'], sample['mask1'])
            sample = self.train_transforms_pre_image(image=pre_image)
            pre_image = sample['image']
            sample = self.train_transforms_post_image(image=post_image)
            post_image = sample['image']

        sample = self.normalize_image(image=pre_image, image1=post_image)
        pre_image, post_image = sample['image'], sample['image1']
        sample = self.to_tensor(image=pre_image, image1=post_image, mask=pre_label, mask1=post_label)
        pre_image_tensor, post_image_tensor, pre_label_tensor, post_label_tensor = sample['image'].contiguous(), \
            sample['image1'].contiguous(), sample['mask'].contiguous(), sample['mask1'].contiguous()

        return pre_image_tensor, post_image_tensor, \
            pre_label_tensor.unsqueeze(dim=0), post_label_tensor.unsqueeze(dim=0), image_id

