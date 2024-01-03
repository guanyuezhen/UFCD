import albumentations as A
from albumentations.pytorch import ToTensorV2
import random
from .base_dataset import BaseDataset


class BCDDataset(BaseDataset):
    def __init__(self,
                 data_cfg=None,
                 dataset='train',
                 transform=None
                 ):
        super(BCDDataset, self).__init__(data_cfg=data_cfg, transform=transform)
        self.data_root = data_cfg['data_root']
        self.dataset = dataset
        self.image_size = data_cfg['image_size']
        self.transform = transform

        self.file_list = open(
            self.data_root + '/' + self.dataset + '/list/' + self.dataset + '.txt').read().splitlines()
        self.pre_images = [self.data_root + '/' + self.dataset + '/A/' + x for x in self.file_list]
        self.post_images = [self.data_root + '/' + self.dataset + '/B/' + x for x in self.file_list]
        self.gts = [self.data_root + '/' + self.dataset + '/label/' + x for x in self.file_list]
        if transform:
            self.train_transforms_all = A.Compose([
                A.Flip(p=0.5),
                A.RandomRotate90(p=0.5)
            ], additional_targets={'image1': 'image'})
            self.train_transforms_pre_image = A.Compose([
                A.ColorJitter(p=0.5)
            ])
            self.train_transforms_post_image = A.Compose([
                A.ColorJitter(p=0.5)
            ])
        self.normalize_image = A.Compose([
            A.Normalize()
        ], additional_targets={'image1': 'image'})
        self.to_tensor = A.Compose([
            ToTensorV2()
        ], additional_targets={'image1': 'image'})

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_path = self.pre_images[idx]
        post_image_path = self.post_images[idx]
        label_path = self.gts[idx]
        #
        pre_image = self.load(pre_image_path, file_type='rbg_image')
        post_image = self.load(post_image_path, file_type='rbg_image')
        label = self.load(label_path, file_type='binary_label')
        #
        if self.transform:
            sample = self.train_transforms_all(image=pre_image, image1=post_image, mask=label)
            pre_image, post_image, label = sample['image'], sample['image1'], sample['mask']
            sample = self.train_transforms_pre_image(image=pre_image)
            pre_image = sample['image']
            sample = self.train_transforms_post_image(image=post_image)
            post_image = sample['image']
            if random.choice([0, 1]):
                pre_image, post_image = post_image, pre_image

        sample = self.normalize_image(image=pre_image, image1=post_image)
        pre_image, post_image = sample['image'], sample['image1']
        sample = self.to_tensor(image=pre_image, image1=post_image, mask=label)
        pre_image_tensor, post_image_tensor, label_tensor = sample['image'].contiguous(), \
            sample['image1'].contiguous(), sample['mask'].contiguous()

        data = {
            'image': {
                'pre_image': pre_image_tensor,
                'post_image': post_image_tensor,
            },
            'label': {
                'binary_label': label_tensor.unsqueeze(dim=0),
            },
            'image_name': self.file_list[idx]
        }

        return data
