import random
from .base_dataset import BaseDataset


class SCDDataset(BaseDataset):
    def __init__(self,
                 data_cfg=None,
                 train_cfg=None,
                 ):
        super(SCDDataset, self).__init__(data_cfg=data_cfg, train_cfg=train_cfg)

        self.file_list = open(
            self.data_root + '/' + self.sub_set + '/list/' + self.sub_set + '.txt').read().splitlines()
        self.pre_images = [self.data_root + '/' + self.sub_set + '/im1/' + x for x in self.file_list]
        self.post_images = [self.data_root + '/' + self.sub_set + '/im2/' + x for x in self.file_list]
        self.pre_gts = [self.data_root + '/' + self.sub_set + '/label1/' + x for x in self.file_list]
        self.post_gts = [self.data_root + '/' + self.sub_set + '/label2/' + x for x in self.file_list]

    def __len__(self):
        return len(self.pre_images)

    def __getitem__(self, idx):
        pre_image_path = self.pre_images[idx]
        post_image_path = self.post_images[idx]
        pre_label_path = self.pre_gts[idx]
        post_label_path = self.post_gts[idx]
        #
        pre_image = self.load(pre_image_path, file_type='rbg_image')
        post_image = self.load(post_image_path, file_type='rbg_image')
        pre_label = self.load(pre_label_path, file_type='rbg_label')
        post_label = self.load(post_label_path, file_type='rbg_label')
        #
        if random.choice([0, 1]):
            pre_image, post_image = post_image, pre_image
            pre_label, post_label = post_label, pre_label
        #
        if self.transform.get('transforms_for_all') is not None:
            sample = self.transform['transforms_for_all'](image=pre_image, image1=post_image,
                                                          mask=pre_label, mask1=post_label)
            pre_image, post_image, pre_label, post_label \
                = (sample['image'], sample['image1'], sample['mask'], sample['mask1'])

        if self.transform.get('transforms_for_pre_image') is not None:
            sample = self.transform['transforms_for_pre_image'](image=pre_image)
            pre_image = sample['image']

        if self.transform.get('transforms_for_post_image') is not None:
            sample = self.transform['transforms_for_post_image'](image=post_image)
            post_image = sample['image']

        if self.transform.get('normalize_image') is not None:
            sample = self.transform['normalize_image'](image=pre_image, image1=post_image)
            pre_image, post_image = sample['image'], sample['image1']

        if self.transform.get('to_tensor') is not None:
            sample = self.transform['to_tensor'](image=pre_image, image1=post_image, mask=pre_label, mask1=post_label)
            pre_image, post_image, pre_label, post_label = (sample['image'].contiguous(), sample['image1'].contiguous(),
                                                            sample['mask'].contiguous(), sample['mask1'].contiguous())

        data = {
            'image': {
                'pre_image': pre_image,
                'post_image': post_image,
            },
            'label': {
                'pre_label': pre_label.unsqueeze(dim=0),
                'post_label': post_label.unsqueeze(dim=0),
            },
            'image_name': self.file_list[idx]
        }

        return data
