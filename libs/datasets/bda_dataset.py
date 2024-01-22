from .base_dataset import BaseDataset


class BDADataset(BaseDataset):
    def __init__(self,
                 data_cfg=None,
                 train_cfg=None,
                 ):
        super(BDADataset, self).__init__(data_cfg=data_cfg, train_cfg=train_cfg)

        self.file_list = open(self.data_root + '/' + self.sub_set + '/' + 'image_list.txt').read().splitlines()

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
            'image_name': image_id,
        }

        return data

