import torch.utils.data
import numpy as np
import cv2


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self,
                 data_cfg: dict,
                 train_cfg: dict,
                 ):
        super(BaseDataset, self).__init__()
        self.data_root = data_cfg['data_root']
        self.image_size = data_cfg['image_size']
        self.color_map = data_cfg['color_map']
        self.classes = data_cfg['classes']
        self.sub_set = train_cfg['sub_set']
        self.transform = train_cfg['transform']

        self.color_map_to_label = np.zeros(256 ** 3)
        for i, cm in enumerate(self.color_map):
            self.color_map_to_label[(cm[0] * 256 + cm[1]) * 256 + cm[2]] = i

    def __len__(self):
        pass

    def load(self, file_path, file_type='rbg_image'):
        """Open image and convert image to array."""
        if file_type == 'rbg_image':
            img = cv2.imread(file_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img_rgb

        elif file_type == 'rbg_label':
            label = cv2.imread(file_path)
            label = cv2.cvtColor(label, cv2.COLOR_BGR2RGB)
            label = self.color_to_index(label)
            return label

        elif file_type == 'binary_label':
            label = cv2.imread(file_path, 0)
            label = np.ceil(label / 255)
            return label
        else:

            raise TypeError('%s has not defined' % file_type)

    def __getitem__(self, idx):
        pass

    def index_to_color(self, index_map):
        color_map = np.asarray(self.color_map, dtype='uint8')
        x = np.asarray(index_map, dtype='int32')

        return color_map[x, :]

    def color_to_index(self, color_map):
        data = color_map.astype(np.int32)
        num_classes = len(self.classes)
        idx = (data[:, :, 0] * 256 + data[:, :, 1]) * 256 + data[:, :, 2]
        index_map = self.color_map_to_label[idx]
        index_map = index_map * (index_map < num_classes)

        return index_map
