import torch
from PIL import Image


class BaseEvaluation:
    def __init__(self,
                 task_cfg,
                 optimizer_cfg=None
                 ):
        self.task_cfg = task_cfg
        self.optimizer_cfg = optimizer_cfg

    def compute_loss(self, predictions, labels):
        pass

    def compute_performance(self, predictions, labels):
        pass

    def compute_per_epoch_performance(self):
        pass

    def save_prediction(self, cmd_cfg, predictions, image_names, test_loader):
        pass

