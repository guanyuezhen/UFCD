import torch
import numpy as np
from PIL import Image
from libs.losses.loss import BinaryDiceLoss, MutilCrossEntropyDiceLoss
from libs.metrics.bcd_metric import BCDConfuseMatrixMeter
from libs.metrics.bda_metric import BDAConfuseMatrixMeter
from libs.utils.evaluations.base_evaluation import BaseEvaluation


class BDAEvaluation(BaseEvaluation):
    def __init__(self,
                 task_cfg,
                 optimizer_cfg=None
                 ):
        super(BDAEvaluation, self).__init__(task_cfg=task_cfg, optimizer_cfg=optimizer_cfg)
        self.task_cfg = task_cfg
        self.optimizer_cfg = optimizer_cfg

        self.evaluation = {
            'bda_evaluation': BDAConfuseMatrixMeter(self.task_cfg['num_bda_class']),
            'bl_evaluation': BCDConfuseMatrixMeter(self.task_cfg['num_bl_class'] + 1)
        }
        if self.optimizer_cfg is not None:
            alpha = [1.] * self.task_cfg['num_bda_class']
            alpha = torch.as_tensor(alpha).contiguous().cuda()
            self.criterion = {
                'building_location_loss': BinaryDiceLoss().cuda(),
                'building_damage_assessment': MutilCrossEntropyDiceLoss(alpha=alpha).cuda()
            }

    def compute_loss(self, predictions, labels):
        loss_bda = 0
        for i in range(len(predictions['cls_mask'])):
            loss_bda += self.criterion['building_damage_assessment'](predictions['cls_mask'][i], labels['post_label'])
        loss_bl = 0
        for i in range(len(predictions['loc_mask'])):
            loss_bl += self.criterion['building_location_loss'](predictions['loc_mask'][i], labels['post_label'])
        loss = loss_bda + loss_bl

        return loss

    def compute_performance(self, predictions, labels):
        with torch.no_grad():
            arg_cls_mask = torch.argmax(predictions['cls_mask'][0], dim=1, keepdim=False).long()
            F1_dmg = self.evaluation['bda_evaluation'].update_cm(pr=arg_cls_mask.cpu().numpy(),
                                                                 gt=labels['post_label'].cpu().numpy())
            loc_mask = torch.sigmoid(predictions['loc_mask'][0])
            loc_mask = (loc_mask > 0.5).long()
            F1 = self.evaluation['bl_evaluation'].update_cm(pr=loc_mask.cpu().numpy(),
                                                            gt=labels['pre_label'].cpu().numpy())
            F1_over = F1_dmg * 0.7 + F1 * 0.3

            return F1_over

    def compute_per_epoch_performance(self):
        bda_score = self.evaluation['bda_evaluation'].get_scores()
        bl_score = self.evaluation['bl_evaluation'].get_scores()
        score = {
            'F1': bda_score['F1'] * 0.7 + bl_score['F1'] * 0.3,
            'F1_loc': bl_score['F1'],
            'F1_dam': bda_score['F1_dam'],
            'F1_no_dam': bda_score['F1_no_dam'],
            'F1_min_dma': bda_score['F1_min_dma'],
            'F1_maj_dam': bda_score['F1_maj_dam'],
            'F1_des': bda_score['F1_des']
        }

        return score

    def save_prediction(self, cmd_cfg, predictions, image_names, test_loader):
        is_save = cmd_cfg.is_train
        pre_dir = cmd_cfg.pre_dir
        post_dir = cmd_cfg.post_dir
        if is_save == 0:
            pre_mask = predictions['pre_mask'][0]
            for i in range(pre_mask.size(0)):
                pre_mask_i = pre_mask[i:i + 1]
                pre_mask_i = torch.sigmoid(pre_mask_i)
                pre_mask_i = (pre_mask_i > 0.5).long()
                loc_map = pre_mask_i[0, 0].cpu().numpy()
                loc_map = np.asarray(loc_map, dtype='uint8')
                loc_map = Image.fromarray(loc_map)
                loc_map.save(pre_dir + image_names[i])
            post_mask = predictions['post_mask']
            for i in range(post_mask.size(0)):
                post_mask_i = post_mask[i:i + 1]
                pre = (torch.argmax(post_mask_i, dim=1))[0].cpu().numpy()
                cls_map = test_loader.dataset.index_to_color(pre)
                cls_map = Image.fromarray(cls_map)
                cls_map.save(post_dir + image_names[i])
