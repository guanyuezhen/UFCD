import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from skimage import measure
from libs.losses.loss import MutilCrossEntropyDiceLoss, BinaryCrossEntropyLoss
from libs.metrics.bcd_metric import BCDConfuseMatrixMeter
from libs.metrics.bda_metric import BDAConfuseMatrixMeter
from libs.utils.evaluations.base_evaluation import BaseEvaluation


def _object_vote(loc, dam):
    damage_cls_list = [1, 2, 3, 4]
    local_mask = loc
    labeled_local, nums = measure.label(local_mask, connectivity=2, background=0, return_num=True)
    region_idlist = np.unique(labeled_local)
    if len(region_idlist) > 1:
        dam_mask = dam
        new_dam = local_mask.copy()
        for region_id in region_idlist:
            if all(local_mask[local_mask == region_id]) == 0:
                continue
            region_dam_count = [int(np.sum(dam_mask[labeled_local == region_id] == dam_cls_i)) * cls_weight \
                                for dam_cls_i, cls_weight in zip(damage_cls_list, [1., 1., 1., 1.])]
            dam_index = np.argmax(region_dam_count) + 1
            new_dam = np.where(labeled_local == region_id, dam_index, new_dam)
    else:
        new_dam = local_mask.copy()
    return new_dam


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
            'bl_evaluation': BCDConfuseMatrixMeter(self.task_cfg['num_bl_class'])
        }
        if self.optimizer_cfg is not None:
            alpha_bda = [1.] * self.task_cfg['num_bda_class']
            alpha_bda = torch.as_tensor(alpha_bda).contiguous().cuda()
            alpha_bl = [1.] * self.task_cfg['num_bl_class']
            alpha_bl = torch.as_tensor(alpha_bl).contiguous().cuda()
            self.criterion = {
                'building_location_loss': MutilCrossEntropyDiceLoss(alpha=alpha_bl).cuda(),
                'building_damage_assessment': MutilCrossEntropyDiceLoss(alpha=alpha_bda).cuda(),
                'uncertainty_loss': BinaryCrossEntropyLoss().cuda(),
            }

    def compute_loss(self, predictions, labels):
        loss_bda = 0
        for i in range(len(predictions['cls_mask'])):
            loss_bda += self.criterion['building_damage_assessment'](predictions['cls_mask'][i], labels['post_label'])
        loss_bl = 0
        for i in range(len(predictions['loc_mask'])):
            loss_bl += self.criterion['building_location_loss'](predictions['loc_mask'][i], labels['pre_label'])
        loss = loss_bda + loss_bl

        if labels.get('uncertainty_mask') is not None:
            uncertainty_label = torch.abs(predictions['change_mask'][0].detach() - labels['post_label'])
            uncertainty_label = uncertainty_label / (F.adaptive_max_pool2d(uncertainty_label, (1, 1)) + 1e-3)
            loss += self.criterion['uncertainty_loss'](predictions['uncertainty_mask'], uncertainty_label)

        return loss

    @staticmethod
    def object_based_infer(pre_logit, post_logit):
        loc = (pre_logit > 0.).cpu().squeeze(1).numpy()
        dam = post_logit.argmax(dim=1).cpu().squeeze(1).numpy()

        refined_dam = np.zeros_like(dam)
        for i, (single_loc, single_dam) in enumerate(zip(loc, dam)):
            refined_dam[i, :, :] = _object_vote(single_loc, single_dam)

        return loc, refined_dam

    def compute_performance(self, predictions, labels, is_use_object_voting=False):
        with torch.no_grad():
            cls_mask = torch.argmax(predictions['cls_mask'][0], dim=1).long().cpu().squeeze(1).numpy()
            loc_mask = torch.argmax(predictions['loc_mask'][0], dim=1).cpu().squeeze(1).numpy()
            if is_use_object_voting:
                refined_cls_mask = np.zeros_like(cls_mask)
                for i, (single_loc, single_cls) in enumerate(zip(loc_mask, cls_mask)):
                    refined_cls_mask[i, :, :] = _object_vote(single_loc, single_cls)
            else:
                refined_cls_mask = cls_mask

            F1_dmg = self.evaluation['bda_evaluation'].update_cm(pr=refined_cls_mask,
                                                                 gt=labels['post_label'].cpu().numpy())
            F1 = self.evaluation['bl_evaluation'].update_cm(pr=loc_mask,
                                                            gt=labels['pre_label'].cpu().numpy())
            F1_over = F1_dmg * 0.7 + F1 * 0.3

            return F1_over

    def compute_per_epoch_performance(self):
        bda_score = self.evaluation['bda_evaluation'].get_scores()
        bl_score = self.evaluation['bl_evaluation'].get_scores()
        score = {
            'F1': bda_score['F1_dam'] * 0.7 + bl_score['F1'] * 0.3,
            'F1_loc': bl_score['F1'],
            'F1_dam': bda_score['F1_dam'],
            'F1_no_dam': bda_score['F1_no_dam'],
            'F1_min_dam': bda_score['F1_min_dam'],
            'F1_maj_dam': bda_score['F1_maj_dam'],
            'F1_des': bda_score['F1_des']
        }

        return score

    def save_prediction(self, cmd_cfg, predictions, image_names, test_loader, is_use_object_voting=False):
        is_save = cmd_cfg.is_train
        pre_dir = cmd_cfg.pre_dir
        post_dir = cmd_cfg.post_dir
        if is_save == 0:
            batch_size = predictions['cls_mask'][0].size()[0]
            cls_mask = torch.argmax(predictions['cls_mask'][0], dim=1).long().cpu().squeeze(1).numpy()
            loc_mask = torch.argmax(predictions['loc_mask'][0], dim=1).cpu().squeeze(1).numpy()
            for i in range(batch_size):
                loc_mask_i = loc_mask[i]
                cls_mask_i = cls_mask[i]
                if is_use_object_voting:
                    refined_cls_mask = _object_vote(loc_mask_i, cls_mask_i)
                else:
                    refined_cls_mask = cls_mask_i
                loc_map = np.asarray(loc_mask_i * 255, dtype='uint8')
                loc_map = Image.fromarray(loc_map)
                loc_map.save(pre_dir + image_names[i] + 'pre_disaster.png')
                cls_map = test_loader.dataset.index_to_color(refined_cls_mask)
                cls_map = Image.fromarray(cls_map)
                cls_map.save(post_dir + image_names[i] + 'post_disaster_rgb.png')
