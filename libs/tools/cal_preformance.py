import torch
from PIL import Image
from libs.losses.loss import ChangeSimilarity, DeepSupervisionLoss, MutilCrossEntropyLoss, MutilCrossEntropyDiceLoss
from libs.metrics.semantic_change_detection_metric import SCDConfuseMatrixMeter
from libs.metrics.binary_change_detection_metric import BCDConfuseMatrixMeter
from libs.metrics.building_damage_assessment_metric import BDAConfuseMatrixMeter


class EvaluationByType:
    def __init__(self,
                 task_type,
                 task_cfg,
                 optimizer_cfg=None
                 ):
        self.task_type = task_type
        self.task_cfg = task_cfg
        self.optimizer_cfg = optimizer_cfg

        assert task_type in ['scd', 'bda']
        if task_type == 'scd':
            self.evaluation = {
                'scd_evaluation': SCDConfuseMatrixMeter(self.task_cfg['num_scd_class']),
                'bcd_evaluation': BCDConfuseMatrixMeter(self.task_cfg['num_bcd_class'] + 1)
            }
            if self.optimizer_cfg is not None:
                alpha = [1.] * self.task_cfg['num_scd_class']
                alpha = torch.as_tensor(alpha).contiguous().cuda()
                self.criterion = {
                    'binary_change_loss': DeepSupervisionLoss().cuda(),
                    'semantic_change_loss': MutilCrossEntropyLoss(alpha=alpha, ignore_index=0).cuda(),
                    'change_similarity_loss': ChangeSimilarity().cuda()
                }
        else:
            self.evaluation = {
                'bda_evaluation': BDAConfuseMatrixMeter(self.task_cfg['num_bda_class']),
                'bl_evaluation': BCDConfuseMatrixMeter(self.task_cfg['num_bl_class'] + 1)
            }
            if self.optimizer_cfg is not None:
                alpha = [1.] * self.task_cfg['num_bda_class']
                alpha = torch.as_tensor(alpha).contiguous().cuda()
                self.criterion = {
                    'building_location_loss': DeepSupervisionLoss().cuda(),
                    'building_damage_assessment': MutilCrossEntropyDiceLoss(alpha=alpha).cuda()
                }

    def compute_loss(self, prediction, pre_target, post_target):
        if self.task_type == 'scd':
            binary_target = (pre_target > 0).float()
            loss_scd = self.criterion['semantic_change_loss'](prediction['pre_mask'], pre_target) \
                       + self.criterion['semantic_change_loss'](prediction['post_mask'], post_target)
            loss_bcd = self.criterion['binary_change_loss'](prediction['change_mask'], binary_target)
            loss_sc = self.criterion['change_similarity_loss'](prediction['pre_mask'], prediction['post_mask'],
                                                               binary_target)
            loss = loss_scd * 0.5 + loss_bcd + loss_sc
        else:
            loss_bda = self.criterion['building_damage_assessment'](prediction['cls_mask'], post_target)
            loss_bl = self.criterion['building_location_loss'](prediction['loc_mask'], pre_target)
            loss = loss_bda + loss_bl

        return loss

    def compute_performance(self, prediction, pre_target, post_target):
        with torch.no_grad():
            if self.task_type == 'scd':
                binary_target = (pre_target > 0).float()

                change_mask = prediction['change_mask'][:, 0:1]
                change_mask = torch.sigmoid(change_mask)
                change_mask = (change_mask > 0.5).long()
                pre_mask = torch.argmax(prediction['pre_mask'], dim=1)
                post_mask = torch.argmax(prediction['post_mask'], dim=1)
                mask = torch.cat(
                    [pre_mask * change_mask.squeeze().long(), post_mask * change_mask.squeeze().long()], dim=0)
                mask_gt = torch.cat([pre_target, post_target], dim=0)
                score = self.evaluation['scd_evaluation'].update_cm(pr=mask.cpu().numpy(),
                                                                    gt=mask_gt.cpu().numpy())
                F1 = self.evaluation['bcd_evaluation'].update_cm(pr=change_mask.cpu().numpy(),
                                                                 gt=binary_target.cpu().numpy())
                return score
            else:
                arg_cls_mask = torch.argmax(prediction['cls_mask'], dim=1, keepdim=False).long()
                F1_dmg = self.evaluation['bda_evaluation'].update_cm(pr=arg_cls_mask.cpu().numpy(),
                                                                     gt=post_target.cpu().numpy())
                loc_mask = torch.sigmoid(prediction['loc_mask'])
                loc_mask = (loc_mask > 0.5).long()
                F1 = self.evaluation['bl_evaluation'].update_cm(pr=loc_mask.cpu().numpy(), gt=pre_target.cpu().numpy())
                F1_over = F1_dmg * 0.7 + F1 * 0.3

                return F1_over

    def compute_pre_epoch_performance(self):
        if self.task_type == 'scd':
            scd_score = self.evaluation['scd_evaluation'].get_scores()
            bcd_score = self.evaluation['bcd_evaluation'].get_scores()
            score = {
                'OA': scd_score['OA'],
                'Score': scd_score['Score'],
                'mIoU': scd_score['mIoU'],
                'Sek': scd_score['Sek'],
                'Fscd': scd_score['Fscd'],
                'Kappa': bcd_score['Kappa'],
                'IoU': bcd_score['IoU'],
                'F1': bcd_score['F1'],
                'Rec': bcd_score['recall'],
                'Pre': bcd_score['precision'],
            }
        else:
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

    def save_prediction(self, cmd_cfg, prediction, img_names, test_loader):
        is_save = cmd_cfg.is_train
        pre_dir = cmd_cfg.pre_dir
        post_dir = cmd_cfg.post_dir
        if is_save == 0:
            if self.task_type == 'scd':
                change_mask = prediction['change_mask'][:, 0:1]
                pre_mask = prediction['pre_mask']
                post_mask = prediction['post_mask']
                for i in range(pre_mask.size(0)):
                    pre_mask_i = pre_mask[i:i + 1]
                    post_mask_i = post_mask[i:i + 1]
                    change_mask_i = change_mask[i:i + 1]
                    change_mask_i = torch.sigmoid(change_mask_i)
                    change_mask_i = (change_mask_i > 0.5).long()
                    pre = (torch.argmax(pre_mask_i, dim=1) * change_mask_i.squeeze(1))[0].cpu().numpy()
                    scd_map = test_loader.dataset.index_to_color(pre)
                    scd_map = Image.fromarray(scd_map)
                    scd_map.save(pre_dir + img_names[i])

                    pre = (torch.argmax(post_mask_i, dim=1) * change_mask_i.squeeze(1))[0].cpu().numpy()
                    scd_map = test_loader.dataset.index_to_color(pre)
                    scd_map = Image.fromarray(scd_map)
                    scd_map.save(post_dir + img_names[i])
            else:
                pass

