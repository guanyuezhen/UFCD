import torch
from PIL import Image
from libs.losses.loss import MutilCrossEntropyDiceLoss, MutilCrossEntropyLoss, ChangeSimilarity
from libs.metrics.scd_metric import SCDConfuseMatrixMeter
from libs.metrics.bcd_metric import BCDConfuseMatrixMeter
from libs.utils.evaluations.base_evaluation import BaseEvaluation


class SCDEvaluation(BaseEvaluation):
    def __init__(self,
                 task_cfg,
                 optimizer_cfg=None
                 ):
        super(SCDEvaluation, self).__init__(task_cfg=task_cfg, optimizer_cfg=optimizer_cfg)
        self.task_cfg = task_cfg
        self.optimizer_cfg = optimizer_cfg

        self.evaluation = {
            'scd_evaluation': SCDConfuseMatrixMeter(self.task_cfg['num_scd_class']),
            'bcd_evaluation': BCDConfuseMatrixMeter(self.task_cfg['num_bcd_class'])
        }
        if self.optimizer_cfg is not None:
            alpha_scd = [1.] * self.task_cfg['num_scd_class']
            alpha_scd = torch.as_tensor(alpha_scd).contiguous().cuda()
            alpha_bcd = [1.] * self.task_cfg['num_bcd_class']
            alpha_bcd = torch.as_tensor(alpha_bcd).contiguous().cuda()
            self.criterion = {
                'binary_change_loss': MutilCrossEntropyDiceLoss(alpha=alpha_bcd).cuda(),
                'semantic_change_loss': MutilCrossEntropyLoss(alpha=alpha_scd, ignore_index=0).cuda(),
                'change_similarity_loss': ChangeSimilarity().cuda(),
            }

    def compute_loss(self, predictions, labels):
        binary_target = (labels['pre_label'] > 0).float()
        loss_pre_scd = 0
        for i in range(len(predictions['pre_mask'])):
            loss_pre_scd += self.criterion['semantic_change_loss'](predictions['pre_mask'][i], labels['pre_label'])

        loss_post_scd = 0
        for i in range(len(predictions['pre_mask'])):
            loss_post_scd += self.criterion['semantic_change_loss'](predictions['post_mask'][i], labels['post_label'])

        loss_bcd = 0
        for i in range(len(predictions['change_mask'])):
            loss_bcd += self.criterion['binary_change_loss'](predictions['change_mask'][i], binary_target)

        loss_sc = self.criterion['change_similarity_loss'](predictions['pre_mask'][0], predictions['post_mask'][0],
                                                           binary_target)

        loss = loss_pre_scd + loss_post_scd + loss_bcd + loss_sc

        return loss

    def compute_performance(self, predictions, labels):
        with torch.no_grad():
            binary_target = (labels['pre_label'] > 0).float()

            change_mask = predictions['change_mask'][0]
            change_mask = torch.argmax(change_mask, dim=1, keepdim=True)
            pre_mask = torch.argmax(predictions['pre_mask'][0], dim=1)
            post_mask = torch.argmax(predictions['post_mask'][0], dim=1)
            mask = torch.cat(
                [pre_mask * change_mask.squeeze().long(), post_mask * change_mask.squeeze().long()], dim=0)
            mask_gt = torch.cat([labels['pre_label'], labels['post_label']], dim=0)
            score = self.evaluation['scd_evaluation'].update_cm(pr=mask.cpu().numpy(),
                                                                gt=mask_gt.cpu().numpy())
            F1 = self.evaluation['bcd_evaluation'].update_cm(pr=change_mask.cpu().numpy(),
                                                             gt=binary_target.cpu().numpy())
            return score

    def compute_per_epoch_performance(self):
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

        return score

    def save_prediction(self, cmd_cfg, predictions, image_names, test_loader):
        is_save = cmd_cfg.is_train
        pre_dir = cmd_cfg.pre_dir
        post_dir = cmd_cfg.post_dir
        if is_save == 0:
            change_mask = predictions['change_mask'][0]
            pre_mask = predictions['pre_mask'][0]
            post_mask = predictions['post_mask'][0]
            for i in range(pre_mask.size(0)):
                pre_mask_i = pre_mask[i:i + 1]
                post_mask_i = post_mask[i:i + 1]
                change_mask_i = change_mask[i:i + 1]
                change_mask_i = torch.argmax(change_mask_i, dim=1, keepdim=True)
                pre = (torch.argmax(pre_mask_i, dim=1) * change_mask_i.squeeze(1))[0].cpu().numpy()
                scd_map = test_loader.dataset.index_to_color(pre)
                scd_map = Image.fromarray(scd_map)
                scd_map.save(pre_dir + image_names[i])

                pre = (torch.argmax(post_mask_i, dim=1) * change_mask_i.squeeze(1))[0].cpu().numpy()
                scd_map = test_loader.dataset.index_to_color(pre)
                scd_map = Image.fromarray(scd_map)
                scd_map.save(post_dir + image_names[i])
