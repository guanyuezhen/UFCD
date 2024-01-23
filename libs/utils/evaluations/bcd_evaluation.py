import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from libs.losses.loss import BinaryCrossEntropyDiceLoss, BinaryCrossEntropyLoss, MutilCrossEntropyDiceLoss
from libs.metrics.bcd_metric import BCDConfuseMatrixMeter
from libs.utils.evaluations.base_evaluation import BaseEvaluation


class BCDEvaluation(BaseEvaluation):
    def __init__(self,
                 task_cfg,
                 optimizer_cfg=None
                 ):
        super(BCDEvaluation, self).__init__(task_cfg=task_cfg, optimizer_cfg=optimizer_cfg)
        self.task_cfg = task_cfg
        self.optimizer_cfg = optimizer_cfg

        self.evaluation = {
            'bcd_evaluation': BCDConfuseMatrixMeter(self.task_cfg['num_bcd_class'])
        }
        if self.optimizer_cfg is not None:
            alpha = [1.] * self.task_cfg['num_bcd_class']
            alpha = torch.as_tensor(alpha).contiguous().cuda()
            self.criterion = {
                'binary_change_loss': MutilCrossEntropyDiceLoss(alpha=alpha).cuda(),
                'uncertainty_loss': BinaryCrossEntropyLoss().cuda(),
            }

    def compute_loss(self, predictions, labels):
        loss = 0
        for i in range(len(predictions['change_mask'])):
            loss += self.criterion['binary_change_loss'](predictions['change_mask'][i], labels['binary_label'])

        if labels.get('uncertainty_mask') is not None:
            uncertainty_label = torch.abs(predictions['change_mask'][0].detach() - labels['binary_label'])
            uncertainty_label = uncertainty_label / (F.adaptive_max_pool2d(uncertainty_label, (1, 1)) + 1e-3)
            loss += self.criterion['uncertainty_loss'](predictions['uncertainty_mask'], uncertainty_label)

        return loss

    def compute_performance(self, predictions, labels):
        with torch.no_grad():
            binary_target = labels['binary_label'].float()

            change_mask = predictions['change_mask'][0]
            change_mask = torch.argmax(change_mask, dim=1, keepdim=True)
            F1 = self.evaluation['bcd_evaluation'].update_cm(pr=change_mask.cpu().numpy(),
                                                             gt=binary_target.cpu().numpy())
            return F1

    def compute_per_epoch_performance(self):
        bcd_score = self.evaluation['bcd_evaluation'].get_scores()
        score = {
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
        if is_save == 0:
            change_mask = predictions['change_mask'][0]
            for i in range(change_mask.size(0)):
                change_mask_i = change_mask[i:i + 1]
                change_mask = torch.argmax(change_mask, dim=1, keepdim=True) * 255
                binary_map = change_mask_i[0, 0].cpu().numpy()
                binary_map = np.asarray(binary_map, dtype='uint8')
                binary_map = Image.fromarray(binary_map)
                binary_map.save(pre_dir + image_names[i])
