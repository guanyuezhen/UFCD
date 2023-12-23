import numpy as np
from .base_metric_tool import AverageMeter, get_confuse_matrix


class BCDConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        super(BCDConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2F1(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def cm2F1(confusion_matrix):
    hist = confusion_matrix
    tp = hist[1, 1]
    fn = hist[1, 0]
    fp = hist[0, 1]
    tn = hist[0, 0]
    # recall
    recall = tp / (tp + fn + np.finfo(np.float32).eps)
    # precision
    precision = tp / (tp + fp + np.finfo(np.float32).eps)
    # F1 score
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    return f1 * 100


def cm2score(confusion_matrix):
    hist = confusion_matrix
    tp = hist[1, 1]
    fn = hist[1, 0]
    fp = hist[0, 1]
    tn = hist[0, 0]
    # acc
    oa = (tp + tn) / (tp + fn + fp + tn + np.finfo(np.float32).eps)
    # recall
    recall = tp / (tp + fn + np.finfo(np.float32).eps)
    # precision
    precision = tp / (tp + fp + np.finfo(np.float32).eps)
    # F1 score
    f1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    # IoU
    iou = tp / (tp + fp + fn + np.finfo(np.float32).eps)
    # pre
    pre = ((tp + fn) * (tp + fp) + (tn + fp) * (tn + fn)) / (tp + fp + tn + fn) ** 2
    # kappa
    kappa = (oa - pre) / (1 - pre)
    #
    kappa *= 100
    iou *= 100
    f1 *= 100
    oa *= 100
    recall *= 100
    precision *= 100
    pre *= 100
    #
    score_dict = {'Kappa': kappa, 'IoU': iou, 'F1': f1, 'OA': oa, 'recall': recall, 'precision': precision, 'Pre': pre}
    return score_dict
