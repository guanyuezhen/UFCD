import numpy as np
from .base_metric import AverageMeter, get_confuse_matrix


class BDAConfuseMatrixMeter(AverageMeter):
    """Computes and stores the average and current value"""

    def __init__(self, n_class):
        super(BDAConfuseMatrixMeter, self).__init__()
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
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    #
    F_dam = 4 / (1 / (F1[1] + np.finfo(np.float32).eps)
                 + 1 / (F1[2] + np.finfo(np.float32).eps)
                 + 1 / (F1[3] + np.finfo(np.float32).eps)
                 + 1 / (F1[4] + np.finfo(np.float32).eps)) * 100

    return F_dam


def cm2score(confusion_matrix):
    hist = confusion_matrix
    tp = np.diag(hist)
    sum_a1 = hist.sum(axis=1)
    sum_a0 = hist.sum(axis=0)

    # recall
    recall = tp / (sum_a1 + np.finfo(np.float32).eps)

    # precision
    precision = tp / (sum_a0 + np.finfo(np.float32).eps)

    # F1 score
    F1 = 2 * recall * precision / (recall + precision + np.finfo(np.float32).eps)
    #
    F_dam = 4 / (1 / (F1[1] + np.finfo(np.float32).eps)
                 + 1 / (F1[2] + np.finfo(np.float32).eps)
                 + 1 / (F1[3] + np.finfo(np.float32).eps)
                 + 1 / (F1[4] + np.finfo(np.float32).eps)) * 100
    F1_no_dam = F1[1] * 100
    F1_min_dam = F1[2] * 100
    F1_maj_dam = F1[3] * 100
    F1_des = F1[4] * 100
    #
    score_dict = {'F1_dam': F_dam, 'F1_no_dam': F1_no_dam, 'F1_min_dam': F1_min_dam,
                  'F1_maj_dam': F1_maj_dam, 'F1_des': F1_des}

    return score_dict
