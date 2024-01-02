import math
import numpy as np
from scipy import stats
from .base_metric import AverageMeter, get_confuse_matrix


class SCDConfuseMatrixMeter(AverageMeter):
    def __init__(self, n_class):
        super(SCDConfuseMatrixMeter, self).__init__()
        self.n_class = n_class

    def update_cm(self, pr, gt, weight=1):
        val = get_confuse_matrix(num_classes=self.n_class, label_gts=gt, label_preds=pr)
        self.update(val, weight)
        current_score = cm2Score(val)
        return current_score

    def get_scores(self):
        scores_dict = cm2score(self.sum)
        return scores_dict


def cm2Score(confusion_matrix):
    hist = confusion_matrix

    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2

    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

    Score = IoU_mean * 0.3 + Sek * 0.7

    return Score * 100


def cal_kappa(hist):
    if hist.sum() == 0:
        po = 0
        pe = 1
        kappa = 0
    else:
        po = np.diag(hist).sum() / hist.sum()
        pe = np.matmul(hist.sum(1), hist.sum(0).T) / hist.sum() ** 2
        if pe == 1:
            kappa = 0
        else:
            kappa = (po - pe) / (1 - pe)

    return kappa


def cm2score(confusion_matrix):
    hist = confusion_matrix
    # acc
    oa = (np.diag(hist).sum()) / (hist.sum() + np.finfo(np.float32).eps)
    # others
    hist_fg = hist[1:, 1:]
    c2hist = np.zeros((2, 2))
    c2hist[0][0] = hist[0][0]
    c2hist[0][1] = hist.sum(1)[0] - hist[0][0]
    c2hist[1][0] = hist.sum(0)[0] - hist[0][0]
    c2hist[1][1] = hist_fg.sum()
    hist_n0 = hist.copy()
    hist_n0[0][0] = 0
    kappa_n0 = cal_kappa(hist_n0)
    iu = np.diag(c2hist) / (c2hist.sum(1) + c2hist.sum(0) - np.diag(c2hist))
    IoU_fg = iu[1]
    IoU_mean = (iu[0] + iu[1]) / 2
    Sek = (kappa_n0 * math.exp(IoU_fg)) / math.e

    pixel_sum = hist.sum()
    change_pred_sum = pixel_sum - hist.sum(1)[0].sum()
    change_label_sum = pixel_sum - hist.sum(0)[0].sum()
    change_ratio = change_label_sum / pixel_sum
    SC_TP = np.diag(hist[1:, 1:]).sum()
    SC_Precision = SC_TP / (change_pred_sum + np.finfo(np.float32).eps)
    SC_Recall = SC_TP / (change_label_sum + np.finfo(np.float32).eps)
    Fscd = stats.hmean([SC_Precision, SC_Recall])
    Score = IoU_mean * 0.3 + Sek * 0.7
    #
    oa *= 100
    Score *= 100
    IoU_mean *= 100
    Sek *= 100
    Fscd *= 100
    #
    score_dict = {'OA': oa, 'Score': Score, 'mIoU': IoU_mean, 'Sek': Sek, 'Fscd': Fscd}

    return score_dict