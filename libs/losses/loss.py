import torch.nn as nn
import torch.nn.functional as F
import torch


class ChangeSimilarity(nn.Module):
    def __init__(self, reduction='mean'):
        super(ChangeSimilarity, self).__init__()
        self.loss_f = nn.CosineEmbeddingLoss(margin=0., reduction=reduction)

    def forward(self, x1, x2, label_change):
        b, c, h, w = x1.size()
        x1 = F.softmax(x1, dim=1)
        x2 = F.softmax(x2, dim=1)
        x1 = x1.permute(0, 2, 3, 1)
        x2 = x2.permute(0, 2, 3, 1)
        x1 = torch.reshape(x1, [b * h * w, c])
        x2 = torch.reshape(x2, [b * h * w, c])

        label_unchange = ~label_change.bool()
        target = label_unchange.float()
        target = target - label_change.float()
        target = torch.reshape(target, [b * h * w])

        loss = self.loss_f(x1, x2, target)
        return loss


# binary loss
class BinaryDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1e-5
        self.eps = 1e-7

    def forward(self, y_pred_logits, y_true):
        y_pred = torch.sigmoid(y_pred_logits)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth).clamp_min(self.eps)
        loss = 1. - dsc
        return loss.mean()


class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()

    def forward(self, y_pred_logits, y_true):
        bs = y_true.size(0)
        num_classes = y_pred_logits.size(1)
        y_pred_logits = y_pred_logits.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        bce = F.binary_cross_entropy_with_logits(y_pred_logits.float(), y_true.float())
        return bce


class BinaryCrossEntropyDiceLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyDiceLoss, self).__init__()

    def forward(self, y_pred_logits, y_true):
        diceloss = BinaryDiceLoss()
        dice = diceloss(y_pred_logits, y_true)
        bceloss = BinaryCrossEntropyLoss()
        bce = bceloss(y_pred_logits, y_true)
        return bce + dice


# mutil loss
class MutilCrossEntropyLoss(nn.Module):
    def __init__(self, alpha, ignore_index=-100):
        super(MutilCrossEntropyLoss, self).__init__()
        self.alpha = alpha
        self.ignore_index = ignore_index

    def forward(self, y_pred_logits, y_true):
        Batchsize, Channel = y_pred_logits.shape[0], y_pred_logits.shape[1]
        y_pred_logits = y_pred_logits.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true_onehot = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true_onehot = y_true_onehot.permute(0, 2, 1).float()  # H, C, H*W
        mask = y_true_onehot.sum((0, 2)) > 0
        loss = F.cross_entropy(y_pred_logits.float(), y_true.long(),
                               weight=mask.to(y_pred_logits.dtype),
                               ignore_index=self.ignore_index)
        return loss


class MutilDiceLoss(nn.Module):
    def __init__(self, alpha):
        super(MutilDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        y_pred = torch.softmax(y_pred_logits, dim=1)
        Batchsize, Channel = y_pred.shape[0], y_pred.shape[1]
        y_pred = y_pred.float().contiguous().view(Batchsize, Channel, -1)
        y_true = y_true.long().contiguous().view(Batchsize, -1)
        y_true = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
        y_true = y_true.permute(0, 2, 1)  # H, C, H*W
        smooth = 1.e-5
        eps = 1e-7
        assert y_pred.size() == y_true.size()
        intersection = torch.sum(y_true * y_pred, dim=(0, 2))
        denominator = torch.sum(y_true + y_pred, dim=(0, 2))
        gen_dice_coef = ((2. * intersection + smooth) / (denominator + smooth)).clamp_min(eps)
        loss = - gen_dice_coef
        mask = y_true.sum((0, 2)) > 0
        loss *= mask.to(loss.dtype)
        return (loss * self.alpha).sum() / torch.count_nonzero(mask)


class MutilCrossEntropyDiceLoss(nn.Module):
    def __init__(self, alpha):
        super(MutilCrossEntropyDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        diceloss = MutilDiceLoss(self.alpha)
        dice = diceloss(y_pred_logits, y_true)
        celoss = MutilCrossEntropyLoss(self.alpha)
        ce = celoss(y_pred_logits, y_true)
        return ce + dice



