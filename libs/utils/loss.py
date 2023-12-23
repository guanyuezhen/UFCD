import torch.nn as nn
import torch.nn.functional as F
import torch


class DeepSupervisionLoss(nn.Module):
    def __init__(self):
        super(DeepSupervisionLoss, self).__init__()
        self.BCEDiceLoss = BinaryCrossEntropyDiceLoss()

    def forward(self, pres, gts):
        if isinstance(gts, tuple):
            gts = gts[0]
        b, n, h, w = pres.size()
        loss = 0
        for i in range(n):
            loss += self.BCEDiceLoss(pres[:, i:i + 1, :, :], gts)

        return loss / n


class ChangeSimilarity(nn.Module):
    """input: x1, x2 multi-class predictions, c = class_num
       label_change: changed part
    """

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
    """
    binary dice loss
    """

    def __init__(self):
        super(BinaryDiceLoss, self).__init__()
        self.smooth = 1e-5
        self.eps = 1e-7

    def forward(self, y_pred_logits, y_true):
        # y_pred = F.logsigmoid(y_pred_logits).exp()
        y_pred = torch.sigmoid(y_pred_logits)
        bs = y_true.size(0)
        num_classes = y_pred.size(1)
        y_pred = y_pred.float().contiguous().view(bs, num_classes, -1)
        y_true = y_true.float().contiguous().view(bs, num_classes, -1)
        intersection = (y_pred * y_true).sum()
        dsc = (2. * intersection + self.smooth) / (y_pred.sum() + y_true.sum() + self.smooth).clamp_min(self.eps)
        loss = 1. - dsc
        return loss.mean()


class BinaryELDiceLoss(nn.Module):
    """
    binary Exponential Logarithmic Dice loss
    """

    def __init__(self):
        super(BinaryELDiceLoss, self).__init__()
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
        return torch.clamp((torch.pow(-torch.log(dsc + self.smooth), 0.3)).mean(), 0, 2)


class BinaryCrossEntropyLoss(nn.Module):
    """
    HybridLoss
    This loss combines a Sigmoid layer and the BCELoss in one single class.
    This version is more numerically stable than using a plain Sigmoid followed by a BCELoss as, by combining the operations into one layer, we take advantage of the log-sum-exp trick for numerical stability.
    pytorch推荐使用binary_cross_entropy_with_logits,
    将sigmoid层和binaray_cross_entropy合在一起计算比分开依次计算有更好的数值稳定性，这主要是运用了log-sum-exp技巧。
    """

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
    """
    binary Dice loss + BCE loss
    """

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


class MutilFocalLoss(nn.Module):
    """
    """

    def __init__(self, alpha, gamma=2, torch=True):
        super(MutilFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.torch = torch

    def forward(self, y_pred_logits, y_true):
        if torch:
            Batchsize, Channel = y_pred_logits.shape[0], y_pred_logits.shape[1]
            y_pred_logits = y_pred_logits.float().contiguous().view(Batchsize, Channel, -1)
            y_true = y_true.long().contiguous().view(Batchsize, -1)
            y_true_onehot = F.one_hot(y_true, Channel)  # N,H*W -> N,H*W, C
            y_true_onehot = y_true_onehot.permute(0, 2, 1).float()  # H, C, H*W
            mask = y_true_onehot.sum((0, 2)) > 0
            CE_loss = nn.CrossEntropyLoss(reduction='none', weight=mask.to(y_pred_logits.dtype))
            logpt = CE_loss(y_pred_logits.float(), y_true.long())
            pt = torch.exp(-logpt)
            loss = (((1 - pt) ** self.gamma) * logpt).mean()
            return loss


class MutilDiceLoss(nn.Module):
    """
        multi label dice loss with weighted
        Y_pred: [None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_pred is softmax result
        Y_gt:[None, self.numclass,self.image_depth, self.image_height, self.image_width],Y_gt is one hot result
        alpha: tensor shape (C,) where C is the number of classes,eg:[0.1,1,1,1,1,1]
        :return:
        """

    def __init__(self, alpha):
        super(MutilDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        # Apply activations to get [0..1] class probabilities
        # Using Log-Exp as this gives more numerically stable result and does not cause vanishing gradient on
        # extreme values 0 and 1
        # y_pred = y_pred_logits.log_softmax(dim=1).exp()
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
        # Dice loss is undefined for non-empty classes
        # So we zero contribution of channel that does not have true pixels
        # NOTE: A better workaround would be to use loss term `mean(y_pred)`
        # for this case, however it will be a modified jaccard loss
        mask = y_true.sum((0, 2)) > 0
        loss *= mask.to(loss.dtype)
        return (loss * self.alpha).sum() / torch.count_nonzero(mask)


class MutilCrossEntropyDiceLoss(nn.Module):
    """
    Mutil Dice loss + CE loss
    """

    def __init__(self, alpha):
        super(MutilCrossEntropyDiceLoss, self).__init__()
        self.alpha = alpha

    def forward(self, y_pred_logits, y_true):
        diceloss = MutilDiceLoss(self.alpha)
        dice = diceloss(y_pred_logits, y_true)
        celoss = MutilCrossEntropyLoss(self.alpha)
        ce = celoss(y_pred_logits, y_true)
        return ce + dice



