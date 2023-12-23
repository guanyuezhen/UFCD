import time
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from libs.utils.semantic_change_detection_metric_tool import SCDConfuseMatrixMeter
from libs.utils.binary_change_detection_metric_tool import BCDConfuseMatrixMeter
from libs.tools.cal_preformance import compute_performance_for_model


def adjust_learning_rate(args, optimizer, iter, max_batches, lr_factor=1):
    max_iter = max_batches * args.max_epochs
    warm_up_iter = np.floor(max_iter * 0.1)
    if args.lr_mode == 'poly':
        cur_iter = iter - warm_up_iter
        max_iter = max_iter - warm_up_iter
        lr = args.lr * (1 - cur_iter * 1.0 / max_iter) ** 0.9
    else:
        raise ValueError('Unknown lr mode {}'.format(args.lr_mode))
    if iter < warm_up_iter:
        lr = args.lr * 0.9 * (iter + 1) / warm_up_iter + 0.1 * args.lr  # warm_up
    lr *= lr_factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def train(args, train_loader, model, criterion, scaler, optimizer, max_batches, cur_iter=0):
    model.train()
    sc_evaluation = SCDConfuseMatrixMeter(n_class=args.num_classes)
    bc_evaluation = BCDConfuseMatrixMeter(n_class=2)
    epoch_loss = []

    for iter, batched_inputs in enumerate(train_loader):
        start_time = time.time()

        pre_img, post_img, pre_target, post_target, img_names = batched_inputs
        pre_img, post_img, pre_target, post_target = map(lambda x: x.cuda(),
                                                         [pre_img, post_img, pre_target, post_target])
        binary_target = (pre_target > 0).float()

        lr = adjust_learning_rate(args, optimizer, iter + cur_iter, max_batches)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            pre_mask, post_mask, change_mask = model(pre_img, post_img)

            pre_mask, post_mask, change_mask = map(lambda x: x.float(), [pre_mask, post_mask, change_mask])
            loss_scd = criterion['semantic_change_loss'](pre_mask, pre_target) \
                       + criterion['semantic_change_loss'](post_mask, post_target)
            loss_bcd = criterion['binary_change_loss'](change_mask, binary_target)
            loss_sc = criterion['change_similarity_loss'](pre_mask, post_mask, binary_target)
            loss = loss_scd * 0.5 + loss_bcd + loss_sc

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter - cur_iter) * time_taken / 3600

        o_score = compute_performance_for_model(change_mask, pre_mask, post_mask,
                                                binary_target, pre_target, post_target,
                                                sc_evaluation, bc_evaluation)

        if iter % 5 == 0:
            print('\riteration: [%d/%d] Score: %.2f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * args.max_epochs, o_score, lr, loss.data.item(),
                res_time), end='')

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    sc_scores = sc_evaluation.get_scores()
    bc_scores = bc_evaluation.get_scores()

    return average_epoch_loss_train, sc_scores, bc_scores, lr
