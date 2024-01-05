import time
import torch
from torch.nn.utils import clip_grad_norm_
from libs.utils.evaluations.scd_evaluation import SCDEvaluation
from libs.utils.evaluations.bda_evaluation import BDAEvaluation
from libs.utils.evaluations.bcd_evaluation import BCDEvaluation


def adjust_learning_rate(optimizer, iter, max_batches, optimizer_cfg):
    max_iter = max_batches * optimizer_cfg['max_epoch']
    base_lr = optimizer_cfg['lr']
    cur_iter = iter
    power = optimizer_cfg['power']
    min_lr = optimizer_cfg['min_lr']
    warm_up_iter = optimizer_cfg['warm_up_iter']
    warm_up_ratio = optimizer_cfg['warm_up_ratio']
    lr_factor = optimizer_cfg['lr_factor']

    coeff = (1 - cur_iter / max_iter) ** power
    target_lr = coeff * (base_lr - min_lr) + min_lr

    if warm_up_iter >= iter:
        k = (1 - cur_iter / warm_up_iter) * (1 - warm_up_ratio)
        target_lr = (1 - k) * target_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = target_lr * lr_factor

    return target_lr


def train(task_type, task_cfg, optimizer_cfg, train_loader, model, scaler, optimizer, max_batches, cur_iter=0):
    Evaluation_SET = {
        'bda': BDAEvaluation,
        'scd': SCDEvaluation,
        'bcd': BCDEvaluation,
    }
    train_evaluation = Evaluation_SET[task_type](task_cfg=task_cfg,
                                                 optimizer_cfg=optimizer_cfg)
    lr = optimizer_cfg['lr']
    model.train()
    epoch_loss = []

    for iter, batched_inputs in enumerate(train_loader):
        start_time = time.time()

        images, labels, image_names = batched_inputs['image'], batched_inputs['label'], batched_inputs['image_name']
        images = {key: value.to('cuda') for key, value in images.items()}
        labels = {key: value.to('cuda') for key, value in labels.items()}

        lr = adjust_learning_rate(optimizer, iter + cur_iter, max_batches, optimizer_cfg)

        with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            predictions = model(images['pre_image'], images['post_image'])
            loss = train_evaluation.compute_loss(predictions, labels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_(model.parameters(), max_norm=0.1)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * optimizer_cfg['max_epoch'] - iter - cur_iter) * time_taken / 3600

        score_per_iter = train_evaluation.compute_performance(predictions, labels)

        if iter % 5 == 0:
            print('\riteration: [%d/%d] Score: %.2f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * optimizer_cfg['max_epoch'], score_per_iter, lr, loss.data.item(),
                res_time), end='')

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    score = train_evaluation.compute_per_epoch_performance()

    return average_epoch_loss_train, score, lr
