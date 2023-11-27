import sys

sys.path.insert(0, '.')
import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import datasets.dataset as myDataLoader
from utils.metric_tool import ConfuseMatrixMeter
import os, time
import numpy as np
from argparse import ArgumentParser
from models.model import get_model
from utils.loss import ChangeSimilarity


def BCEDiceLoss(pres, gts):
    pres = torch.sigmoid(pres)
    bce = F.binary_cross_entropy(pres, gts)
    inter = (pres * gts).sum()
    eps = 1e-5
    dice = (2 * inter + eps) / (pres.sum() + gts.sum() + eps)

    return bce + 1 - dice


@torch.no_grad()
def val(args, val_loader, model, criterion):
    model.eval()

    cd_evaluation = ConfuseMatrixMeter(n_class=args.num_classes)
    criterion_sc = ChangeSimilarity().cuda()
    epoch_loss = []

    total_batches = len(val_loader)
    print(len(val_loader))
    for iter, batched_inputs in enumerate(val_loader):

        pre_img, post_img, pre_target, post_target = batched_inputs

        start_time = time.time()

        if args.onGPU == True:
            pre_img = pre_img.cuda()
            post_img = post_img.cuda()
            pre_target = pre_target.cuda()
            post_target = post_target.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        pre_target_var = torch.autograd.Variable(pre_target).long()
        post_target_var = torch.autograd.Variable(post_target).long()
        binary_target_val = (pre_target_var > 0).float()

        # run the model
        pre_mask, post_mask, mask_p2, mask_p3, mask_p4, mask_p5 = model(pre_img_var, post_img_var)
        change_mask = mask_p2
        # loss
        loss_seg = criterion(pre_mask, pre_target_var.squeeze(1)) + criterion(post_mask, post_target_var.squeeze(1))
        loss_bn = BCEDiceLoss(mask_p2, binary_target_val) \
                  + BCEDiceLoss(mask_p3, binary_target_val)\
                  + BCEDiceLoss(mask_p4, binary_target_val)\
                  + BCEDiceLoss(mask_p5, binary_target_val)
        loss_sc = criterion_sc(pre_mask[:, 1:], post_mask[:, 1:], binary_target_val)
        loss = loss_seg * 0.5 + loss_bn * 0.25 + loss_sc

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        epoch_loss.append(loss.data.item())

        # Computing Performance
        with torch.no_grad():
            change_mask = torch.sigmoid(change_mask)
            change_mask = torch.where(change_mask > 0.5, torch.ones_like(change_mask),
                                      torch.zeros_like(change_mask)).long()
            pre_mask = torch.argmax(pre_mask, dim=1)
            post_mask = torch.argmax(post_mask, dim=1)
            mask = (torch.cat(
                [pre_mask * change_mask.squeeze().long(),
                 post_mask * change_mask.squeeze().long()], dim=0
            ))
            mask_gt = torch.cat([pre_target_var, post_target_var], dim=0)
            Iou = cd_evaluation.update_cm(pr=mask.cpu().numpy(), gt=mask_gt.cpu().numpy())

        if iter % 5 == 0:
            print('\r[%d/%d] IoU: %3f loss: %.3f time: %.3f' % (iter, total_batches, Iou, loss.data.item(), time_taken),
                  end='')

    average_epoch_loss_val = sum(epoch_loss) / len(epoch_loss)
    scores = cd_evaluation.get_scores()

    return average_epoch_loss_val, scores


def train(args, train_loader, model, criterion, optimizer, max_batches, cur_iter=0, lr_factor=1.):
    # switch to train mode
    model.train()

    cd_evaluation = ConfuseMatrixMeter(n_class=args.num_classes)
    criterion_sc = ChangeSimilarity().cuda()

    epoch_loss = []

    for iter, batched_inputs in enumerate(train_loader):

        pre_img, post_img, pre_target, post_target = batched_inputs
        #
        start_time = time.time()
        if args.onGPU == True:
            pre_img = pre_img.cuda()
            post_img = post_img.cuda()
            pre_target = pre_target.cuda()
            post_target = post_target.cuda()

        pre_img_var = torch.autograd.Variable(pre_img).float()
        post_img_var = torch.autograd.Variable(post_img).float()
        pre_target_var = torch.autograd.Variable(pre_target).long()
        post_target_var = torch.autograd.Variable(post_target).long()
        binary_target_val = (pre_target_var > 0).float()

        # adjust the learning rate
        lr = adjust_learning_rate(args, optimizer, iter + cur_iter, max_batches, lr_factor)

        # run the model
        pre_mask, post_mask, mask_p2, mask_p3, mask_p4, mask_p5 = model(pre_img_var, post_img_var)
        change_mask = mask_p2
        # loss
        loss_seg = criterion(pre_mask, pre_target_var.squeeze(1)) + criterion(post_mask, post_target_var.squeeze(1))
        loss_bn = BCEDiceLoss(mask_p2, binary_target_val) \
                  + BCEDiceLoss(mask_p3, binary_target_val)\
                  + BCEDiceLoss(mask_p4, binary_target_val)\
                  + BCEDiceLoss(mask_p5, binary_target_val)
        loss_sc = criterion_sc(pre_mask[:, 1:], post_mask[:, 1:], binary_target_val)
        loss = loss_seg * 0.5 + loss_bn * 0.25 + loss_sc
        #
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss.append(loss.data.item())
        time_taken = time.time() - start_time
        res_time = (max_batches * args.max_epochs - iter - cur_iter) * time_taken / 3600

        # Computing Performance
        with torch.no_grad():
            change_mask = torch.sigmoid(change_mask)
            change_mask = torch.where(change_mask > 0.5, torch.ones_like(change_mask),
                                      torch.zeros_like(change_mask)).long()
            pre_mask = torch.argmax(pre_mask, dim=1)
            post_mask = torch.argmax(post_mask, dim=1)
            mask = (torch.cat(
                [pre_mask * change_mask.squeeze().long(),
                 post_mask * change_mask.squeeze().long()], dim=0
            ))
            mask_gt = torch.cat([pre_target_var, post_target_var], dim=0)
            Iou = cd_evaluation.update_cm(pr=mask.cpu().numpy(), gt=mask_gt.cpu().numpy())

        if iter % 5 == 0:
            print('\riteration: [%d/%d] mIoU: %.3f lr: %.7f loss: %.3f time:%.3f h' % (
                iter + cur_iter, max_batches * args.max_epochs, Iou, lr, loss.data.item(),
                res_time), end='')

    average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
    scores = cd_evaluation.get_scores()

    return average_epoch_loss_train, scores, lr


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


def train_val_change_detection(args):
    torch.backends.cudnn.benchmark = True
    SEED = 3047
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if args.file_name == 'SECOND':
        args.data_root = '/mnt/disk_d/Change Detection/Datasets_SCD/SECOND_rgb'
        args.num_classes = 7
    elif args.file_name == 'LandsatSCD':
        args.data_root = '/mnt/disk_d/Change Detection/Datasets_SCD/LandsatSCD_rgb'
        args.num_classes = 5
    else:
        raise TypeError('%s has not defined' % args.file_name)

    model = get_model(input_nc=3, output_nc=args.num_classes)

    args.save_dir = args.save_dir + args.file_name + '_iter_' + str(args.max_steps) + '_lr_' + str(
        args.lr) + '/'
    args.vis_dir = args.save_dir + '/vis/'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    train_data = myDataLoader.Dataset("train", file_name=args.file_name, data_root=args.data_root, transform=True)
    trainLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )

    test_data = myDataLoader.Dataset("test", file_name=args.file_name, data_root=args.data_root, transform=False)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    max_batches = len(trainLoader)
    print('For each epoch, we have {} batches'.format(max_batches))

    if args.onGPU:
        cudnn.benchmark = True

    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0
    max_value = 0

    logFileLoc = args.save_dir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s" % ('Epoch', 'OA (Tr)', 'mIoU (Tr)', 'Sek (Tr)', 'Fscd (Tr)'))
    logger.flush()

    criterion = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)

    # optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), args.lr, (0.9, 0.999), weight_decay=1e-2)

    for epoch in range(start_epoch, args.max_epochs):

        loss_tr, score_tr, lr = \
            train(args, trainLoader, model, criterion, optimizer, max_batches, cur_iter)
        cur_iter += len(trainLoader)

        torch.cuda.empty_cache()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': loss_tr,
            'mIou_Tr': score_tr['mIoU'],
            'lr': lr
        }, args.save_dir + 'checkpoint.pth.tar')

        loss_val, score_val = val(args, testLoader, model, criterion)

        logger.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f" %
                     (epoch, score_val['OA'], score_val['mIoU'], score_val['Sek'], score_val['Fscd']))
        logger.flush()

        # save the model also
        model_file_name = args.save_dir + 'best_model.pth'
        if epoch % 1 == 0 and max_value <= score_val['mIoU']:
            max_value = score_val['mIoU']
            torch.save(model.state_dict(), model_file_name)

        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t mIoU(tr) = %.4f\t mIoU(val) = %.4f"
              % (epoch, loss_tr, loss_val, score_tr['mIoU'], score_val['mIoU']))
        torch.cuda.empty_cache()
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    loss_test, score_test = val(args, testLoader, model, criterion)
    print("\nTest :\t OA (te) = %.4f\t mIoU (te) = %.4f\t Sek (te) = %.4f\t Fscd (te) = %.4f" \
          % (score_test['OA'], score_test['mIoU'], score_test['Sek'], score_test['Fscd']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f"
                 % ('Test', score_test['OA'], score_test['mIoU'], score_test['Sek'], score_test['Fscd']))
    logger.flush()
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_name', default="LandsatSCD", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=416, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=416, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=10000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy')
    parser.add_argument('--save_dir', default='./weights/', help='Directory to save the results')
    parser.add_argument('--logFile', default='trainLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    train_val_change_detection(args)
