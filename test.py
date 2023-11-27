import sys

sys.path.insert(0, '.')
import torch
import scipy.io as scio
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import datasets.dataset as myDataLoader
from utils.metric_tool import ConfuseMatrixMeter
from PIL import Image
import os, time
import numpy as np
from argparse import ArgumentParser
from models.model import get_model
from datasets.dataset import Index2Color


@torch.no_grad()
def val(args, val_loader, model, pre_vis_dir, post_vis_dir):
    model.eval()

    cd_evaluation = ConfuseMatrixMeter(n_class=args.num_classes)

    total_batches = len(val_loader)
    print(len(val_loader))

    for iter, batched_inputs in enumerate(val_loader):

        pre_img, post_img, pre_target, post_target = batched_inputs
        img_name = val_loader.sampler.data_source.file_list[iter]
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

        # run the model
        pre_mask, post_mask, mask_p2, mask_p3, mask_p4, mask_p5 = model(pre_img_var, post_img_var)
        change_mask = mask_p2

        # torch.cuda.synchronize()
        time_taken = time.time() - start_time

        # save change maps
        change_mask = torch.sigmoid(change_mask)
        change_mask = torch.where(change_mask > 0.5, torch.ones_like(change_mask), torch.zeros_like(change_mask)).long()
        pre = (torch.argmax(pre_mask, dim=1) * change_mask.squeeze())[0].cpu().numpy()
        gt = pre_target_var[0, 0].cpu().numpy()
        scd_map = Index2Color(pre, val_loader.dataset.ST_COLORMAP)
        scd_map = Image.fromarray(scd_map)
        scd_map.save(pre_vis_dir + img_name)

        pre = (torch.argmax(post_mask, dim=1) * change_mask.squeeze())[0].cpu().numpy()
        gt = post_target_var[0, 0].cpu().numpy()
        scd_map = Index2Color(pre, val_loader.dataset.ST_COLORMAP)
        scd_map = Image.fromarray(scd_map)
        scd_map.save(post_vis_dir + img_name)

        # Computing Performance
        pre_mask = torch.argmax(pre_mask, dim=1)
        post_mask = torch.argmax(post_mask, dim=1)
        mask = (torch.cat(
            [pre_mask * change_mask.squeeze().long(),
             post_mask * change_mask.squeeze().long()], dim=0
        ))
        mask_gt = torch.cat([pre_target_var, post_target_var], dim=0)
        Iou = cd_evaluation.update_cm(pr=mask.cpu().numpy(), gt=mask_gt.cpu().numpy())

        if iter % 5 == 0:
            print('\r[%d/%d] IoU: %3f time: %.3f' % (iter, total_batches, Iou, time_taken),
                  end='')

    scores = cd_evaluation.get_scores()

    return scores


def val_change_detection(args):
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

    args.save_dir = args.save_dir + args.file_name + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '/'

    args.pre_vis_dir = './predict/' + args.file_name + '/pre/'
    args.post_vis_dir = './predict/' + args.file_name + '/post/'

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    if not os.path.exists(args.pre_vis_dir):
        os.makedirs(args.pre_vis_dir)

    if not os.path.exists(args.post_vis_dir):
        os.makedirs(args.post_vis_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    if args.file_name == 'SECOND':
        test_data = myDataLoader.Dataset("val", file_name=args.file_name, data_root=args.data_root, transform=False)
    elif args.file_name == 'LandsatSCD':
        test_data = myDataLoader.Dataset("test", file_name=args.file_name, data_root=args.data_root, transform=False)
    else:
        raise TypeError('%s has not defined' % args.file_name)

    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    if args.onGPU:
        cudnn.benchmark = True

    logFileLoc = args.save_dir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write(
            "\n%s\t%s\t%s\t%s\t%s\t%s" % ('Epoch', 'Kappa', 'IoU', 'F1', 'R', 'P'))
    logger.flush()

    # load the model
    model_file_name = args.save_dir + 'best_model.pth'
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    score_test = val(args, testLoader, model, args.pre_vis_dir, args.post_vis_dir)
    torch.cuda.empty_cache()
    print("\nTest :\t OA (te) = %.4f\t mIoU (te) = %.4f\t Sek (te) = %.4f\t Fscd (te) = %.4f" \
          % (score_test['OA'], score_test['mIoU'], score_test['Sek'], score_test['Fscd']))
    logger.write("\n%s\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f"
                 % ('Test', score_test['OA'], score_test['mIoU'], score_test['Sek'], score_test['Fscd']))
    logger.flush()
    scio.savemat(args.pre_vis_dir + 'results.mat', score_test)

    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--file_name', default="SECOND", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=512, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=10000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy')
    parser.add_argument('--save_dir', default='./weights/', help='Directory to save the results')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    val_change_detection(args)
