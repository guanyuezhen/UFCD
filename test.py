import os
import time
from argparse import ArgumentParser
import numpy as np
import scipy.io as scio
from PIL import Image
import torch
import torch.optim.lr_scheduler
from models.initial_model import get_model_by_name
from datasets.dataset import Index2Color
import datasets.dataset as myDataLoader
from utils.sc_metric_tool import SCConfuseMatrixMeter
from utils.bs_metric_tool import BSConfuseMatrixMeter


@torch.no_grad()
def val(args, val_loader, model, pre_vis_dir, post_vis_dir):
    model.eval()
    sc_evaluation = SCConfuseMatrixMeter(n_class=args.num_classes)
    bc_evaluation = BSConfuseMatrixMeter(n_class=2)
    total_batches = len(val_loader)
    print(len(val_loader))

    for iter, batched_inputs in enumerate(val_loader):
        start_time = time.time()
        pre_img, post_img, pre_target, post_target, img_names = batched_inputs
        pre_img = pre_img.cuda()
        post_img = post_img.cuda()
        pre_target = pre_target.cuda()
        post_target = post_target.cuda()
        binary_target = (pre_target > 0).float()

        # run the model
        pre_mask, post_mask, change_mask = model(pre_img, post_img)

        time_taken = time.time() - start_time

        # save change maps
        change_mask = change_mask[:, 0:1]
        change_mask = torch.sigmoid(change_mask)
        change_mask = torch.where(change_mask > 0.5, torch.ones_like(change_mask), torch.zeros_like(change_mask)).long()

        # Modify the following lines to handle the batch dimension
        for i in range(pre_mask.size(0)):
            pre_mask_i = pre_mask[i:i + 1]
            post_mask_i = post_mask[i:i + 1]
            change_mask_i = change_mask[i:i + 1]
            pre = (torch.argmax(pre_mask_i, dim=1) * change_mask_i.squeeze(1))[0].cpu().numpy()
            scd_map = Index2Color(pre, val_loader.dataset.ST_COLORMAP)
            scd_map = Image.fromarray(scd_map)
            scd_map.save(pre_vis_dir + img_names[i])

            pre = (torch.argmax(post_mask_i, dim=1) * change_mask_i.squeeze(1))[0].cpu().numpy()
            scd_map = Index2Color(pre, val_loader.dataset.ST_COLORMAP)
            scd_map = Image.fromarray(scd_map)
            scd_map.save(post_vis_dir + img_names[i])

        # Computing Performance
        pre_mask = torch.argmax(pre_mask, dim=1)
        post_mask = torch.argmax(post_mask, dim=1)
        mask = torch.cat([pre_mask * change_mask.squeeze().long(), post_mask * change_mask.squeeze().long()], dim=0)
        mask_gt = torch.cat([pre_target[:, 0], post_target[:, 0]], dim=0)
        o_score = sc_evaluation.update_cm(pr=mask.cpu().numpy(), gt=mask_gt.cpu().numpy())
        f1 = bc_evaluation.update_cm(pr=change_mask.cpu().numpy(), gt=binary_target.cpu().numpy())
        if iter % 5 == 0:
            print('\r[%d/%d] Score: %3f time: %.3f' % (iter, total_batches, o_score, time_taken), end='')

    sc_scores = sc_evaluation.get_scores()
    bc_scores = bc_evaluation.get_scores()

    return sc_scores, bc_scores


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

    args.save_dir = os.path.join(args.save_dir, args.model_name,
                                 args.file_name + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '/')
    args.pre_vis_dir = './predict/' + args.model_name + '/' + args.file_name + '/pre/'
    args.post_vis_dir = './predict/' + args.model_name + '/' + args.file_name + '/post/'
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.pre_vis_dir, exist_ok=True)
    os.makedirs(args.post_vis_dir, exist_ok=True)

    # formulate models
    model = get_model_by_name(args.model_name, args.num_classes, args.inWidth)
    model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    test_data = myDataLoader.Dataset("val", file_name=args.file_name, data_root=args.data_root, transform=False)
    testLoader = torch.utils.data.DataLoader(test_data, shuffle=False, batch_size=args.batch_size,
                                             num_workers=args.num_workers, pin_memory=False)

    logFileLoc = os.path.join(args.save_dir, args.logFile)
    column_width = 12  # Adjust this width based on your preference

    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Parameters: %s" % (str(total_params)))
        header = "\n{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|" \
                 "{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}".format(
            'Epoch', 'OA (sc)', 'Score (sc)', 'mIoU (sc)', 'Sek (sc)', 'Fscd (sc)',
            'Kappa (bc)', 'IoU (bc)', 'F1 (bc)', 'Rec (bc)', 'Pre (bc)', width=column_width
        )
        logger.write(header)
    logger.flush()

    # load the model
    model_file_name = os.path.join(args.save_dir, 'best_model.pth')
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)
    sc_score_test, bc_score_test = val(args, testLoader, model, args.pre_vis_dir, args.post_vis_dir)
    print("\nTest :\t OA (te) = %.2f\t mIoU (te) = %.2f\t Sek (te) = %.2f\t Fscd (te) = %.2f" \
          % (sc_score_test['OA'], sc_score_test['mIoU'], sc_score_test['Sek'], sc_score_test['Fscd']))
    test_line = "\n{: ^{width}}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|" \
                "{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}".format(
        'Test', sc_score_test['OA'], sc_score_test['Score'], sc_score_test['mIoU'], sc_score_test['Sek'],
        sc_score_test['Fscd'], bc_score_test['Kappa'], bc_score_test['IoU'], bc_score_test['F1'],
        bc_score_test['recall'], bc_score_test['precision'], width=column_width
    )
    logger.write(test_line)
    logger.flush()
    scio.savemat(os.path.join(args.pre_vis_dir, 'sc_results.mat'), sc_score_test)
    scio.savemat(os.path.join(args.pre_vis_dir, 'bc_results.mat'), bc_score_test)
    logger.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--model_name', default="SSCDL", help='Data directory')
    parser.add_argument('--file_name', default="LandsatSCD", help='Data directory')
    parser.add_argument('--inWidth', type=int, default=416, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=416, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=10000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--save_dir', default='./weights/', help='Directory to save the results')
    parser.add_argument('--logFile', default='trainLog.txt', help='File that stores the training and validation logs')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    train_val_change_detection(args)
