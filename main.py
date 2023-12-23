import sys

sys.path.insert(0, '.')
import os
import numpy as np
from argparse import ArgumentParser
import torch
import torch.backends.cudnn
import scipy.io as scio
from libs.models.get_model import get_model_by_name
from libs.datasets.get_data import get_data_by_name
from libs.utils.loss import ChangeSimilarity, DeepSupervisionLoss, MutilCrossEntropyLoss
from libs.tools.train import train
from libs.tools.test import test


def main(args):
    torch.backends.cudnn.benchmark = True
    SEED = 3047
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    args.save_dir = os.path.join(args.save_dir, args.model_name,
                                 args.data_name + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '/')
    args.pre_vis_dir = './results/' + args.model_name + '/' + args.data_name + '/pre/'
    args.post_vis_dir = './results/' + args.model_name + '/' + args.data_name + '/post/'
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.pre_vis_dir, exist_ok=True)
    os.makedirs(args.post_vis_dir, exist_ok=True)

    model = get_model_by_name(args.model_name, args.num_classes, args.inWidth)
    model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    total_params = total_params / 1e6
    print('Total network parameters (excluding idr): ' + str(total_params))
    total_params_to_update = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params_to_update = total_params_to_update / 1e6
    print('Total parameters to update: ' + str(total_params_to_update))

    logFileLoc = args.save_dir + args.logFile
    column_width = 12  # Adjust this width based on your preference
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        logger.write("Total network parameters: %.2f" % total_params)
        logger.write("\nTotal parameters to update: %.2f" % total_params_to_update)
        header = "\n{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}|" \
                 "{: ^{width}}|{: ^{width}}|{: ^{width}}|{: ^{width}}".format(
            'Epoch', 'OA (sc)', 'Score (sc)', 'mIoU (sc)', 'Sek (sc)', 'Fscd (sc)',
            'Kappa (bc)', 'IoU (bc)', 'F1 (bc)', 'Rec (bc)', 'Pre (bc)', width=column_width
        )
        logger.write(header)
    logger.flush()

    if args.is_train > 0:
        train_loader, val_loader, test_loader = get_data_by_name(data_name=args.data_name, batch_size=args.batch_size,
                                                                 num_workers=args.num_workers, is_train=True)
        max_batches = len(train_loader)
        print('For each epoch, we have {} batches'.format(max_batches))

        args.max_epochs = int(np.ceil(args.max_steps / max_batches))
        start_epoch = 0
        cur_iter = 0
        max_value = 0

        alpha = [1.] * args.num_classes
        alpha = torch.as_tensor(alpha).contiguous().cuda()
        criterion = {'binary_change_loss': DeepSupervisionLoss().cuda(),
                     'semantic_change_loss': MutilCrossEntropyLoss(alpha=alpha, ignore_index=0).cuda(),
                     'change_similarity_loss': ChangeSimilarity().cuda()}

        optimizer = torch.optim.AdamW(model.parameters(), args.lr, (0.9, 0.999), weight_decay=1e-2)
        scaler = torch.cuda.amp.GradScaler(enabled=True)

        for epoch in range(start_epoch, args.max_epochs):
            loss_tr, sc_score_tr, bc_score_tr, lr = train(args, train_loader, model, criterion,
                                                          scaler, optimizer, max_batches, cur_iter)
            cur_iter += len(train_loader)
            torch.cuda.empty_cache()

            torch.save({
                'epoch': epoch + 1,
                'arch': str(model),
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lossTr': loss_tr,
                'ScoreTr': sc_score_tr['Score'],
                'lr': lr
            }, args.save_dir + 'checkpoint.pth.tar')

            sc_score_val, bc_score_val = test(args, val_loader, model)

            val_line = "\n{: ^{width}}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|" \
                       "{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}|{: ^{width}.2f}".format(
                epoch, sc_score_val['OA'], sc_score_val['Score'], sc_score_val['mIoU'], sc_score_val['Sek'],
                sc_score_val['Fscd'], bc_score_val['Kappa'], bc_score_val['IoU'], bc_score_val['F1'],
                bc_score_val['recall'], bc_score_val['precision'], width=column_width
            )
            logger.write(val_line)
            logger.flush()

            model_file_name = args.save_dir + 'best_model.pth'
            if epoch % 1 == 0 and max_value <= sc_score_val['Score']:
                max_value = sc_score_val['Score']
                torch.save(model.state_dict(), model_file_name)

            print("Epoch " + str(epoch) + ': Details')
            print("\nEpoch No. %d:\tTrain Loss = %.2f\t Score(tr) = %.2f\t Score(val) = %.2f"
                  % (epoch, loss_tr, sc_score_tr['Score'], sc_score_val['Score']))
            torch.cuda.empty_cache()

    test_loader = get_data_by_name(data_name=args.data_name, batch_size=args.batch_size,
                                   num_workers=args.num_workers, is_train=False)
    model_file_name = args.save_dir + 'best_model.pth'
    state_dict = torch.load(model_file_name)
    model.load_state_dict(state_dict)

    sc_score_test, bc_score_test = test(args, test_loader, model)
    print("\nTest :\t OA (te) = %.2f\t mIoU (te) = %.2f\t Sek (te) = %.2f\t Fscd (te) = %.2f"
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
    parser.add_argument('--data_name', default="SECOND", help='Data directory')
    parser.add_argument('--num_classes', type=int, default=7, help='Number of classes')
    parser.add_argument('--inWidth', type=int, default=512, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=512, help='Height of RGB image')
    parser.add_argument('--model_name', default="A2Net", help='Name of method')
    parser.add_argument('--max_steps', type=int, default=10000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy')
    parser.add_argument('--save_maps', type=int, default=0, help='Save change maps')
    parser.add_argument('--is_train', type=int, default=1, help='Is train model')
    parser.add_argument('--save_dir', default='./weights/', help='Directory to save the results')
    parser.add_argument('--logFile', default='trainLog.txt', help='File that stores the training and validation logs')

    args = parser.parse_args()
    print('Called with args:')
    print(args)

    main(args)
