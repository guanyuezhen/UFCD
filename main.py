import sys

sys.path.insert(0, '.')
import os
import numpy as np
import argparse
import torch
import torch.backends.cudnn
import scipy.io as scio
from libs.builder import get_model_dataset_by_name
from libs.tools.train import train
from libs.tools.test import test
from libs.tools.set_seed import set_seed


def parser_args():
    parser = argparse.ArgumentParser(description='An open source change detection toolbox based on PyTorch')
    parser.add_argument('--data_name', default="xBD", help='Data directory')
    parser.add_argument('--model_name', default="ChangeOS", help='Name of method')
    parser.add_argument('--dataloader_name', default="bs_8", help='Batch size')
    parser.add_argument('--is_train', type=int, default=1, help='Is train model')
    parser.add_argument('--save_dir', default='./weights/', help='Directory to save the results')
    parser.add_argument('--log_file', default='trainLog.txt', help='File that stores the training and validation logs')
    cmd_cfg = parser.parse_args()
    cmd_cfg.save_dir = cmd_cfg.save_dir + cmd_cfg.model_name + '/' + cmd_cfg.data_name + '/'
    cmd_cfg.pre_dir = cmd_cfg.save_dir + '/pre/'
    cmd_cfg.post_dir = cmd_cfg.save_dir + '/post/'
    cmd_cfg.log_file_loc = cmd_cfg.save_dir + cmd_cfg.log_file
    cmd_cfg.model_file_name = cmd_cfg.save_dir + 'best_model.pth'
    print('Called with cmd_cfg:')
    print(cmd_cfg)

    return cmd_cfg


def main():
    set_seed()
    cmd_cfg = parser_args()
    os.makedirs(cmd_cfg.save_dir, exist_ok=True)
    os.makedirs(cmd_cfg.pre_dir, exist_ok=True)
    os.makedirs(cmd_cfg.post_dir, exist_ok=True)

    if cmd_cfg.is_train > 0:
        (logger, model, train_loader, val_loader, test_loader, optimizer, scaler, optimizer_cfg,
         task_type, task_cfg) = get_model_dataset_by_name(cmd_cfg)
        model = model.cuda()

        total_params = sum([np.prod(p.size()) for p in model.parameters()])
        total_params = total_params / 1e6
        print('Total network parameters (excluding idr): ' + str(total_params))
        total_params_to_update = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_to_update = total_params_to_update / 1e6
        print('Total parameters to update: ' + str(total_params_to_update))

        logger.write_parameters(total_params, total_params_to_update)
        logger.write_header()

        max_batches = len(train_loader)
        print('For each epoch, we have {} batches'.format(max_batches))

        max_epochs = optimizer_cfg['max_epoch']
        start_epoch = 0
        cur_iter = 0

        for epoch in range(start_epoch, max_epochs):
            loss_tr, score_tr, lr = train(task_type, task_cfg, optimizer_cfg, train_loader, model, scaler, optimizer,
                                          max_batches, cur_iter)
            cur_iter += len(train_loader)
            torch.cuda.empty_cache()
            logger.save_checkpoint(epoch, model, optimizer, loss_tr, score_tr, lr)

            score_val = test(cmd_cfg, task_type, task_cfg, val_loader, model)
            logger.write_val(epoch, loss_tr, score_tr, score_val)
            logger.save_model(epoch, model, score_val)
            torch.cuda.empty_cache()

        logger.close_logger()

    else:
        logger, model, test_loader, task_type, task_cfg = get_model_dataset_by_name(cmd_cfg)
        model = model.cuda()

        total_params = sum([np.prod(p.size()) for p in model.parameters()])
        total_params = total_params / 1e6
        print('Total network parameters (excluding idr): ' + str(total_params))
        total_params_to_update = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params_to_update = total_params_to_update / 1e6
        print('Total parameters to update: ' + str(total_params_to_update))
        logger.write_parameters(total_params, total_params_to_update)
        logger.write_header()

        state_dict = torch.load(cmd_cfg.model_file_name)
        model.load_state_dict(state_dict)
        score_test = test(cmd_cfg, task_type, task_cfg, test_loader, model)
        logger.write_test(score_test)
        scio.savemat(os.path.join(cmd_cfg.save_dir, 'results.mat'), score_test)
        logger.close_logger()


if __name__ == '__main__':
    main()
