import time
import torch
from libs.tools.cal_preformance import EvaluationByType


@torch.no_grad()
def test(cmd_cfg, task_type, task_cfg, test_loader, model):
    model.eval()
    test_evaluation = EvaluationByType(task_type=task_type,
                                       task_cfg=task_cfg)
    total_batches = len(test_loader)
    print(f'Total batches for validation: {total_batches}')

    for iter, batched_inputs in enumerate(test_loader):
        start_time = time.time()
        pre_img, post_img, pre_target, post_target, img_names = batched_inputs
        pre_img, post_img, pre_target, post_target = map(lambda x: x.cuda(),
                                                         [pre_img, post_img, pre_target, post_target])

        prediction = model(pre_img, post_img)

        time_taken = time.time() - start_time

        test_evaluation.save_prediction(cmd_cfg, prediction, img_names, test_loader)

        score_per_iter = test_evaluation.compute_performance(prediction, pre_target, post_target)
        if iter % 5 == 0:
            print('\r[%d/%d] Score: %3f time: %.3f' % (iter, total_batches, score_per_iter, time_taken), end='')

    score = test_evaluation.compute_pre_epoch_performance()

    return score
