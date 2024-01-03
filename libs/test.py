import time
import torch
from libs.utils.evaluations.scd_evaluation import SCDEvaluation
from libs.utils.evaluations.bda_evaluation import BDAEvaluation
from libs.utils.evaluations.bcd_evaluation import BCDEvaluation


@torch.no_grad()
def test(cmd_cfg, task_type, task_cfg, test_loader, model):
    Evaluation_SET = {
        'bda': BDAEvaluation,
        'scd': SCDEvaluation,
        'bcd': BCDEvaluation,
    }
    test_evaluation = Evaluation_SET[task_type](task_cfg=task_cfg)

    model.eval()

    total_batches = len(test_loader)
    print(f'Total batches for validation: {total_batches}')

    for iter, batched_inputs in enumerate(test_loader):
        start_time = time.time()
        images, labels, image_names = batched_inputs['image'], batched_inputs['label'], batched_inputs['image_name']
        images = {key: value.to('cuda') for key, value in images.items()}
        labels = {key: value.to('cuda') for key, value in labels.items()}

        predictions = model(images['pre_image'], images['post_image'])

        time_taken = time.time() - start_time

        test_evaluation.save_prediction(cmd_cfg, predictions, image_names, test_loader)

        score_per_iter = test_evaluation.compute_performance(predictions, labels)
        if iter % 5 == 0:
            print('\r[%d/%d] Score: %3f time: %.3f' % (iter, total_batches, score_per_iter, time_taken), end='')

    score = test_evaluation.compute_per_epoch_performance()

    return score
