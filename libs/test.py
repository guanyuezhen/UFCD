import time
import torch
import torch.nn.functional as F
from collections import defaultdict
from libs.utils.evaluations.scd_evaluation import SCDEvaluation
from libs.utils.evaluations.bda_evaluation import BDAEvaluation
from libs.utils.evaluations.bcd_evaluation import BCDEvaluation


def multi_scale_testing(image, scale):
    if scale != 1:
        batch, channel, width, height = image.size()
        resize_width = int(round(width * scale / 32.0) * 32)
        resize_height = int(round(height * scale / 32.0) * 32)
        resize_image = F.interpolate(image, size=(resize_width, resize_height), mode='bilinear', align_corners=True)
        return resize_image
    else:
        return image


def re_size_prediction(prediction, original_size):
    batch, channel, width, height = prediction.size()
    if width == original_size[0] and height == original_size[1]:
        return prediction
    else:
        resize_prediction = F.interpolate(prediction, size=(original_size[0], original_size[1]),
                                          mode='bilinear', align_corners=True)
        return resize_prediction


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

    is_multi_scale_testing = cmd_cfg.is_multi_scale_testing

    for iter, batched_inputs in enumerate(test_loader):

        start_time = time.time()
        images, labels, image_names = batched_inputs['image'], batched_inputs['label'], batched_inputs['image_name']
        images = {key: value.to('cuda') for key, value in images.items()}
        labels = {key: value.to('cuda') for key, value in labels.items()}

        if is_multi_scale_testing > 0:
            scales = [0.75, 1, 1.25]
            predictions = defaultdict(lambda: [])
            batch, channel, width, height = images['pre_image'].size()
            for selected_scale in scales:
                images = {key: multi_scale_testing(value, selected_scale) for key, value in images.items()}
                selected_scale_predictions = model(images['pre_image'], images['post_image'])
                selected_scale_predictions = {key: [re_size_prediction(v, [width, height]) for v in value]
                                              for key, value in selected_scale_predictions.items()}
                for key, value in selected_scale_predictions.items():
                    if key not in predictions:
                        predictions[key] = [0] * len(value)

                    predictions[key] = [x + y for x, y in zip(predictions[key], value)]

            predictions = {key: [v / len(scales) for v in sum_values] for key, sum_values in predictions.items()}

        else:
            predictions = model(images['pre_image'], images['post_image'])

        time_taken = time.time() - start_time

        test_evaluation.save_prediction(cmd_cfg, predictions, image_names, test_loader)

        score_per_iter = test_evaluation.compute_performance(predictions, labels)
        if iter % 5 == 0:
            print('\r[%d/%d] Score: %3f time: %.3f' % (iter, total_batches, score_per_iter, time_taken), end='')

    score = test_evaluation.compute_per_epoch_performance()

    return score
