import time
from PIL import Image
import torch
import torch.optim.lr_scheduler
from libs.datasets.dataset import Index2Color
from libs.utils.semantic_change_detection_metric_tool import SCDConfuseMatrixMeter
from libs.utils.binary_change_detection_metric_tool import BCDConfuseMatrixMeter
from libs.tools.cal_preformance import compute_performance_for_model


@torch.no_grad()
def test(args, val_loader, model):
    model.eval()
    sc_evaluation = SCDConfuseMatrixMeter(n_class=args.num_classes)
    bc_evaluation = BCDConfuseMatrixMeter(n_class=2)

    total_batches = len(val_loader)
    print(f'Total batches for validation: {total_batches}')

    for iter, batched_inputs in enumerate(val_loader):
        start_time = time.time()
        pre_img, post_img, pre_target, post_target, img_names = batched_inputs
        pre_img, post_img, pre_target, post_target = map(lambda x: x.cuda(),
                                                         [pre_img, post_img, pre_target, post_target])

        binary_target = (pre_target > 0).float()

        pre_mask, post_mask, change_mask = model(pre_img, post_img)
        change_mask = change_mask[:, 0:1]

        time_taken = time.time() - start_time

        if args.save_maps > 0:
            for i in range(pre_mask.size(0)):
                pre_mask_i = pre_mask[i:i + 1]
                post_mask_i = post_mask[i:i + 1]
                change_mask_i = change_mask[i:i + 1]
                change_mask_i = torch.sigmoid(change_mask_i)
                change_mask_i = torch.where(change_mask_i > 0.5, torch.ones_like(change_mask_i),
                                            torch.zeros_like(change_mask_i)).long()
                pre = (torch.argmax(pre_mask_i, dim=1) * change_mask_i.squeeze(1))[0].cpu().numpy()
                scd_map = Index2Color(pre, val_loader.dataset.ST_COLORMAP)
                scd_map = Image.fromarray(scd_map)
                scd_map.save(args.pre_vis_dir + img_names[i])

                pre = (torch.argmax(post_mask_i, dim=1) * change_mask_i.squeeze(1))[0].cpu().numpy()
                scd_map = Index2Color(pre, val_loader.dataset.ST_COLORMAP)
                scd_map = Image.fromarray(scd_map)
                scd_map.save(args.post_vis_dir + img_names[i])

        o_score = compute_performance_for_model(change_mask, pre_mask, post_mask,
                                                binary_target, pre_target, post_target,
                                                sc_evaluation, bc_evaluation)
        if iter % 5 == 0:
            print('\r[%d/%d] Score: %3f time: %.3f' % (iter, total_batches, o_score, time_taken), end='')

    sc_scores = sc_evaluation.get_scores()
    bc_scores = bc_evaluation.get_scores()

    return sc_scores, bc_scores
