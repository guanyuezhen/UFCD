import torch


def compute_performance_for_model(change_mask, pre_mask, post_mask,
                                  binary_target, pre_target, post_target,
                                  sc_evaluation, bc_evaluation):
    with torch.no_grad():
        change_mask = change_mask[:, 0:1]
        change_mask = torch.sigmoid(change_mask)
        change_mask = torch.where(change_mask > 0.5, torch.ones_like(change_mask), torch.zeros_like(change_mask)).long()
        pre_mask = torch.argmax(pre_mask, dim=1)
        post_mask = torch.argmax(post_mask, dim=1)
        mask = torch.cat([pre_mask * change_mask.squeeze().long(), post_mask * change_mask.squeeze().long()], dim=0)
        mask_gt = torch.cat([pre_target, post_target], dim=0)
        o_score = sc_evaluation.update_cm(pr=mask.cpu().numpy(), gt=mask_gt.cpu().numpy())
        f1 = bc_evaluation.update_cm(pr=change_mask.cpu().numpy(), gt=binary_target.cpu().numpy())

        return o_score
