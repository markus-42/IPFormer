import torch
import torch.nn.functional as F


def ce_ssc_loss(pred, target):
    pred['ssc_logits'] = pred['ssc_logits'].permute(0,4,1,2,3)
    return F.cross_entropy(
        pred['ssc_logits'].float(),
        target["semantic_label"].long(),
        weight=target['class_weights'].float(),
        ignore_index=255,
        reduction='mean',
    )


def sem_scal_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)
    target = target["semantic_label"]
    mask = target != 255
    target = target[mask]

    loss, cnt = 0, 0
    num_classes = pred.shape[1]
    for i in range(0, num_classes):
        p = pred[:, i]
        p = p[mask]
        completion_target = torch.ones_like(target)
        completion_target[target != i] = 0

        if torch.sum(completion_target) > 0:
            cnt += 1.0
            nominator = (p * completion_target).sum()
            if p.sum() > 0:
                precision = nominator / p.sum()
                loss += F.binary_cross_entropy(precision, torch.ones_like(precision))
            if completion_target.sum() > 0:
                recall = nominator / completion_target.sum()
                loss += F.binary_cross_entropy(recall, torch.ones_like(recall))
            if (1 - completion_target).sum() > 0:
                specificity = (((1 - p) * (1 - completion_target)).sum() /
                               (1 - completion_target).sum())
                loss += F.binary_cross_entropy(specificity, torch.ones_like(specificity))
    return loss / cnt


def geo_scal_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)
    target = target["semantic_label"]
    mask = target != 255

    empty_probs = pred[:, 0]
    nonempty_probs = 1 - empty_probs
    empty_probs = empty_probs[mask]
    nonempty_probs = nonempty_probs[mask]

    nonempty_target = target != 0
    nonempty_target = nonempty_target[mask].float()

    intersection = (nonempty_target * nonempty_probs).sum()
    precision = intersection / nonempty_probs.sum()
    recall = intersection / nonempty_target.sum()
    specificity = ((1 - nonempty_target) * (empty_probs)).sum() / (1 - nonempty_target).sum()
    return (F.binary_cross_entropy(precision, torch.ones_like(precision)) +
            F.binary_cross_entropy(recall, torch.ones_like(recall)) +
            F.binary_cross_entropy(specificity, torch.ones_like(specificity)))


def frustum_proportion_loss(pred, target):
    pred = pred['ssc_logits'].float()
    pred = F.softmax(pred, dim=1)

    frustums_masks = target['frustums_masks']
    frustums_class_dists = target['frustums_class_dists']
    num_frustums = frustums_class_dists.shape[1]
    batch_cnt = frustums_class_dists.sum(0) 

    frustum_loss = 0
    frustum_nonempty = 0
    for f in range(num_frustums):
        frustum_mask = frustums_masks[:, f].unsqueeze(1)
        prob = frustum_mask * pred 
        prob = prob.flatten(2).transpose(0, 1)
        prob = prob.flatten(1) 
        cum_prob = prob.sum(dim=1) 

        total_cnt = batch_cnt[f].sum()
        total_prob = prob.sum()
        if total_prob > 0 and total_cnt > 0:
            fp_target = batch_cnt[f] / total_cnt
            cum_prob = cum_prob / total_prob

            nonzeros = fp_target != 0
            nonzero_p = cum_prob[nonzeros]
            frustum_loss += F.kl_div(torch.log(nonzero_p), fp_target[nonzeros], reduction='sum')
            frustum_nonempty += 1
    return frustum_loss / frustum_nonempty


def inst_label_cls_loss(pred, target, indices, dustbin_class=False):
    """
    Classification loss (NLL), CE loss
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    
    Avoid using complicated helper functions from Mask2Former, iterate over batch size with loop instead
    """
    assert "query_logits" in pred
    bs, _, num_classes = pred["query_logits"].shape
    class_weight = torch.ones(num_classes, device=pred["query_logits"].device)
    class_weight[0] = 0.1
    
    if dustbin_class:
        class_weight[-1] = 0.1
    
    loss = 0
    non_empty_batch_cnt = bs
    for b in range(bs):
        src_idx, target_idx = indices[b]
        if len(src_idx) == 0 and len(target_idx) == 0:
            non_empty_batch_cnt -= 1
            continue
        src_logits = pred["query_logits"][b] 

        target_classes_o = target["mask_label"]["labels"][b][target_idx]
        if dustbin_class:
            target_classes = torch.full(  
                src_logits.shape[:1],
                len(class_weight) - 1,
                dtype=torch.int64,
                device=src_logits.device,
            )
        else:
                target_classes = torch.full(
                src_logits.shape[:1],
                0,
                dtype=torch.int64,
                device=src_logits.device,
            )

        target_classes[src_idx] = target_classes_o.long()

        loss_ce = F.cross_entropy(
            src_logits, target_classes, class_weight, reduction="none"
        )

        loss += loss_ce.mean()
    return loss / max(non_empty_batch_cnt, 1)


def inst_mask_loss(pred, target, indices, dustbin_class=False):
    """
    Should take from loss_masks in https://github.com/astra-vision/PaSCo/blob/fe47d2d8e3e992c86ac11ef87d30b61e1582e9b1/pasco/loss/criterion_sparse.py#L83
    Compute the losses related to the masks: the focal loss and the dice loss. The code seems to also include the unmatched masks.
    targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
    """
    assert "voxel_logits" in pred
    bs, num_queries, num_classes = pred["query_logits"].shape
    class_weight = torch.ones(num_classes, device=pred["query_logits"].device)
    class_weight[0] = 0.1
    
    if dustbin_class:
        class_weight[-1] = 0.1

    total_loss_mask = 0
    total_loss_dice = 0
    total_loss_iou = 0
    non_empty_batch_cnt = bs

    for b in range(bs):
        src_idx, tgt_idx = indices[b]
        if len(src_idx) == 0 and len(tgt_idx) == 0:
            non_empty_batch_cnt -= 1
            continue

        src_masks = pred["voxel_logits"][b] 
        tgt_masks = target["mask_label"]["masks"][b]  


        src_mask = src_masks[..., src_idx] 
        tgt_mask = tgt_masks[..., tgt_idx].type_as(src_mask) 


        tgt_mask_label = target["mask_label"]["labels"][b][tgt_idx]
        tgt_weights = class_weight[tgt_mask_label.long()]


        unknown_mask = target["semantic_label"][b] == 255 
        valid_mask = ~unknown_mask

        
        src_mask = src_mask[valid_mask] 
        tgt_mask = tgt_mask[valid_mask] 

        loss_mask = sigmoid_focal_loss(src_mask, tgt_mask) * tgt_weights.unsqueeze(0) 
        loss_dice = dice_loss(src_mask, tgt_mask) * tgt_weights
        loss_iou = adaptive_iou_loss(src_mask, tgt_mask) * tgt_weights

        total_loss_mask += loss_mask.mean()
        total_loss_dice += loss_dice.mean()

        total_loss_iou += loss_iou.mean()

    losses = {
        "loss_mask": total_loss_mask / max(non_empty_batch_cnt, 1),
        "loss_dice": total_loss_dice / max(non_empty_batch_cnt, 1),
        "loss_iou" : total_loss_iou / max(non_empty_batch_cnt, 1)
    }

    return losses

def dice_loss(inputs, targets, is_inputs_logit=True):
    """
    Compute the DICE loss, similar to generalized IOU for masks
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
    """
    if is_inputs_logit:
        inputs = inputs.sigmoid()
    numerator = 2 * (inputs * targets).sum(0)
    denominator = inputs.sum(0) + targets.sum(0)
    loss = 1 - (numerator + 1) / (denominator + 1)
    return loss


def sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    default values was alpha: float = 0.25, gamma: float = 2
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
    Returns:
        Loss tensor
    """
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma) 

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    
    return loss

def adaptive_iou_loss(inputs, targets, is_inputs_logit=True, alpha=0.75, eps=1e-6):
    """
    Compute the Adaptive IoU Loss.
    
    Args:
        inputs: predictions
        targets: gt
                 Stores the binary classification label for each element.
        is_inputs_logit: If True, applies sigmoid activation to the inputs.
        alpha: Scaling factor to give more weight to small objects (alpha < 1).
        eps: Small value to prevent division by zero.
        
    Returns:
        Adaptive IoU loss value.
    """
    if is_inputs_logit:
        inputs = inputs.sigmoid() 
    
    intersection = (inputs * targets).sum(0) 
    union = inputs.sum(0) + targets.sum(0) - intersection 

    iou = (intersection + eps) / (union + eps) 

    object_size_factor = (targets.sum(0) + eps) ** alpha
    adaptive_iou = iou / object_size_factor

    loss = 1 - adaptive_iou 
    return loss



