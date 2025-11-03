"""
Modules to compute the matching cost and solve the corresponding LSAP.
"""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from torch import nn


def batch_dice_loss(inputs, targets):
    """
    Compute the DICE loss, similar to generalized IOU for masks.

    Args:
        inputs: Tensor of shape (N, num_queries) from `voxel_logits`.
        targets: Tensor of shape (N, num_target_boxes) from GT masks.

    Returns:
        Tensor of shape (num_queries, num_target_boxes), Dice loss matrix.
    """
    inputs = inputs.sigmoid() 

    numerator = 2 * torch.einsum("ni, nj -> ij", inputs, targets) 
    denominator = inputs.sum(0, keepdim=True).T + targets.sum(0, keepdim=True) 

    loss = 1 - (numerator + 1) / (denominator + 1) 
    return loss 


def batch_sigmoid_focal_loss(inputs, targets, alpha: float = 0.25, gamma: float = 2):
    """
    Computes Sigmoid Focal Loss.

    Args:
        inputs: Tensor of shape (N, num_queries), predicted masks.
        targets: Tensor of shape (N, num_target_boxes), ground truth masks.

    Returns:
        Tensor of shape (num_queries, num_target_boxes), Focal loss matrix.
    """

    prob = inputs.sigmoid()

    focal_pos = ((1 - prob) ** gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.ones_like(inputs), reduction="none"
    ) 

    focal_neg = (prob**gamma) * F.binary_cross_entropy_with_logits(
        inputs, torch.zeros_like(inputs), reduction="none"
    ) 

    if alpha >= 0:
        focal_pos *= alpha
        focal_neg *= (1 - alpha)

    loss = torch.einsum("ni, nj -> ij", focal_pos, targets) + torch.einsum(
        "ni, nj -> ij", focal_neg, (1 - targets)
    ) 

    return loss / inputs.shape[0] 


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network.

    Uses Hungarian matching to assign queries to ground truth instances.
    """

    def __init__(
        self, cost_class: float = 1.0, cost_mask: float = 1.0, cost_dice: float = 1.0
    ):
        """Creates the matcher.

        Params:
            cost_class: Weight for classification error in the matching cost.
            cost_mask: Weight for focal loss of the binary mask in the matching cost.
            cost_dice: Weight for dice loss of the binary mask in the matching cost.
        """
        super().__init__()
        self.cost_class = cost_class 
        self.cost_mask = 40.0 
        self.cost_dice = cost_dice 

        assert (
            cost_class != 0 or cost_mask != 0 or cost_dice != 0
        ), "All costs cannot be 0."

    @torch.no_grad()
    def memory_efficient_forward(self, outputs, targets):
        """
        Performs Hungarian Matching in a memory-efficient manner.

        Args:
            outputs: Dictionary containing:
                - "query_logits": (B, num_queries, num_classes), classification logits.
                - "voxel_logits": (B, 256, 256, 32, num_queries), predicted masks.

            targets: Dictionary containing:
                - "semantic_label": (B, 256, 256, 32), semantic labels.
                - "mask_label":
                    - "labels": (B, num_target_boxes), ground truth instance labels.
                    - "masks": (B, 256, 256, 32, num_target_boxes), GT masks.

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j), where:
                - index_i: indices of the selected predictions.
                - index_j: indices of the corresponding selected targets.
        """

        bs, H, W, D, num_queries = outputs['voxel_logits'].shape 
        num_queries = outputs["query_logits"].shape[1] 
        unknown_mask_dense = (targets["semantic_label"] == 255) 

        indices = []
        for b in range(bs):
            out_prob = (outputs["query_logits"][b]).softmax(-1) 

            out_mask = outputs["voxel_logits"][b] 

            if torch.all(targets["mask_label"]["labels"][b] == -1) and torch.all(targets["mask_label"]["masks"][b] == -1):
                indices.append(([], [])) 
                continue

            tgt_ids = targets["mask_label"]["labels"][b].long() 
            tgt_mask = targets["mask_label"]["masks"][b].to(out_mask) 

            tgt_weights = targets['class_weights'][tgt_ids] 

            valid_mask = (tgt_mask.sum(-1) > 0) & (~unknown_mask_dense[b]) 

            out_mask_flat = out_mask[valid_mask].reshape(-1, num_queries) 
            tgt_mask_flat = tgt_mask[valid_mask].reshape(-1, tgt_mask.shape[-1]) 

            cost_class = -out_prob[:, tgt_ids] 

            cost_dice = batch_dice_loss(out_mask_flat, tgt_mask_flat) 

            if out_mask_flat.shape[0] != 0 and tgt_mask_flat.shape[0] != 0:
                cost_mask = batch_sigmoid_focal_loss(out_mask_flat, tgt_mask_flat) 
            else:
                cost_mask = torch.zeros_like(cost_dice) 

            C = (
                self.cost_mask * cost_mask
                + self.cost_class * cost_class
                + self.cost_dice * cost_dice
            ) 

            C *= tgt_weights[None, :] 

            indices.append(linear_sum_assignment(C.cpu())) 

        return [
            (torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64))
            for i, j in indices
        ]


    @torch.no_grad()
    def forward(self, outputs, targets):
        """Performs the matching

  
        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        return self.memory_efficient_forward(
            outputs, targets
        )

    def __repr__(self):
        head = "Matcher " + self.__class__.__name__
        body = [
            "cost_class: {}".format(self.cost_class),
            "cost_mask: {}".format(self.cost_mask),
            "cost_dice: {}".format(self.cost_dice),
        ]
        _repr_indent = 4
        lines = [head] + [" " * _repr_indent + line for line in body]
        return "\n".join(lines)
