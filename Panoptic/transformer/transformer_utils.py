import torch
import torch.nn.functional as F
from typing import Tuple, List, Dict


def panoptic_segmentation_inference(
    voxel_output: torch.Tensor, 
    query_output: torch.Tensor, 
    object_mask_threshold: float = 0.7, 
    vox_occ_threshold: float = 0.3, 
    thing_ids=[1, 2, 3, 4, 5, 6, 7, 8], 
    overlap_threshold: float = 0.4, 
    num_classes=20,
    overlap_flag=True,
):
    if overlap_flag == False:
        overlap_threshold = 0.0
    """
    Performs dense panoptic segmentation by processing the transformer outputs.
    - Assigns unique instance IDs to "things".
    - Merges "stuff" classes with shared IDs per category.

    Args:
        voxel_output (Tensor): (B, num_queries, D, H, W) - voxel logits.
        query_output (Tensor): (B, num_queries, num_classes) - Query logits.
        object_mask_threshold (float): Confidence threshold for filtering low-confidence queries.
        vox_occ_threshold (float): Minimum voxel probability for a valid instance.
        thing_ids (list): Class IDs that correspond to "things".
        overlap_threshold (float): Minimum overlap required for instance assignment.

    Returns:
        dict: Processed panoptic segmentation outputs with instance assignments.
        - "semantic_seg": (B, D, H, W) Semantic segmentation map (equivalent to `argmax(ssc_logits, dim=1)`)

    """


    voxel_output = voxel_output.permute(0,4,1,2,3)
    B, num_queries, D, H, W = (
        voxel_output.shape
    )  

    voxel_probs = torch.sigmoid(voxel_output) 

    query_probs = F.softmax(query_output, dim=-1) 

    query_confidence, query_labels = query_probs.max(dim=-1) 

    panoptic_seg_denses = [] 
    segments_infos = [] 
    semantic_seg_denses = [] 

    for b in range(B):
        panoptic_seg = torch.zeros(
            (D, H, W), dtype=torch.uint8, device=voxel_output.device
        ) 
        semantic_seg = torch.zeros(
            (D, H, W), dtype=torch.uint8, device=voxel_output.device
        ) 

        current_segment_id = 0 
        segments_info = [] 
        stuff_memory = {} 

        keep = (
            (query_labels[b] != 0)
            & (query_labels[b] != num_classes)
            & (query_confidence[b] > object_mask_threshold)
        )
        if keep.sum() == 0:
            panoptic_seg_denses.append(panoptic_seg)
            segments_infos.append(segments_info)
            semantic_seg_denses.append(semantic_seg)

            continue

        filtered_query_probs = query_probs[b, keep] 
        filtered_probs = query_confidence[b,keep].reshape(-1,1) 
        filtered_query_labels = query_labels[b, keep] 
        filtered_voxel_probs = voxel_probs[
            b, keep, :, :, :
        ] 
        filtered_voxel_probs_reshaped = filtered_voxel_probs.reshape(
            -1, D * H * W
        )
        if overlap_flag:
            combined_mask_query_probs = filtered_probs * filtered_voxel_probs_reshaped
            cur_mask_ids = torch.argmax(combined_mask_query_probs, dim=0)
            cur_mask_ids = cur_mask_ids.reshape(D,H,W)
        else:
            cur_mask_ids = torch.argmax(filtered_voxel_probs_reshaped, dim=0)
            cur_mask_ids = cur_mask_ids.reshape(D,H,W)

        voxel_instance_labels = filtered_voxel_probs.argmax(dim=0) 

        for k in range(filtered_query_labels.shape[0]):
            pred_class = filtered_query_labels[k].item() 
            query_max_class_prob = (
                filtered_query_probs[k].max().item()
            ) 

            isthing = pred_class in thing_ids

            mask = (
                cur_mask_ids == k
            ) 
            mask = mask & (
                filtered_voxel_probs[k, :, :, :] >= vox_occ_threshold
            ) 

            mask_area = mask.sum().item() 
            original_area = (
                (filtered_voxel_probs[k, :, :, :] >= vox_occ_threshold).sum().item()
            ) 

            if ((mask_area > 0) and (original_area > 0)) or overlap_flag == False:
                if ((mask_area / original_area) < overlap_threshold) and overlap_flag == True:
                    continue

                if pred_class == 0:
                    panoptic_seg[mask] = 0
                    semantic_seg[mask] = 0
                else:
                    if not isthing:
                        if int(pred_class) in stuff_memory.keys():
                            panoptic_seg[mask] = stuff_memory[int(pred_class)]
                            continue
                        else:
                            stuff_memory[int(pred_class)] = current_segment_id + 1
                    current_segment_id += 1
                    panoptic_seg[mask] = current_segment_id
                    semantic_seg[mask] = pred_class

                    segments_info.append(
                        {
                            "id": current_segment_id, 
                            "isthing": isthing, 
                            "category_id": pred_class, 
                            "confidence": query_max_class_prob, 
                        }
                    )

        panoptic_seg_denses.append(panoptic_seg) 
        segments_infos.append(
            segments_info
        ) 

        semantic_seg_denses.append(semantic_seg)
    return {
        "panoptic_seg": torch.stack(
            panoptic_seg_denses
        ), 
        "segments_info": segments_infos, 
        "semantic_perd": torch.stack(
            semantic_seg_denses
        ), 
    }