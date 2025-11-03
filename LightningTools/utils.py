import yaml
import numpy as np
import torch
from torch import Tensor
from typing import List

def get_inv_map():
  '''
  remap_lut to remap classes of semantic kitti for training...
  :return:
  '''
  config_path = "./configs/semantickitti/SemanticKITTI.yaml"
  dataset_config = yaml.safe_load(open(config_path, 'r'))
  inv_map = np.zeros(20, dtype=np.int32)
  inv_map[list(dataset_config['learning_map_inv'].keys())] = list(dataset_config['learning_map_inv'].values())

  return inv_map

def merge_instance_masks(
    voxel_probs: Tensor,
    query_probs: Tensor,
    query_class_pred: Tensor,
    thing_classes: List[int],
    object_mask_threshold: float = 0.25,
    query_threshold: float = 0.7
):
    """
    Process the logit voxel probabilities and merge the instance masks to compare with mask-wise merge's final outputs, and reduce space
    
    Args:
        voxel_probs: Tensor (bs, num_instances, 256, 256, 32), sigmoid probability of each voxel indicating which voxel belongs to the mask
        query_probs: Tensor (bs, num_instances), maximum probability for each mask
        query_class_pred: Tensor (bs, num_instances), class with the maximum probability for each mask
        thing_classes: List[int]
        object_mask_threshold: float, sigmoid threshold
        query_threshold: float, class threshold

    Returns:
        raw_instance_pred: Tensor (bs, 256, 256, 32)
    """
    bs, _, h, w, d = voxel_probs.shape
    raw_instance_pred = torch.zeros((bs, h, w, d), dtype=torch.uint8, device=voxel_probs.device)
    
    for b in range(bs):
        valid_mask_indices = torch.where(query_probs[b] > query_threshold)[0]
        
        instance_id = 1
        for idx in valid_mask_indices:
            if query_class_pred[b, idx].item() in thing_classes:
                instance_mask = voxel_probs[b, idx] > object_mask_threshold
                
                raw_instance_pred[b] = torch.where(instance_mask, instance_id, raw_instance_pred[b])
                
                instance_id += 1
    
    return raw_instance_pred
