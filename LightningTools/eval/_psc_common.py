"""Code from PaSCo and Symphonies"""
import numpy as np
from collections import defaultdict
import torch
from ..excp import SkipIteration
from typing import Dict, List, Tuple

OFFSET = 256 * 256 * 256

class PQStatCat:
    def __init__(self):
        self.all_iou = 0.0
        self.all_n = 0.0
        self.iou = 0.0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def __iadd__(self, pq_stat_cat):
        self.iou += pq_stat_cat.iou
        self.tp += pq_stat_cat.tp
        self.fp += pq_stat_cat.fp
        self.fn += pq_stat_cat.fn
        self.all_iou += pq_stat_cat.all_iou
        self.all_n += pq_stat_cat.all_n
        return self


class PQStat:
    def __init__(self):
        self.reset()

    def reset(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def print_state(self, ignore_cat_id):
        for label in self.pq_per_cat.keys():
            if label == ignore_cat_id:
                continue
            print(f"Label: {label}, TP: {self.pq_per_cat[label].tp}, FP: {self.pq_per_cat[label].fp}, FN: {self.pq_per_cat[label].fn}")
            print(f"iou: {self.pq_per_cat[label].iou}, all_iou: {self.pq_per_cat[label].all_iou}, all_n: {self.pq_per_cat[label].all_n}")
            print("-" * 50)

    def pq_average(self, isthing, ignore_cat_id, thing_ids):
        pq_dagger, pq, sq, rq, n = 0, 0, 0, 0, 0
        per_class_results = {}
        for label in self.pq_per_cat.keys():

            if label == ignore_cat_id:
                continue
            if isthing is not None:
                cat_isthing = label in thing_ids
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            all_iou = self.pq_per_cat[label].all_iou
            all_n = self.pq_per_cat[label].all_n

            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            if tp + fp + fn == 0:
                per_class_results[label] = {"pq": 0.0, "sq": 0.0, "rq": 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {"pq": pq_class, "sq": sq_class, "rq": rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class

            if isthing is None:
                if label in thing_ids:
                    pq_dagger += pq_class
                else:
                    pq_dagger += all_iou / max(all_n, 1)

        n = max(n, 1)
        return {
            "pq_dagger": pq_dagger / n,
            "pq": pq / n,
            "sq": sq / n,
            "rq": rq / n,
            "n": n,
        }, per_class_results
    
def pq_compute_single_core(
    pq_stat,
    gt_segments_info,
    pred_segments_info,
    pan_gt,
    pan_pred,
    thing_ids,
    ignore_label=0,
):
    """Update the states in the passed pq_stat with TP, FP, FN, IoU."""
    pred_ids = np.unique(pan_pred) 
    gt_segms = {el["id"]: el for el in gt_segments_info}
    pred_segms = {el["id"]: el for el in pred_segments_info}

    pred_labels_set = set(el["id"] for el in pred_segments_info)
    labels, labels_cnt = np.unique(pan_pred, return_counts=True)
    for label, label_cnt in zip(labels, labels_cnt):
        if label not in pred_segms:
            if label == ignore_label:
                continue
            print("Error segment", pred_segms[label])
            raise KeyError(
                "segment with ID {} is presented in PNG and not presented in JSON.".format(
                    label
                )
            )
        pred_segms[label]["area"] = label_cnt
        pred_labels_set.remove(label)
    assert (
        len(pred_labels_set) == 0
    ), "Some segments from JSON are not presented in PNG."

    pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
    gt_pred_map = {}
    labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
    for label, intersection in zip(labels, labels_cnt):
        gt_id = label // OFFSET
        pred_id = label % OFFSET
        if gt_id == ignore_label or pred_id == ignore_label:
            continue
        gt_pred_map[(gt_id, pred_id)] = intersection

    gt_matched = set()
    pred_matched = set()
    pred_gt_matched = set()
    for label_tuple, intersection in gt_pred_map.items():
        gt_label, pred_label = label_tuple
        if gt_label not in gt_segms:
            continue
        if pred_label not in pred_segms:
            continue
        if gt_segms[gt_label]["category_id"] != pred_segms[pred_label]["category_id"]:
            continue
        union = (
            pred_segms[pred_label]["area"] + gt_segms[gt_label]["area"] - intersection
        ) 
        iou = intersection / union
        if gt_segms[gt_label]["category_id"] not in thing_ids:
            pq_stat[gt_segms[gt_label]["category_id"]].all_iou += iou
            pq_stat[gt_segms[gt_label]["category_id"]].all_n += 1
            pred_matched.add(pred_label)
            pred_gt_matched.add(label_tuple)
        if iou > 0.5:
            pq_stat[gt_segms[gt_label]["category_id"]].tp += 1
            pq_stat[gt_segms[gt_label]["category_id"]].iou += iou
            gt_matched.add(gt_label)
            pred_matched.add(pred_label)
            pred_gt_matched.add(label_tuple)

    crowd_labels_dict = {}
    for gt_label, gt_info in gt_segms.items():
        if gt_label in gt_matched:
            continue
        pq_stat[gt_info["category_id"]].fn += 1

    for pred_label, pred_info in pred_segms.items():
        if pred_label in pred_matched:
            continue
        pq_stat[pred_info["category_id"]].fp += 1
    return pred_gt_matched

def preprocess_seminst_vox(
    semantic_voxel,
    instance_voxel,
    thing_ids,
) -> Tuple[torch.Tensor, List[Dict]]:
    """Combine the semantic and instance information into single voxel and dict metadata
    
    The code is adapted from PaSCo panoptic_quality.py and dataloader pipeline

    Args:
        semantic_voxel: np.ndarray, dtype=np.uint8, shape=(256, 256, 32)
        instance_voxel: np.ndarray, dtype=np.uint8, shape=(256, 256, 32)
        thing_ids: List[int]
    Returns:
        pan_seg: torch.Tensor containing instance ids of both things and stuff, dtype=torch.uint64, shape=(256, 256, 32)
        pan_seg_info: List[Dict], length=n_instances, each element is a dict with keys: id, isthing, category_id
    """
    assert semantic_voxel.shape == instance_voxel.shape
    assert semantic_voxel.shape == (256, 256, 32)

    semantic_label = torch.from_numpy(semantic_voxel)
    semantic_label = semantic_label.cuda() if torch.cuda.is_available() else semantic_label
    instance_label = torch.from_numpy(instance_voxel)
    instance_label = instance_label.cuda() if torch.cuda.is_available() else instance_label

    mask_label = prepare_mask_label(semantic_label, instance_label, thing_ids)

    pan_seg, pan_seg_info = convert_mask_label_to_panoptic_output(
        mask_label["labels"], mask_label["masks"], thing_ids
    )

    return pan_seg, pan_seg_info

def prepare_mask_label(semantic_label, instance_label, thing_ids):
    mask_semantic_label = prepare_target(
        semantic_label, ignore_labels=[0, 255]
    ) 
    stuff_filtered_mask = [
        t not in thing_ids for t in mask_semantic_label["labels"]
    ]
    stuff_semantic_labels = mask_semantic_label["labels"][stuff_filtered_mask]
    stuff_semantic_masks = mask_semantic_label["masks"][stuff_filtered_mask]
    labels = [stuff_semantic_labels]
    masks = [stuff_semantic_masks]

    mask_instance_label = prepare_instance_target(
        semantic_target=semantic_label,
        instance_target=instance_label,
        ignore_label=0,
    ) 

    if mask_instance_label is not None: 
        labels.append(mask_instance_label["labels"])
        masks.append(mask_instance_label["masks"])

    mask_label = {
        "labels": torch.cat(labels, dim=0),
        "masks": torch.cat(masks, dim=0),
    }

    return mask_label

def prepare_target(target: torch.Tensor, ignore_labels: List[int]) -> Dict:
    unique_ids = torch.unique(target)
    unique_ids = torch.tensor(
        [unique_id for unique_id in unique_ids if unique_id not in ignore_labels]
    )
    masks = []

    for id in unique_ids:
        masks.append(target == id)
    
    if len(masks) == 0:
        raise SkipIteration("RAISED HERE! Skipping this iteration due to condition ")
    
    masks = torch.stack(masks)
    return {"labels": unique_ids, "masks": masks}

def prepare_instance_target(
    semantic_target: torch.Tensor, instance_target: torch.Tensor, ignore_label: int
) -> Dict:
    unique_instance_ids = torch.unique(instance_target)

    unique_instance_ids = unique_instance_ids[unique_instance_ids != ignore_label]
    masks = []
    semantic_labels = []

    for id in unique_instance_ids:
        masks.append(instance_target == id)
        semantic_labels.append(semantic_target[instance_target == id][0])

    if len(masks) == 0:
        return None

    masks = torch.stack(masks)
    semantic_labels = torch.tensor(semantic_labels)

    return {
        "labels": semantic_labels,
        "masks": masks,
    }

def convert_mask_label_to_panoptic_output(labels, masks, thing_ids):
    """
    labels: [ 0.,  9., 10., 11., 13., 14., 15., 16., 17.,  1.,  1.,  1.,  3.,  5.,....]
    masks: [25, 256, 256, 32]
    """
    segments_info = []
    current_segment_id = 0
    panoptic_seg = torch.zeros(masks.shape[1:])
    stuff_memory_list = {}
    for id, cat_id in enumerate(labels):
        if cat_id == 0: 
            continue
        isthing = cat_id in thing_ids
        mask = masks[id, :, :, :] 

        if not isthing:
            if int(cat_id) in stuff_memory_list.keys():
                panoptic_seg[mask] = stuff_memory_list[int(cat_id)]
                continue
            else:
                stuff_memory_list[int(cat_id)] = current_segment_id + 1

        current_segment_id += 1
        panoptic_seg[mask] = current_segment_id
        segments_info.append(
            {
                "id": current_segment_id,
                "isthing": isthing,
                "category_id": int(cat_id),
                "area": mask.sum(),
            }
        )
    return panoptic_seg, segments_info

def _calculate_sc_scores(ignore_index, preds, targets, nonempty=None):
    preds = preds.clone()
    targets = targets.clone()
    bs = preds.shape[0]

    mask = targets == ignore_index
    preds[mask] = 0
    targets[mask] = 0

    preds = preds.flatten(1)
    targets = targets.flatten(1)
    preds = torch.where(preds > 0, 1, 0)
    targets = torch.where(targets > 0, 1, 0)

    tp, fp, fn = 0, 0, 0
    for i in range(bs):
        pred = preds[i]
        target = targets[i]
        if nonempty is not None:
            nonempty_ = nonempty[i].flatten()
            pred = pred[nonempty_]
            target = target[nonempty_]
        pred = pred.bool()
        target = target.bool()

        tp += torch.logical_and(pred, target).sum()
        fp += torch.logical_and(pred, ~target).sum()
        fn += torch.logical_and(~pred, target).sum()
    return tp, fp, fn

def _calculate_ssc_scores(num_classes, ignore_index, preds, targets, nonempty=None):
    preds = preds.clone()
    targets = targets.clone()
    bs = preds.shape[0]
    C = num_classes

    mask = targets == ignore_index
    preds[mask] = 0
    targets[mask] = 0

    preds = preds.flatten(1)
    targets = targets.flatten(1)

    tp = torch.zeros(C, dtype=torch.int).to(preds.device)
    fp = torch.zeros(C, dtype=torch.int).to(preds.device)
    fn = torch.zeros(C, dtype=torch.int).to(preds.device)
    for i in range(bs):
        pred = preds[i]
        target = targets[i]
        if nonempty is not None:
            mask = nonempty[i].flatten() & (target != ignore_index)
            pred = pred[mask]
            target = target[mask]
        for c in range(C):
            tp[c] += torch.logical_and(pred == c, target == c).sum()
            fp[c] += torch.logical_and(pred == c, ~(target == c)).sum()
            fn[c] += torch.logical_and(~(pred == c), target == c).sum()
    return tp, fp, fn