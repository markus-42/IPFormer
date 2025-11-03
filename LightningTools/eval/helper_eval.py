from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .psc_metric import PSCMetrics
from torch import Tensor
import torch

import numpy as np

from ._psc_common import (
    _calculate_sc_scores,
    _calculate_ssc_scores,
    preprocess_seminst_vox,
    pq_compute_single_core,
)


def update_states_psc_pasco_format(
    psc_metric: "PSCMetrics",
    panoptic_pred: Tensor,
    segments_info,
    semantic_pred: Tensor,
    pq_target: Tensor,
) -> None:
    """Reuseable function for updating all states of PSCMetrics using PaSCo's PQ metric.

    Args:
        psc_metric: the class object of PSCMetrics.
        panoptic_pred: tensor of shape [bs, h, w, d], containing the instance ids of both things and stuff
        segments_info: list of list of dicts, containing the instance metadata
        pq_target: tensor of shape [bs, h, w, d, 2], the first channel is the semantic target
        and the second channel is the instance target.
    Raises:
        AssertionError: if the unpacked semantic and instance voxel grids do not have the same shape.
        TypeError: if psc_metric is not an instance of PSCMetrics.
    """
    bs = panoptic_pred.shape[0]
    for b in range(bs):
        assert panoptic_pred.shape == pq_target[:, :, :, :, 1].shape

        semantic_label_origin_single = (
            pq_target[b, :, :, :, 0].cpu().numpy().astype(np.uint8)
        )
        instance_label_origin_single = (
            pq_target[b, :, :, :, 1].cpu().numpy().astype(np.uint8)
        )

        gt_panoptic_seg, gt_segments_info = preprocess_seminst_vox(
            semantic_label_origin_single,
            instance_label_origin_single,
            thing_ids=psc_metric.thing_ids,
        )
        pred_panoptic_seg, pred_segments_info = panoptic_pred[b], segments_info
        pred_panoptic_seg = pred_panoptic_seg.cpu().numpy()

        invalid_mask = semantic_label_origin_single == psc_metric.ignore_index
        unknown_mask = (
            invalid_mask.detach().cpu().numpy()
            if isinstance(invalid_mask, Tensor)
            else invalid_mask
        )

        gt_panoptic_seg[unknown_mask] = 0
        pred_panoptic_seg[unknown_mask] = 0
        gt_panoptic_seg = (
            gt_panoptic_seg.detach().cpu().numpy()
            if isinstance(gt_panoptic_seg, Tensor)
            else gt_panoptic_seg
        )
        pred_panoptic_seg = (
            pred_panoptic_seg.detach().cpu().numpy()
            if isinstance(pred_panoptic_seg, Tensor)
            else pred_panoptic_seg
        )

        gt_ids = np.unique(gt_panoptic_seg)
        pred_ids = np.unique(pred_panoptic_seg)
        gt_segments_info = [el for el in gt_segments_info if el["id"] in gt_ids]
        pred_segments_info = [el for el in pred_segments_info[0] if el["id"] in pred_ids]

        _ = pq_compute_single_core(
            psc_metric.pq_stat,
            gt_segments_info,
            pred_segments_info,
            gt_panoptic_seg,
            pred_panoptic_seg,
            thing_ids=psc_metric.thing_ids,
        )

    sem_preds = semantic_pred
    sem_target = (
        pq_target[:, :, :, :, 0].cuda()
        if torch.cuda.is_available()
        else pq_target[:, :, :, :, 0]
    )

    assert sem_preds.shape == sem_target.shape
    mask = sem_target != psc_metric.ignore_index
    mask = mask.cuda() if torch.cuda.is_available() else mask

    tp, fp, fn = _calculate_sc_scores(
        ignore_index=psc_metric.ignore_index,
        preds=sem_preds,
        targets=sem_target,
        nonempty=mask,
    )
    psc_metric.tp_sc += tp
    psc_metric.fp_sc += fp
    psc_metric.fn_sc += fn

    tp, fp, fn = _calculate_ssc_scores(
        num_classes=psc_metric.num_classes,
        ignore_index=psc_metric.ignore_index,
        preds=sem_preds,
        targets=sem_target,
        nonempty=mask,
    )
    psc_metric.tps_ssc += tp
    psc_metric.fps_ssc += fp
    psc_metric.fns_ssc += fn


def update_states_psc(
    psc_metric: "PSCMetrics",
    pq_preds: Tensor,
    pq_target: Tensor,
) -> None:
    """Reuseable function for updating all states of PSCMetrics using PaSCo's PQ metric.

    Args:
        psc_metric: the class object of PSCMetrics.
        pq_preds: tensor of shape [bs, h, w, d, 2], the first channel is the semantic prediction
        and the second channel is the instance prediction.
        pq_target: tensor of shape [bs, h, w, d, 2], the first channel is the semantic target
        and the second channel is the instance target.
    Raises:
        AssertionError: if the unpacked semantic and instance voxel grids do not have the same shape.
        TypeError: if psc_metric is not an instance of PSCMetrics.
    """
    bs = pq_preds.shape[0]
    for b in range(bs):
        sem_pred_single = pq_preds[b, :, :, :, 0].cpu().numpy().astype(np.uint8)
        instance_pred_single = pq_preds[b, :, :, :, 1].cpu().numpy().astype(np.uint8)
        semantic_label_origin_single = (
            pq_target[b, :, :, :, 0].cpu().numpy().astype(np.uint8)
        )
        instance_label_origin_single = (
            pq_target[b, :, :, :, 1].cpu().numpy().astype(np.uint8)
        )

        assert semantic_label_origin_single.shape == instance_label_origin_single.shape
        assert sem_pred_single.shape == instance_pred_single.shape
        assert semantic_label_origin_single.shape == instance_label_origin_single.shape

        gt_panoptic_seg, gt_segments_info = preprocess_seminst_vox(
            semantic_label_origin_single,
            instance_label_origin_single,
            thing_ids=psc_metric.thing_ids,
        )
        pred_panoptic_seg, pred_segments_info = preprocess_seminst_vox(
            sem_pred_single, instance_pred_single, thing_ids=psc_metric.thing_ids
        )

        invalid_mask = semantic_label_origin_single == psc_metric.ignore_index
        unknown_mask = (
            invalid_mask.detach().cpu().numpy()
            if isinstance(invalid_mask, Tensor)
            else invalid_mask
        )

        gt_panoptic_seg[unknown_mask] = 0
        pred_panoptic_seg[unknown_mask] = 0
        gt_panoptic_seg = (
            gt_panoptic_seg.detach().cpu().numpy()
            if isinstance(gt_panoptic_seg, Tensor)
            else gt_panoptic_seg
        )
        pred_panoptic_seg = (
            pred_panoptic_seg.detach().cpu().numpy()
            if isinstance(pred_panoptic_seg, Tensor)
            else pred_panoptic_seg
        )

        gt_ids = np.unique(gt_panoptic_seg)
        pred_ids = np.unique(pred_panoptic_seg)
        gt_segments_info = [el for el in gt_segments_info if el["id"] in gt_ids]
        pred_segments_info = [el for el in pred_segments_info if el["id"] in pred_ids]

        _ = pq_compute_single_core(
            psc_metric.pq_stat,
            gt_segments_info,
            pred_segments_info,
            gt_panoptic_seg,
            pred_panoptic_seg,
            thing_ids=psc_metric.thing_ids,
        )

    sem_preds = (
        pq_preds[:, :, :, :, 0].cuda()
        if torch.cuda.is_available()
        else pq_preds[:, :, :, :, 0]
    )
    sem_target = (
        pq_target[:, :, :, :, 0].cuda()
        if torch.cuda.is_available()
        else pq_target[:, :, :, :, 0]
    )
    mask = sem_target != psc_metric.ignore_index
    mask = mask.cuda() if torch.cuda.is_available() else mask

    tp, fp, fn = _calculate_sc_scores(
        ignore_index=psc_metric.ignore_index,
        preds=sem_preds,
        targets=sem_target,
        nonempty=mask,
    )
    psc_metric.tp_sc += tp
    psc_metric.fp_sc += fp
    psc_metric.fn_sc += fn

    tp, fp, fn = _calculate_ssc_scores(
        num_classes=psc_metric.num_classes,
        ignore_index=psc_metric.ignore_index,
        preds=sem_preds,
        targets=sem_target,
        nonempty=mask,
    )
    psc_metric.tps_ssc += tp
    psc_metric.fps_ssc += fp
    psc_metric.fns_ssc += fn


def print_metrics_table_panop_per_class(pq_stats, results, class_names):
    print("=====================================")
    metrics_list = ["pq", "sq", "rq"]
    for metric in metrics_list:
        print("==>", metric)
        print("method" + ", " + (", ".join(class_names[1:])))
        for i in range(len(pq_stats)):
            if i == len(pq_stats) - 1:
                row_name = "ensemble"
            else:
                row_name = "subnet {}".format(i)
            panop_results = results

            ts = []
            for i in range(1, len(class_names)):
                if i in panop_results["per_class"]:
                    ts.append(panop_results["per_class"][i][metric])
                else:
                    ts.append(0)
            print(
                row_name + ", " + (", ".join(["{:0.2f}".format(t * 100) for t in ts]))
            )


def print_metrics_table_panop_avg(results, metrics):
    print("{:15s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"))
    print("-" * (15 + 7 * 4))

    print(f"pq_dagger: {100 * results['All']['pq_dagger']}")

    for name, _isthing in metrics:
        print(
            "{:15s}| {:5.2f}  {:5.2f}  {:5.2f} {:5d}".format(
                name,
                100 * results[name]["pq"],
                100 * results[name]["sq"],
                100 * results[name]["rq"],
                results[name]["n"],
            )
        )
    print("-" * (15 + 7 * 4))


def print_metrics_table_ssc(precision, recall, iou, iou_per_class, class_names):
    print("{:15s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "P", "R", "IoU", "mIoU"))
    print("-" * (15 + 7 * 4))
    print(
        "{:15s}| {:5.2f}  {:5.2f}  {:5.2f} {:5.2f}".format(
            "Value",
            100 * precision,
            100 * recall,
            100 * iou,
            100 * iou_per_class[1:].mean(),
        )
    )
    print("-" * (15 + 7 * 4))
    print("=====================================")
    print("==>", "iou_per_class")
    print("method" + ", " + (", ".join(class_names)))

    row_name = "ensemble"
    print(
        row_name
        + ", "
        + (", ".join(["{:0.2f}".format(t * 100) for t in iou_per_class]))
    )
    print("=====================================")