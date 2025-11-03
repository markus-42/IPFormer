import torch
from torch import Tensor
from torchmetrics import Metric
from typing import List, Dict, Union
from Panoptic.transformer.transformer_utils import panoptic_segmentation_inference
from Panoptic.utils import mask_wise_merge
from ..utils import merge_instance_masks
from ._psc_common import PQStat
from .helper_eval import (
    print_metrics_table_panop_avg,
    print_metrics_table_panop_per_class,
    print_metrics_table_ssc,
)

from .helper_eval import update_states_psc_pasco_format


class PSCMetrics(Metric):
    """
    Computes Panoptic Quality (PQ), Segmentation Quality (SQ), and Recognition Quality (RQ) using model's logits outputs with PaSCo's evaluation protocol.

    This class combines the code of SSCMetric with the code of PQStat.
    Nesting Metric classes inside Metric is not recommended since it can lead to undefined behaviors.
    https://lightning.ai/docs/torchmetrics/stable/pages/overview.html
    """

    def __init__(
        self,
        num_classes: int,
        with_logits: bool,
        thing_classes: List[int],
        stuff_classes: List[int],
        class_names: List[str],
        ignore_index: int = 255,
        print_out: bool = False,
        stuff_from_ssc = False,
        debug: bool = False,
        overlap_flag=True,
        dual_head=False,
        object_mask_threshold: float = 0.0, 
        query_threshold: float = 0.0, 
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.use_raw_sem_head_output = "False"
        self.dual_head = dual_head
        self.stuff_from_ssc = stuff_from_ssc

        self.object_mask_threshold = object_mask_threshold
        self.query_threshold = query_threshold
        self.overlap_flag = overlap_flag

        self.thing_ids = thing_classes
        self.stuff_ids = stuff_classes
        self.class_names = class_names
        self.print_out = print_out
        self.debug = debug
        self.with_logits = with_logits
        

        for metric in ("tp_sc", "fp_sc", "fn_sc"):
            self.add_state(metric, torch.tensor(0), dist_reduce_fx="sum")
        for metric in ("tps_ssc", "fps_ssc", "fns_ssc"):
            self.add_state(metric, torch.zeros(num_classes), dist_reduce_fx="sum")

        self.pq_stat = PQStat()

    def update(self, preds: Dict, target: Dict) -> None:
        """
        Update states by accumulating TP, FN, FP, IoU, etc.
        Ensures correct semantic and instance predictions for evaluation.

        Args:
            preds: Dict containing model predictions.
            target: Dict containing ground truth labels.
        Raises:
            AssertionError: If batch dimensions do not match expectations.
        """
        if self.with_logits:
            voxel_logits = preds["voxel_logits"] 
            query_logits = preds["query_logits"] 

            panop_out = panoptic_segmentation_inference(
                voxel_output=voxel_logits,
                query_output=query_logits,
                thing_ids=self.thing_ids,
                overlap_flag=self.overlap_flag
            )
            if self.dual_head:
                sem_preds = torch.argmax(preds['ssc_logits'], dim=1) 
                panop_preds = panop_out["panoptic_seg"] 
                pred_segments_info = panop_out["segments_info"]
                if self.stuff_from_ssc:
                    for b in range(panop_preds.shape[0]):
                        panop = panop_preds[b]
                        ssc_sem = sem_preds[b]
                        segments = pred_segments_info[b]

                        thing_mask = torch.zeros_like(panop, dtype=torch.bool)

                        stuff_instance_ids = [seg['id'] for seg in segments if not seg['isthing']]
                        for sid in stuff_instance_ids:
                            panop[panop == sid] = 0 
                        pred_segments_info[b] = [seg for seg in segments if seg['isthing']]

                        current_max_id = max([seg['id'] for seg in pred_segments_info[b]] + [0]) + 1
                        for class_id in self.stuff_ids:
                            replace_mask = (ssc_sem == class_id) & (~thing_mask)
                            if replace_mask.sum() == 0:
                                continue

                            new_instance_id = current_max_id
                            current_max_id += 1

                            panop[replace_mask] = new_instance_id

                            pred_segments_info[b].append({
                                'id': new_instance_id,
                                'category_id': class_id,
                                'isthing': False
                            })

                        panop_preds[b] = panop

            else:
                sem_preds = panop_out["semantic_perd"] 
                panop_preds = panop_out["panoptic_seg"] 
                pred_segments_info = panop_out["segments_info"]

        else:
            sem_preds = preds["semantic_perd"] 
            panop_preds = preds["panoptic_seg"] 
            pred_segments_info = preds["segments_info"]
            
        sem_label = target["semantic_label"] 
        inst_label = target["instance_label"] 
        

        sem_preds = sem_preds.unsqueeze(0) if len(sem_preds.shape) == 3 else sem_preds
        sem_label = sem_label.unsqueeze(0) if len(sem_label.shape) == 3 else sem_label
        inst_label = (
            inst_label.unsqueeze(0) if len(inst_label.shape) == 3 else inst_label
        )

        pq_target = torch.stack((sem_label, inst_label), dim=-1) 

        assert (
            len(pq_target.shape) == 5
        ), f"Expected shape [B, D, H, W, 2], but got {pq_target.shape}"

        update_states_psc_pasco_format(
            psc_metric=self,
            panoptic_pred=panop_preds,
            segments_info=pred_segments_info,
            semantic_pred=sem_preds,
            pq_target=pq_target,
        )

    def compute(self) -> Dict[str, Union[float, List[float], Tensor]]:
        """
        Compute the metrics of SSC and PanopticQuality at the end of the iteration through the dataset.
        """
        if self.tp_sc != 0:
            precision = self.tp_sc / (self.tp_sc + self.fp_sc)
            recall = self.tp_sc / (self.tp_sc + self.fn_sc)
            iou = self.tp_sc / (self.tp_sc + self.fp_sc + self.fn_sc)
        else:
            precision, recall, iou = 0, 0, 0
        ious = self.tps_ssc / (self.tps_ssc + self.fps_ssc + self.fns_ssc + 1e-6)

        metrics = [("All", None), ("Things", True), ("Stuff", False)]
        results = {}
        for name, isthing in metrics:
            results[name], per_class_results = self.pq_stat.pq_average(
                isthing=isthing, thing_ids=self.thing_ids, ignore_cat_id=0
            )
            if name == "All":
                results["per_class"] = per_class_results

        per_class_return = {}
        metrics_list = ["pq", "sq", "rq"]
        for metric in metrics_list:
            ts = []
            for i in range(1, len(self.class_names)):
                if i in results["per_class"]:
                    ts.append(results["per_class"][i][metric])
                else:
                    ts.append(0)
            per_class_return[metric + "_per_class"] = ts

        if self.print_out:
            print_metrics_table_panop_avg(results, metrics)
            print_metrics_table_panop_per_class(
                [self.pq_stat], results, self.class_names
            )
            print_metrics_table_ssc(precision, recall, iou, ious, self.class_names)

        if self.debug:
            self.pq_stat.print_state(ignore_cat_id=0)

        return {
            "ssc_Precision": precision,
            "ssc_Recall": recall,
            "ssc_IoU": iou,
            "ssc_iou_per_class": ious,
            "ssc_mIoU": ious[1:].mean(),
            "pq_dagger": results["All"]["pq_dagger"],
            "pq_all": results["All"]["pq"],
            "sq_all": results["All"]["sq"],
            "rq_all": results["All"]["rq"],
            "pq_things": results["Things"]["pq"],
            "sq_things": results["Things"]["sq"],
            "rq_things": results["Things"]["rq"],
            "pq_stuff": results["Stuff"]["pq"],
            "sq_stuff": results["Stuff"]["sq"],
            "rq_stuff": results["Stuff"]["rq"],
            "pq_per_class": per_class_return["pq_per_class"],
            "sq_per_class": per_class_return["sq_per_class"],
            "rq_per_class": per_class_return["rq_per_class"],
        }

    def reset(self) -> None:
        """Reset metric state variables to their default value."""
        self._update_count = 0
        self._forward_cache = None
        self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, torch.Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                getattr(self, attr).clear() 

        self._cache = None
        self._is_synced = False

        self.pq_stat.reset()