import torch
from torchmetrics import Metric

from Panoptic.utils import mask_wise_merge
from ..utils import merge_instance_masks
from ._greedy_pq import GreedyPanopticQuality
from ._psc_common import _calculate_sc_scores, _calculate_ssc_scores
from .helper_eval import print_metrics_table_ssc

from typing import List, Dict, Union
from torch import Tensor
class GreedyPSCMetrics(Metric):
    """
    Wrapper class for GreedyPanopticQuality to compute PQ, SQ, and RQ using David's greedy search 
    implementation, and add SSCMetric calculation on top of it.
    """
    def __init__(
        self,
        num_classes: int,
        with_logits: bool,
        thing_classes: List[int],
        stuff_classes: List[int],
        ignore_index: int = 255,
        empty_id: int = 0,
        invalid_id: int = 255,
        iou_threshold: float = 0.5,
        mod_iou_threshold: float = 0.0,
        class_names: List[str] = None,
        print_out: bool = False,
        debug: bool = False,
        use_raw_sem_head_output: bool = False,
        inst_from_mask_wise_merge: bool = True,
        object_mask_threshold: float = 0.25,
        query_threshold: float = 0.7
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.use_raw_sem_head_output = use_raw_sem_head_output

        self.object_mask_threshold = object_mask_threshold
        self.query_threshold = query_threshold
        self.inst_from_mask_wise_merge = inst_from_mask_wise_merge

        self.thing_classes = thing_classes
        self.stuff_classes = stuff_classes
        self.class_names = class_names
        self.print_out = print_out
        self.debug = debug
        self.with_logits = with_logits

        for metric in ('tp_sc', 'fp_sc', 'fn_sc'):
            self.add_state(metric, torch.tensor(0), dist_reduce_fx='sum')
        for metric in ('tps_ssc', 'fps_ssc', 'fns_ssc'):
            self.add_state(metric, torch.zeros(num_classes), dist_reduce_fx='sum')

        self.greedy_pq = GreedyPanopticQuality(
            thing_classes=thing_classes,
            stuff_classes=stuff_classes,
            empty_id=empty_id,
            invalid_id=invalid_id,
            iou_threshold=iou_threshold,
            mod_iou_threshold=mod_iou_threshold
        )

    def update(self, preds: Dict, target: Dict) -> None:
        """
        Update states by accumulating the TP, FN, FP, IoU, etc.
        All states will be set to 0 with call of evaluator.reset() in lit_module.py

        Args:
            preds: Dict, keys: ['ssc_logits', 'voxel_logits', 'query_logits'] or ['final_ssc_pred', 'merged_instance_pred', 'semantic_pred', 'raw_instance_pred']
            target: Dict, keys: ["semantic_label", 'instance_label', 'global_frustum_mask_1'] or ['sem_target', 'instance_label']
        Raises:
            AssertionError: If the batch dim is not expanded.
        """
        if self.with_logits:
            voxel_logits = preds['voxel_logits']
            query_logits = preds['query_logits']
            ssc_logits = preds['ssc_logits']
            frustum_masks = target['global_frustum_mask_1']
            pq_preds_postprocess = mask_wise_merge(
                voxel_logits=voxel_logits, 
                query_logits=query_logits , 
                ssc_logits=ssc_logits, 
                thing_classes=self.thing_classes, 
                frustrum_masks=frustum_masks,
            )

            raw_instance_pred = merge_instance_masks(
                voxel_probs=pq_preds_postprocess['voxel_probs'], 
                query_probs=pq_preds_postprocess['query_probs'], 
                query_class_pred=pq_preds_postprocess['query_class_pred'], 
                thing_classes=self.thing_classes,
                object_mask_threshold=self.object_mask_threshold, 
                query_threshold=self.query_threshold
            )
            sem_preds = torch.argmax(preds['ssc_logits'], dim=1) if self.use_raw_sem_head_output else pq_preds_postprocess['final_ssc_pred']
            inst_preds = raw_instance_pred if not self.inst_from_mask_wise_merge else pq_preds_postprocess['merged_instance_pred']
            sem_label = target["semantic_label"]
            inst_label = target['instance_label']
        else:
            sem_preds = preds['semantic_pred'] if self.use_raw_sem_head_output else preds['final_ssc_pred']
            inst_preds = preds['raw_instance_pred'] if not self.inst_from_mask_wise_merge else preds['merged_instance_pred']
            sem_label = target['sem_target']
            inst_label = target['instance_label']
            if torch.cuda.is_available():
                sem_preds = sem_preds.cuda()
                sem_label = sem_label.cuda()
                inst_preds = inst_preds.cuda()
                inst_label = inst_label.cuda()

        mask = sem_label != self.ignore_index
        tp, fp, fn = _calculate_sc_scores(
            ignore_index=self.ignore_index, 
            preds=sem_preds, 
            targets=sem_label, 
            nonempty=mask
        )
        self.tp_sc += tp
        self.fp_sc += fp
        self.fn_sc += fn

        tp, fp, fn = _calculate_ssc_scores(
            num_classes=self.num_classes, 
            ignore_index=self.ignore_index, 
            preds=sem_preds, 
            targets=sem_label, 
            nonempty=mask
        )
        self.tps_ssc += tp
        self.fps_ssc += fp
        self.fns_ssc += fn

        sem_preds = sem_preds.squeeze()
        sem_label = sem_label.squeeze()
        inst_preds = inst_preds.squeeze()
        inst_label = inst_label.squeeze()
        assert sem_preds.shape == sem_label.shape == inst_preds.shape == inst_label.shape, print(f"Expect shape [H, W, D], but got {sem_preds.shape}, {sem_label.shape}, {inst_preds.shape}, {inst_label.shape}")
        assert len(sem_preds.shape) == 3, print(f"Expect no batch dim after squeeze, but got {sem_preds.shape}. This evaluator support batch size 1 for now.")
        self.greedy_pq.update(sem_preds, inst_preds, sem_label, inst_label)
        

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

        classwise_metrics, grouped_pq, total_pq = self.greedy_pq.compute(show_results=self.print_out)
        if self.print_out:
            print_metrics_table_ssc(precision, recall, iou, ious, self.class_names)

        if self.debug:
            self.greedy_pq.display_stats()

        class_ids = self.thing_classes + self.stuff_classes
        per_class_return = {
            'pq_per_class': [],
            'sq_per_class': [], 
            'rq_per_class': []
        }
        for class_id in class_ids:
            if class_id in classwise_metrics:
                per_class_return['pq_per_class'].append(classwise_metrics[class_id]['PQ'])
                per_class_return['sq_per_class'].append(classwise_metrics[class_id]['SQ']) 
                per_class_return['rq_per_class'].append(classwise_metrics[class_id]['RQ'])
            else:
                per_class_return['pq_per_class'].append(0)
                per_class_return['sq_per_class'].append(0)
                per_class_return['rq_per_class'].append(0)


        return {
            'ssc_Precision': precision,
            'ssc_Recall': recall,
            'ssc_IoU': iou,
            'ssc_iou_per_class': ious,
            'ssc_mIoU': ious[1:].mean(),
            'pq_dagger': total_pq['PQ_mod'],
            'pq_all': total_pq['PQ'],
            'sq_all': total_pq['SQ'],
            'rq_all': total_pq['RQ'],
            'pq_things': grouped_pq['thing']['PQ'],
            'sq_things': grouped_pq['thing']['SQ'],
            'rq_things': grouped_pq['thing']['RQ'],
            'pq_stuff': grouped_pq['stuff']['PQ'],
            'sq_stuff': grouped_pq['stuff']['SQ'],
            'rq_stuff': grouped_pq['stuff']['RQ'],
            'pq_per_class': per_class_return['pq_per_class'],
            'sq_per_class': per_class_return['sq_per_class'],
            'rq_per_class': per_class_return['rq_per_class'],
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

        self.greedy_pq.reset()