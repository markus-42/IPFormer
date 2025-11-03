import torch
from torchmetrics import Metric
from ._psc_common import _calculate_sc_scores, _calculate_ssc_scores


class SSCMetrics(Metric):

    def __init__(
            self, 
            num_classes, 
            use_raw_sem_head_output=False,
            with_logits=True, 
            ignore_index=255
    ):
        super().__init__()
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.with_logits = with_logits
        self.use_raw_sem_head_output = use_raw_sem_head_output

        for metric in ('tp_sc', 'fp_sc', 'fn_sc'):
            self.add_state(metric, torch.tensor(0), dist_reduce_fx='sum')
        for metric in ('tps_ssc', 'fps_ssc', 'fns_ssc'):
            self.add_state(metric, torch.zeros(num_classes), dist_reduce_fx='sum')

    def update(self, preds, target):
        if self.with_logits:
            sem_preds = torch.argmax(preds['ssc_logits'], dim=1)
            sem_target = target['target']
        else:
            try:
                sem_preds = preds['final_ssc_pred'] if not self.use_raw_sem_head_output else preds['semantic_pred']
                sem_target = target['sem_target']
            except KeyError:
                sem_preds = preds['pred']
                sem_target = target['sem_label']
        if sem_preds is None or sem_target is None:
            raise KeyError(f"No data with keys 'final_ssc_pred' or 'semantic_pred' or 'sem_target' or 'sem_label' in preds or target")
        mask = sem_target != self.ignore_index

        tp, fp, fn = _calculate_sc_scores(
            ignore_index=self.ignore_index, 
            preds=sem_preds, 
            targets=sem_target, 
            nonempty=mask
        )
        self.tp_sc += tp
        self.fp_sc += fp
        self.fn_sc += fn

        tp, fp, fn = _calculate_ssc_scores(
            num_classes=self.num_classes, 
            ignore_index=self.ignore_index, 
            preds=sem_preds, 
            targets=sem_target, 
            nonempty=mask
        )
        self.tps_ssc += tp
        self.fps_ssc += fp
        self.fns_ssc += fn

    def compute(self):
        if self.tp_sc != 0:
            precision = self.tp_sc / (self.tp_sc + self.fp_sc)
            recall = self.tp_sc / (self.tp_sc + self.fn_sc)
            iou = self.tp_sc / (self.tp_sc + self.fp_sc + self.fn_sc)
        else:
            precision, recall, iou = 0, 0, 0
        ious = self.tps_ssc / (self.tps_ssc + self.fps_ssc + self.fns_ssc + 1e-6)
        return {
            'ssc_Precision': precision,
            'ssc_Recall': recall,
            'ssc_IoU': iou,
            'ssc_iou_per_class': ious,
            'ssc_mIoU': ious[1:].mean()
        }