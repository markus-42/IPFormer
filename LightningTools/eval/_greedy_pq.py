import numpy as np
import os
import pickle
import torch
from tqdm import tqdm

from typing import Dict, List, Tuple, Union


PRED_DIR = "/home/jovyan/workspace-dn-symphonies-infer-env/symphonies/outputs/baseline-infer/panoptic"
GT_DIR = PRED_DIR

THING_CLASSES = [1, 2, 3, 4, 5, 6, 7, 8]
STUFF_CLASSES = [9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
EMPTY_CLASS = 0
INVALID_CLASS = 255
INVALID_INSTANCE = 255
EMPTY_AND_STUFF_INSTANCE_ID = 0

PQ_IOU_THRESH = 0.5

PRED_EQU_GT = False 


class ClassStats:
    """
    Keeps track of the true positives, false positives, false negatives and IoU for a class
    """
    def __init__(
            self
            ) -> None:
        self.tp = 0
        self.fp = 0
        self.fn = 0
        self.iou = 0.0

    def update(
            self,
            tp: int = 0,
            fp: int = 0,
            fn: int = 0,
            iou: float = 0.0
            ) -> None:
        """
        Updates the class statistics with the given values
        """
        self.tp += tp
        self.fp += fp
        self.fn += fn
        self.iou += iou


class GreedyPanopticQuality:
    """
    Computes the Panoptic Quality (PQ) metric for a given set of thing and stuff classes using a greedy matching
    strategy.
    """
    def __init__(
            self,
            thing_classes: List[int],
            stuff_classes: List[int],
            empty_id: int = 0,
            invalid_id: int = 255,
            iou_threshold: float = 0.5,
            mod_iou_threshold: float = 0.0
            ) -> None:
        assert len(set(thing_classes) & set(stuff_classes)) == 0, "Thing and stuff classes must be disjoint"
        assert empty_id not in thing_classes + stuff_classes, "Empty class cannot be a thing or stuff class"
        assert invalid_id not in thing_classes + stuff_classes, "Invalid class cannot be a thing or stuff class"
        assert empty_id != invalid_id, "Empty and invalid classes must be different"
        assert 0 <= iou_threshold <= 1, "IoU threshold must be between 0 and 1"
        assert 0 <= mod_iou_threshold <= 1, "Modified IoU threshold must be between 0 and 1"

        self.thing_classes = thing_classes
        self.stuff_classes = stuff_classes
        self.empty_id = empty_id
        self.invalid_id = invalid_id
        self.iou_threshold = iou_threshold
        self.mod_iou_threshold = mod_iou_threshold

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.reset()

    def reset(
            self
            ) -> None:
        """
        Resets the class statistics for both the original and modified IoU thresholds.
        """
        self.class_stats = {class_id: ClassStats() for class_id in self.thing_classes + self.stuff_classes}
        self.mod_class_stats = {class_id: ClassStats() for class_id in self.thing_classes + self.stuff_classes}

    def update(
            self,
            sem_pred: torch.Tensor,
            inst_pred: torch.Tensor,
            sem_gt: torch.Tensor,
            inst_gt: torch.Tensor
            ) -> None:
        """
        Updates the class statistics based on the predictions and labels for a single image.

        Parameters
        ----------
        sem_pred : torch.Tensor
            The predicted semantic labels for the image
        inst_pred : torch.Tensor
            The predicted instance labels for the image
        sem_gt : torch.Tensor
            The ground truth semantic labels for the image
        inst_gt : torch.Tensor
            The ground truth instance labels for the image
        """
        assert sem_pred.shape == inst_pred.shape == sem_gt.shape == inst_gt.shape, "Tensors must have the same shape"

        sem_pred, inst_pred = sem_pred.to(self.device), inst_pred.to(self.device)
        sem_gt, inst_gt = sem_gt.to(self.device), inst_gt.to(self.device)
        sem_pred, inst_pred, inst_gt = self.mark_invalid_class(sem_pred, sem_gt, inst_pred, inst_gt)

        for class_id in self.stuff_classes:
            sem_pred_class = (sem_pred == class_id)
            sem_gt_class = (sem_gt == class_id)

            if self._check_and_update_edge_cases_stuff(sem_pred_class, sem_gt_class, class_id):
                continue

            iou = self.calc_iou(sem_pred_class, sem_gt_class)
            if iou > self.iou_threshold:
                self.class_stats[class_id].update(tp=1, iou=iou)
                self.mod_class_stats[class_id].update(tp=1, iou=iou)
            elif iou > self.mod_iou_threshold:
                self.class_stats[class_id].update(fp=1, fn=1)
                self.mod_class_stats[class_id].update(tp=1, iou=iou)
            else:
                self.class_stats[class_id].update(fp=1, fn=1)
                self.mod_class_stats[class_id].update(fp=1, fn=1)

        inst_masks_pred = self.extract_instance_masks(sem_pred, inst_pred)
        inst_masks_gt = self.extract_instance_masks(sem_gt, inst_gt)

        if self._check_and_update_edge_cases_things_all(inst_masks_pred, inst_masks_gt,
                                                        sem_pred, inst_pred, sem_gt, inst_gt):
            return
        
        for class_id in self.thing_classes:
            inst_masks_pred_class = inst_masks_pred['inst_masks'][inst_masks_pred['sem_values'] == class_id] 
            inst_masks_gt_class = inst_masks_gt['inst_masks'][inst_masks_gt['sem_values'] == class_id] 

            if self._check_and_update_edge_cases_things_class(inst_masks_pred_class, inst_masks_gt_class, class_id):
                continue            
        
            iou_inst_pairs = []
            for id_pred, mask_pred in enumerate(inst_masks_pred_class, start=1):
                for id_gt, mask_gt in enumerate(inst_masks_gt_class, start=1):
                    iou = self.calc_iou(mask_pred, mask_gt)
                    iou_inst_pairs.append((iou, id_pred, id_gt))

            iou_inst_pairs.sort(key=lambda x: x[0], reverse=True)
            matched_ids_pred, matched_ids_gt = set(), set()
            for iou, id_pred, id_gt in iou_inst_pairs:
                if iou < self.iou_threshold:
                    continue

                if id_pred not in matched_ids_pred and id_gt not in matched_ids_gt:
                    self.class_stats[class_id].update(tp=1, iou=iou)
                    matched_ids_pred.add(id_pred)
                    matched_ids_gt.add(id_gt)

            self.class_stats[class_id].update(fp=inst_masks_pred_class.size(0) - len(matched_ids_pred))
            self.class_stats[class_id].update(fn=inst_masks_gt_class.size(0) - len(matched_ids_gt))
            
    def compute(
            self,
            show_results: bool = False
            ) -> Tuple[Dict[int, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, float]]:
        """
        Computes the Panoptic Quality (PQ) metric for the given set of classes.

        Parameters
        ----------
        show_results : bool, optional
            Whether to display the results, by default False

        Returns
        -------
        Tuple[Dict[int, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, float]]
            The classwise, grouped and total PQ metrics
        """
        classwise_metrics = {class_id: {'PQ_mod': 0.0, 'PQ': 0.0, 'SQ': 0.0, 'RQ': 0.0}
                             for class_id in self.thing_classes + self.stuff_classes}
        grouped_metrics = {'thing': {'PQ_mod': 0.0, 'PQ': 0.0, 'SQ': 0.0, 'RQ': 0.0},
                           'stuff': {'PQ_mod': 0.0, 'PQ': 0.0, 'SQ': 0.0, 'RQ': 0.0}}
        total_metrics = {'PQ_mod': 0.0, 'PQ': 0.0, 'SQ': 0.0, 'RQ': 0.0}

        class_count, thing_count, stuff_count = 0, 0, 0
        for class_id in self.thing_classes + self.stuff_classes:
            stats = self.class_stats[class_id]
            mod_stats = self.mod_class_stats[class_id]
            tp, fp, fn, iou = stats.tp, stats.fp, stats.fn, stats.iou

            if tp + fp + fn == 0:
                continue

            sq = iou / tp if tp != 0 else 0.0
            rq = tp / (tp + 0.5 * fp + 0.5 * fn)
            pq = sq * rq

            if class_id in self.stuff_classes:
                tp_mod, fp_mod, fn_mod, iou_mod = mod_stats.tp, mod_stats.fp, mod_stats.fn, mod_stats.iou
                pq_mod = iou_mod / (tp_mod + fn_mod) if (tp_mod + fn_mod) != 0 else 0.0
            else:
                pq_mod = pq

            classwise_metrics[class_id] = {
                'PQ_mod': pq_mod,
                'PQ': pq,
                'SQ': sq,
                'RQ': rq
            }

            if class_id in self.thing_classes:
                group = 'thing'
                thing_count += 1
            else:
                group = 'stuff'
                stuff_count += 1

            grouped_metrics[group]['PQ_mod'] += pq_mod
            grouped_metrics[group]['PQ'] += pq
            grouped_metrics[group]['SQ'] += sq
            grouped_metrics[group]['RQ'] += rq

            total_metrics['PQ_mod'] += pq_mod
            total_metrics['PQ'] += pq
            total_metrics['SQ'] += sq
            total_metrics['RQ'] += rq

            class_count += 1

        for metric in ['PQ_mod', 'PQ', 'SQ', 'RQ']:
            if thing_count > 0:
                grouped_metrics['thing'][metric] /= thing_count
            if stuff_count > 0:
                grouped_metrics['stuff'][metric] /= stuff_count
            total_metrics[metric] /= class_count

        if show_results:
            self.display_results(classwise_metrics, grouped_metrics, total_metrics)

        return classwise_metrics, grouped_metrics, total_metrics

    def mark_invalid_class(
            self,
            sem_pred: torch.Tensor,
            sem_gt: torch.Tensor,
            inst_pred: torch.Tensor,
            inst_gt: torch.Tensor
            ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Marks the invalid class in the predictions and labels.

        Parameters
        ----------
        sem_pred : torch.Tensor
            The predicted semantic labels
        sem_gt : torch.Tensor
            The ground truth semantic labels
        inst_pred : torch.Tensor
            The predicted instance labels
        inst_gt : torch.Tensor
            The ground truth instance labels

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            The updated predicted semantic labels, predicted instance labels and ground truth instance labels
        """
        invalid_mask = (sem_gt == self.invalid_id)
        sem_pred[invalid_mask] = self.invalid_id
        inst_pred[invalid_mask] = self.invalid_id
        inst_gt[invalid_mask] = self.invalid_id
        return sem_pred, inst_pred, inst_gt

    def extract_instance_masks(
            self,
            sem: torch.Tensor,
            inst: torch.Tensor
            ) -> Union[Dict[str, torch.Tensor], None]:
        """
        Extracts the instance masks and their corresponding semantic labels.

        Parameters
        ----------
        sem : torch.Tensor
            The semantic labels for the image
        inst : torch.Tensor
            The instance labels for the image

        Returns
        -------
        Union[Dict[int, Dict[str, torch.Tensor]], None]
            A dictionary containing the instance masks and their corresponding semantic labels
        """
        inst_ids = torch.unique(inst)
        inst_ids = inst_ids[inst_ids != self.invalid_id]
        inst_ids = inst_ids[inst_ids != self.empty_id]

        inst_masks, sem_values = [], []
        for id in inst_ids:
            inst_mask = (inst == id)
            sem_value = sem[inst_mask][0]

            if sem_value in self.thing_classes:
                inst_masks.append(inst_mask)
                sem_values.append(sem_value)

        if len(inst_masks) == 0:
            return None
        
        inst_masks = torch.stack(inst_masks).to(self.device)
        sem_values = torch.tensor(sem_values).to(self.device)
        return {
            'inst_masks': inst_masks,
            'sem_values': sem_values
        }

    @staticmethod
    def calc_iou(
            pred: torch.Tensor,
            gt: torch.Tensor
            ) -> float:
        """
        Calculates the Intersection over Union (IoU) between the predicted and ground truth masks.

        Parameters
        ----------
        pred : torch.Tensor
            The predicted mask
        gt : torch.Tensor
            The ground truth mask

        Returns
        -------
        float
            The IoU between the predicted and ground truth masks
        """
        intersection = (pred & gt).sum().item()
        union = (pred | gt).sum().item()
        return intersection / union if union != 0 else 0.0

    def _check_and_update_edge_cases_stuff(
            self,
            sem_pred_class: torch.Tensor,
            sem_gt_class: torch.Tensor,
            class_id: int
            ) -> bool:
        """
        Checks for edge cases in stuff classes where the predicted or ground truth masks are empty.

        Parameters
        ----------
        sem_pred_class : torch.Tensor
            The predicted semantic labels for the current class
        sem_gt_class : torch.Tensor
            The ground truth semantic labels for the current class
        class_id : int
            The class id of the current class

        Returns
        -------
        bool
            Whether an edge case was encountered
        """
        if sem_pred_class.sum().item() == 0 and sem_gt_class.sum().item() == 0:
            return True
        if sem_pred_class.sum().item() == 0:
            self.class_stats[class_id].update(fn=1)
            self.mod_class_stats[class_id].update(fn=1)
            return True
        if sem_gt_class.sum().item() == 0:
            self.class_stats[class_id].update(fp=1)
            self.mod_class_stats[class_id].update(fp=1)
            return True
        return False

    def _check_and_update_edge_cases_things_all(
            self,
            inst_masks_pred: Union[Dict[str, torch.Tensor], None],
            inst_masks_gt: Union[Dict[str, torch.Tensor], None],
            sem_pred: torch.Tensor,
            inst_pred: torch.Tensor,
            sem_gt: torch.Tensor,
            inst_gt: torch.Tensor
            ) -> bool:
        """
        Checks for edge cases where the predictions or ground truth don't contain any instances.

        Parameters
        ----------
        inst_masks_pred : Union[Dict[int, Dict[str, torch.Tensor]], None]
            The predicted instance masks and their corresponding semantic labels
        inst_masks_gt : Union[Dict[int, Dict[str, torch.Tensor]], None]
            The ground truth instance masks and their corresponding semantic labels
        sem_pred : torch.Tensor
            The predicted semantic labels
        inst_pred : torch.Tensor
            The predicted instance labels
        sem_gt : torch.Tensor
            The ground truth semantic labels
        inst_gt : torch.Tensor
            The ground truth instance labels

        Returns
        -------
        bool
            Whether an edge case was encountered
        """
        if inst_masks_pred is None and inst_masks_gt is None:
            return True
        if inst_masks_pred is None:
            for class_id in self.thing_classes:
                sem_gt_class = (sem_gt == class_id)
                num_inst = torch.unique(inst_gt[sem_gt_class]).numel()
                self.class_stats[class_id].update(fn=num_inst)
            return True
        if inst_masks_gt is None:
            for class_id in self.thing_classes:
                sem_pred_class = (sem_pred == class_id)
                num_inst = torch.unique(inst_pred[sem_pred_class]).numel()
                self.class_stats[class_id].update(fp=num_inst)
            return True
        return False

    def _check_and_update_edge_cases_things_class(
            self,
            inst_masks_pred_class: torch.Tensor,
            inst_masks_gt_class: torch.Tensor,
            class_id: int
            ) -> bool:
        """
        Checks for edge cases in thing classes where the predicted or ground truth masks are empty.

        Parameters
        ----------
        inst_masks_pred_class : torch.Tensor
            The predicted instance masks for the current class
        inst_masks_gt_class : torch.Tensor
            The ground truth instance masks for the current class
        class_id : int
            The class id of the current class

        Returns
        -------
        bool
            Whether an edge case was encountered
        """
        if inst_masks_pred_class.size(0) == 0 and inst_masks_gt_class.size(0) == 0:
            return True
        if inst_masks_pred_class.size(0) == 0:
            self.class_stats[class_id].update(fn=inst_masks_gt_class.size(0))
            return True
        if inst_masks_gt_class.size(0) == 0:
            self.class_stats[class_id].update(fp=inst_masks_pred_class.size(0))
            return True
        return False

    def display_stats(
            self
            ) -> None:
        """
        Displays the true positives, false positives, false negatives and IoU for each class.
        """
        print("==== class stats ==========")
        print("Class     TP     FP     FN        IoU")
        print("-------------------------------------")
        for class_id, stats in self.class_stats.items():
            print(f"{class_id:5}  {stats.tp:5}  {stats.fp:5}  {stats.fn:5}  {stats.iou:9.4f}")
        print()

    @staticmethod
    def display_results(
            classwise_metrics: Dict[int, Dict[str, float]],
            grouped_metrics: Dict[str, Dict[str, float]],
            total_metrics: Dict[str, float]
            ) -> None:
        """
        Displays the classwise, grouped and total PQ metrics.

        Parameters
        ----------
        classwise_metrics : Dict[int, Dict[str, float]
            The classwise PQ metrics
        grouped_metrics : Dict[str, Dict[str, float]
            The grouped PQ metrics
        total_metrics : Dict[str, float]
            The total PQ metrics
        """
        print('==== classwise ============')
        print("Class    PQ_mod        PQ        SQ        RQ")
        print("---------------------------------------------")
        for class_id, metrics in classwise_metrics.items():
            print(f"{class_id:5} ", *(f"{(metric * 100):8.4f} " for metric in metrics.values()))
        print()

        print('==== grouped ==============')
        print("Group    PQ_mod        PQ        SQ        RQ")
        print("---------------------------------------------")
        for group, metrics in grouped_metrics.items():
            print(f"{group:5} ", *(f"{(metric * 100):8.4f} " for metric in metrics.values()))
        print()

        print('==== total ================')
        print("         PQ_mod        PQ        SQ        RQ")
        print("--------------------------------------")
        print('      ', *(f"{(metric * 100):8.4f} " for metric in total_metrics.values()))
        print()


def extract_pickle(
    pkl_file_pred: str,
    pkl_file_gt: str,
    sem_pred_key: str,
    inst_pred_key: str,
    sem_gt_key: str,
    inst_gt_key: str,
    trace_gt_key: str = '',
    empty_class: int = 0,
    empty_inst: int = 0,
    sanity_check: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extracts the semantic and instance predictions and labels from a pickle file and returns them as tensors.

    Parameters
    ----------
    pkl_file_pred : str
        The path to the pickle file containing the predictions
    pkl_file_gt : str
        The path to the pickle file containing the ground truth labels
    sem_pred_key : str
        The key to extract the predicted semantic labels
    inst_pred_key : str
        The key to extract the predicted instance labels
    sem_gt_key : str
        The key to extract the ground truth semantic labels
    inst_gt_key : str
        The key to extract the ground truth instance labels
    trace_gt_key : str, optional
        The key to extract the ground truth trace labels, by default None
    empty_class : int, optional
        The class id to replace the trace labels with, by default 0
    empty_inst : int, optional
        The instance id to replace the trace labels with, by default 0
    sanity_check : bool, optional
        Whether to make the predictions equal to the ground truth, by default False

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        The predicted and ground truth semantic and instance labels as tensors
    """
    with open(pkl_file_pred, 'rb') as f:
        data = pickle.load(f)

    if trace_gt_key == '':
        sem_gt = torch.tensor(data[sem_gt_key].astype(np.uint8))
        inst_gt = torch.tensor(data[inst_gt_key].astype(np.uint8))
    else:
        sem_gt = torch.tensor(data[sem_gt_key].astype(np.uint8))
        inst_gt = torch.tensor(data[inst_gt_key].astype(np.uint8))
        trace_gt = torch.tensor(data[trace_gt_key].astype(np.uint8))

        sem_gt[trace_gt == 1] = empty_class
        inst_gt[trace_gt == 1] = empty_inst

    if sanity_check:
        sem_pred = sem_gt.clone()
        inst_pred = inst_gt.clone()
    else:
        with open(pkl_file_gt, 'rb') as f:
            data = pickle.load(f)

        sem_pred = torch.tensor(data[sem_pred_key].astype(np.uint8))
        inst_pred = torch.tensor(data[inst_pred_key].astype(np.uint8))

    return sem_pred, sem_gt, inst_pred, inst_gt


if __name__ == '__main__':
    panoptic_quality = GreedyPanopticQuality(THING_CLASSES, STUFF_CLASSES, EMPTY_CLASS, INVALID_CLASS, PQ_IOU_THRESH)

    files_pred = sorted([file for file in os.listdir(PRED_DIR) if file.endswith('.pkl')])
    files_gt = sorted([file for file in os.listdir(GT_DIR) if file.endswith('.pkl')])

    for file_pred, file_gt in tqdm(zip(files_pred, files_gt), leave=False):
        file_path_pred = os.path.join(PRED_DIR, file_pred)
        file_path_gt = os.path.join(GT_DIR, file_gt)
        sem_pred, sem_gt, inst_pred, inst_gt = extract_pickle(file_path_pred, file_path_gt,
                                                              sem_pred_key='final_ssc_pred',
                                                              inst_pred_key='merged_instance_pred',
                                                              sem_gt_key='sem_target',
                                                              inst_gt_key='instance_label',
                                                              sanity_check=PRED_EQU_GT)
        sem_pred, inst_pred = sem_pred.squeeze(), inst_pred.squeeze()

        panoptic_quality.update(sem_pred, inst_pred, sem_gt, inst_gt)

    panoptic_quality.display_stats()
    classwise_pq, grouped_pq, total_pq = panoptic_quality.compute(show_results=True)