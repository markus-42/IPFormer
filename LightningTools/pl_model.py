import os
import torch
import numpy as np
import pytorch_lightning as pl
from .basemodel import LightningBaseModel
from LightningTools.eval.psc_metric import PSCMetrics
from mmdet3d.models import build_model
from .utils import get_inv_map
from mmcv.runner.checkpoint import load_checkpoint
import torch.optim as optim
from omegaconf import open_dict
from torch.cuda.amp import autocast
from .excp import SkipIteration

class pl_model(LightningBaseModel):
    def __init__(
        self,
        config):
        super(pl_model, self).__init__(config)

        model_config = config['model']
        self.model = build_model(model_config)
        if 'load_from' in config:
            load_checkpoint(self.model, config['load_from'], map_location='cpu')
        self.model_type = self.model.__class__.__name__
        
        self.num_class = config['num_class']
        self.class_names = config['class_names']
        self.criterion = self.model.loss
        self.train_evaluator = PSCMetrics(**config['evaluator'])
        self.val_evaluator = PSCMetrics(**config['evaluator'])
        self.test_evaluator = PSCMetrics(**config['evaluator'])
        self.save_path = config['save_path']
        self.test_mapping = config['test_mapping']
        self.pretrain = config['pretrain']
        self.thing_classes = config['thing_ids']
        self.stuff_classes = config['stuff_ids']
        all_classes = sorted(self.thing_classes) + sorted(self.stuff_classes)
        self.rearranged_class_names = [self.class_names[i] for i in all_classes] 
              

    

    def forward(self, data_dict):
        return self.model(data_dict)
    
    def _step(self, batch, evaluator=None):
        output_dict = self(batch) 
        with autocast(enabled=False):
            loss = self.criterion(output_dict, batch)
            
        
        if evaluator:
            evaluator.update(output_dict, batch)
        return loss

    def training_step(self, batch, batch_idx):
        try:
            loss = self._step(batch, self.train_evaluator)
            if isinstance(loss, dict):
                loss['loss_total'] = sum(loss.values())
                categorized_losses = self.categorize_losses(loss)  
                loss.update(categorized_losses)
                
                if isinstance(loss, dict):
                    loss['loss_total'] = sum(loss.values())
                    self.log_dict({f'train/{k}': v for k, v in loss.items()}, sync_dist=True, logger=True)
            else:
                self.log('train/loss', loss, sync_dist=True, prog_bar=True, logger=True)

            return sum(loss.values()) if isinstance(loss, dict) else loss
        except SkipIteration as e:
            print(f"Batch {batch_idx} skipped: {e}")
            return None
    
    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, 'test', self.test_evaluator)

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, 'val', self.val_evaluator)

    def _shared_eval(self, batch, prefix, evaluator):
        loss = self._step(batch, evaluator)

        if isinstance(loss, dict):
            loss['loss_total'] = sum(loss.values())
            categorized_losses = self.categorize_losses(loss)
            loss.update(categorized_losses)
            self.log_dict({f'{prefix}/{k}': v for k, v in loss.items()}, sync_dist=True)
        else:
            self.log(f'{prefix}/loss', loss, sync_dist=True)



    def on_train_epoch_end(self):
        self._log_metrics(self.train_evaluator, 'train')

    def on_validation_epoch_end(self):
        self._log_metrics(self.val_evaluator, 'val')

    def on_test_epoch_end(self):
        self._log_metrics(self.test_evaluator, 'test')


    
    def categorize_losses(self, loss_dict):
        loss_seg = 0
        loss_inst = 0
        
        seg_keys = ['loss_ce_inst', 'loss_dice_inst', 'loss_mask_inst']
        inst_keys = ['loss_sem_scal', 'loss_geo_scal', 'loss_frustum', 'loss_ce_ssc']
        
        for key, value in loss_dict.items():
            if any(seg_key in key for seg_key in seg_keys):
                loss_seg += value
            
            elif any(inst_key in key for inst_key in inst_keys):
                loss_inst += value
        
        categorized_losses = {
            'loss_seg': loss_seg,
            'loss_inst': loss_inst
        }
        
        return categorized_losses
        
    def _log_metrics(self, evaluator, prefix=None):
        
       
            metrics = evaluator.compute()
            iou_per_class = metrics.pop('ssc_iou_per_class')
            if self.model_type in ('IPFormer','IPFormerDualHead') and 'pq_per_class' in metrics:
                pq_per_class = metrics.pop('pq_per_class')
                sq_per_class = metrics.pop('sq_per_class')
                rq_per_class = metrics.pop('rq_per_class')
            if prefix:
                metrics = {'/'.join((prefix, k)): v for k, v in metrics.items()}
            self.log_dict(metrics, sync_dist=True)

            if hasattr(self, 'class_names'):
                if self.model_type == 'Symphonies':
                    self.log_dict(
                        {
                            f'{prefix}/ssc_iou_{c}': s.item()
                            for c, s in zip(self.class_names, iou_per_class)
                        },
                    sync_dist=True)
            else:
                    self.log_dict(
                        {
                            f'{prefix}/pq_{c}': s
                            for c, s in zip(self.rearranged_class_names, pq_per_class)
                        },
                        sync_dist=True)
                    self.log_dict(
                        {
                            f'{prefix}/sq_{c}': s
                            for c, s in zip(self.rearranged_class_names, sq_per_class)
                        },
                        sync_dist=True)
                    self.log_dict(
                        {
                            f'{prefix}/rq_{c}': s
                            for c, s in zip(self.rearranged_class_names, rq_per_class)
                        },
                        sync_dist=True)
                    self.log_dict(
                        {
                            f'{prefix}/ssc_iou_{c}': s
                            for c, s in zip(self.class_names, iou_per_class)
                        },
                        sync_dist=True)
            evaluator.reset()
