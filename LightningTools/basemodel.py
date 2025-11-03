import torch
import pytorch_lightning as pl
import torch.optim as optim
from omegaconf import open_dict
from torch.cuda.amp import autocast
from .torch_util import WarmupCosine


class LightningBaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
    
    def configure_optimizers(self):
        if self.config['optimizer']['type'] == 'AdamW':

            params_to_optimize = [param for param in self.model.parameters() if param.requires_grad]

            optimizer = torch.optim.AdamW(
                params_to_optimize,
                lr=self.config['optimizer']['lr'],
                weight_decay=self.config['optimizer']['weight_decay']
            )

        else:
            raise NotImplementedError
        
        if self.config['lr_scheduler']['type'] == 'OneCycleLR':
            lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.config['lr_scheduler']['max_lr'],
                total_steps=self.config['lr_scheduler']['total_steps'],
                pct_start=self.config['lr_scheduler']['pct_start'],
                cycle_momentum=self.config['lr_scheduler']['cycle_momentum'],
                anneal_strategy=self.config['lr_scheduler']['anneal_strategy'])

            interval=self.config['lr_scheduler']['interval']
            frequency=self.config['lr_scheduler']['frequency']
            scheduler = {
            'scheduler': lr_scheduler,
            'interval': interval,
            'frequency': frequency
            }

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
    
        elif self.config['lr_scheduler']['type'] == 'ReduceLROnPlateau':
        
            lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=self.config['lr_scheduler']['mode'],
                factor=self.config['lr_scheduler']['factor'],
                patience=self.config['lr_scheduler']['patience'],
                threshold=self.config['lr_scheduler']['threshold'],
                threshold_mode=self.config['lr_scheduler']['threshold_mode'],
                cooldown=self.config['lr_scheduler']['cooldown'],
                min_lr=self.config['lr_scheduler']['min_lr'],
                eps=self.config['lr_scheduler']['eps']
            )

            interval='epoch'
            frequency=1
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': interval,
                'frequency': frequency
            }

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler,
                "monitor": "loss"
            }
        elif self.config['lr_scheduler']['type'] == 'CosineAnnealingLR':
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config['lr_scheduler']['T_max'],
                eta_min=self.config['lr_scheduler']['eta_min']
            )
            interval='epoch'
            frequency=1
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': interval,
                'frequency': frequency
                }

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        elif self.config['lr_scheduler']['type'] == 'MultiStepLR':    
            lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=self.config['lr_scheduler']['milestones'],
                gamma=self.config['lr_scheduler']['gamma']
            )
            interval='epoch'
            frequency=1
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': interval,
                'frequency': frequency
                }

            return {
                'optimizer': optimizer,
                'lr_scheduler': scheduler
            }
        elif self.config['lr_scheduler']['type'] == 'LambdaLR':
            lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                WarmupCosine(
                max_iter=self.config['lr_scheduler']['max_iter'],  
                warmup_end=self.config['lr_scheduler']['warmup_end'],  
                factor_min=self.config['lr_scheduler']['factor_min']
                ),
            ),
            
            interval='epoch'
            frequency=1
            scheduler = {
                'scheduler': lr_scheduler,
                'interval': interval,
                'frequency': frequency
                }
        else:
            raise NotImplementedError
        
