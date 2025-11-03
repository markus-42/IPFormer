import os
import torch
import misc
import pytorch_lightning as pl
from mmcv import Config
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from datetime import datetime
from LightningTools.eval.psc_metric import PSCMetrics
import torch
import numpy as np
from argparse import ArgumentParser
from mmdet3d_plugin import *
import sys
from LightningTools.pl_model import pl_model 
from LightningTools.dataset_dm import DataModule
from Panoptic.transformer.transformer_utils import panoptic_segmentation_inference
from mmcv import Config
from mmdet.datasets import build_dataset

def parse_config():
    parser = ArgumentParser()

    parser.add_argument('--config_path', default='./configs/IPFormer_DualHead_generalization_kitti360.py')
    parser.add_argument('--ckpt_path', default='./ckpts/C2_ep28_pq_dagger=0.1445.ckpt')
    parser.add_argument('--output_dir', default='./outputs/inference')
    parser.add_argument('--inference_mode', choices=['panoptic', 'semantic'], default='panoptic', help='Type of inference')
    parser.add_argument('--measure_time', action='store_true', help='Measure inference time')
    parser.add_argument('--generate_saliency', action='store_true', help='Enable saliency map generation')
    parser.add_argument('--smoothgrad', action='store_true', help='Use SmoothGrad for saliency maps')

    
    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default='semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)
    cfg.update(vars(args))
    return args, cfg

def main():
    print("Evaluation Started!")
    args, config = parse_config()
    log_folder = os.path.join('logs', config['log_folder'])
    misc.check_path(log_folder)

    misc.check_path(os.path.join(log_folder, 'tensorboard'))
    tb_logger = TensorBoardLogger(
        save_dir=log_folder,
        name='tensorboard'
    )

    config.dump(os.path.join(log_folder, 'config.py'))
    profiler = SimpleProfiler(dirpath=log_folder, filename="profiler.txt")

    pl.seed_everything(config.seed)
    num_gpu = torch.cuda.device_count()
       
    model = pl_model.load_from_checkpoint(config['ckpt_path'], config=config)
    data_dm = DataModule(config)


    trainer = pl.Trainer(
        devices=[i for i in range(num_gpu)],
        strategy=DDPStrategy(accelerator='gpu', find_unused_parameters=False),
        logger=tb_logger,
        profiler=profiler,
        max_epochs=config['max_epochs'],
    )
    
    results = trainer.test(model=model, datamodule=data_dm, ckpt_path=config['ckpt_path'])
    
    


if __name__ == '__main__':
    main()
