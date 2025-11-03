

import os
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "INFO"

import misc
import torch
from mmcv import Config
from mmdet3d_plugin import *  # noqa: F401,F403 (kept as in original)
import pytorch_lightning as pl
from argparse import ArgumentParser
from LightningTools.pl_model import pl_model
from LightningTools.dataset_dm import DataModule
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from datetime import datetime

current = datetime.now()
formatted_date = current.strftime('%d_%m_%H_%M')


def parse_config():
    parser = ArgumentParser()
    parser.add_argument('--config_path', default='./configs/IPFormer_config.py')
    parser.add_argument('--seed', type=int, default=7240)
    parser.add_argument('--log_folder', default='./logs/')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--checkpoints_dir', default='./outputs/')
    # Additional override to control stage-1 dir if desired
    parser.add_argument('--stage1_dir', default='./outputs/1st_stage/')
    parser.add_argument('--stage1_ckpt_name', default='ssc_onlyIPFormer.ckpt')
    args = parser.parse_args()

    cfg = Config.fromfile(args.config_path)
    cfg.update(vars(args))
    return args, cfg


def setup_logger_and_profiler(save_dir, log_folder):
    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.path.join('logs', log_folder),
        name='tensorboard'
    )
    profiler = SimpleProfiler(dirpath=save_dir, filename="profiler.txt")
    return tb_logger, profiler


def setup_trainer(config, save_dir, monitor_metric, monitor_mode):
    num_gpu = torch.cuda.device_count()
    tb_logger, profiler = setup_logger_and_profiler(save_dir, config['log_folder'])

    checkpoint_callback = ModelCheckpoint(
        dirpath=save_dir,
        filename="IPFormer" + formatted_date + "_lr{lr:.6f}_{epoch:02d}_{" + monitor_metric + ":.4f}",
        monitor=monitor_metric,
        mode=monitor_mode,
        save_last=True,
        save_top_k=5,
    )

    trainer = pl.Trainer(
        devices=[i for i in range(num_gpu)],
        strategy=DDPStrategy(accelerator='gpu', find_unused_parameters=False),  # kept identical to original
        max_steps=config.get("training_steps", None),
        max_epochs=config.get("max_epochs", 25),
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval='step')],
        logger=tb_logger,
        profiler=profiler,
        sync_batchnorm=True,
        num_sanity_val_steps=0,
        log_every_n_steps=config['log_every_n_steps'],
        check_val_every_n_epoch=config['check_val_every_n_epoch'],
    )
    return trainer


if __name__ == '__main__':
    print("Stage 1 (SSC-only) Training Started!")
    args, config = parse_config()

    # Output structure: ./outputs/1st_stage/
    stage1_output_dir = config.get('stage1_dir', './outputs/1st_stage/')
    stage1_ckpt_name = config.get('stage1_ckpt_name', 'ssc_onlyIPFormer.ckpt')
    stage1_ckpt_path = os.path.join(stage1_output_dir, stage1_ckpt_name)

    pl.seed_everything(config.seed)

    # Force evaluator quiet during Stage 1, as requested
    try:
        if 'evaluator' in config:
            if isinstance(config['evaluator'], dict):
                config['evaluator']['print_out'] = False
                config['evaluator']['debug'] = False
            else:
                # mmcv Config may expose attributes; attempt attribute set as safeguard
                setattr(config.evaluator, 'print_out', False)
                setattr(config.evaluator, 'debug', False)
    except Exception as _:
        pass  # do not crash if evaluator not present

    # Stage 1 flags
    config.ssc_only = True
    config.model['ssc_only'] = True

    # If the user had freeze_ssc=True for two-stage schemes, we must un-freeze for SSC pretraining
    if config.get('freeze_ssc', False):
        config['freeze_ssc'] = False
        if 'freeze_ssc' in config.model:
            config.model['freeze_ssc'] = False

    # Training length for Stage 1
    config.max_epochs = config.get('first_stage_epochs', 25)

    # Ensure directories
    config['checkpoints_dir'] = stage1_output_dir
    misc.check_path(config['checkpoints_dir'])

    # Resume handling (prefer last.ckpt)
    last_ckpt = os.path.join(config['checkpoints_dir'], 'last.ckpt')
    if os.path.exists(last_ckpt):
        print(f"Resuming Stage 1 model from checkpoint: {last_ckpt}")
        model = pl_model.load_from_checkpoint(last_ckpt, config=config)
    else:
        print("Initializing Stage 1 model from scratch.")
        model = pl_model(config)

    data_dm = DataModule(config)
    trainer = setup_trainer(config, save_dir=config['checkpoints_dir'],
                            monitor_metric='val/ssc_mIoU', monitor_mode='max')

    trainer.fit(model=model, datamodule=data_dm)

    # Save canonical Stage-1 handoff checkpoint for Stage 2
    trainer.save_checkpoint(stage1_ckpt_path)
    print(f"✅ Stage 1 complete. Saved handoff checkpoint to: {stage1_ckpt_path}")


