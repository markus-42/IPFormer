import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

import os.path as osp
import pickle
import time
import torch
import gzip
torch.cuda.empty_cache() 
import pytorch_lightning as pl
import numpy as np
from argparse import ArgumentParser
from mmdet3d_plugin import *
import sys
from LightningTools.pl_model import pl_model 
from LightningTools.dataset_dm import DataModule
from Panoptic.transformer.transformer_utils import panoptic_segmentation_inference
from mmcv import Config
from mmdet.datasets import build_dataset


def parse_inference_config():
    parser = ArgumentParser()

    parser.add_argument('--config_path', default='./configs/IPFormer_DualHead_config.py')
    parser.add_argument('--ckpt_path', default='/home/jovyan/danit-semantickitti/C2_ep28_pq_dagger=0.1445.ckpt')
    parser.add_argument('--inference_output_dir', default='/home/jovyan/danit-semantickitti/panop-post/ipformer_dualHead/')
    parser.add_argument('--inference_mode', choices=['panoptic', 'semantic'], default='panoptic', help='Type of inference')
    parser.add_argument('--measure_time', action='store_true', help='Measure inference time')


    parser.add_argument('--seed', type=int, default=7240, help='random seed point')
    parser.add_argument('--log_folder', default='semantic_kitti')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--test_mapping', action='store_true')
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--log_every_n_steps', type=int, default=1000)
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--pretrain', action='store_true')
    parser.add_argument('--checkpoints_dir', default='./outputs/PQVFormer/checkpoints/')
    
    args = parser.parse_args()
    cfg = Config.fromfile(args.config_path)
    cfg.update(vars(args))
    return args, cfg


def main():
    args, cfg = parse_inference_config()
    os.makedirs(args.inference_output_dir, exist_ok=True)


    data_module = DataModule(cfg)
    data_module.setup() 
    data_loader = data_module.val_dataloader()

    pl.seed_everything(cfg.seed)
    num_gpu = torch.cuda.device_count()
       
    model = pl_model.load_from_checkpoint(cfg['ckpt_path'], config=cfg)
    model.cuda()
    model.eval()

    count = 0
    evaluation_time = 0.0
    device = next(model.parameters()).device

    with torch.no_grad():
        for batch_inputs in data_loader:

            for key in batch_inputs:
                if isinstance(batch_inputs[key], torch.Tensor):
                    batch_inputs = move_to_device(batch_inputs, device)


            start_time = time.time()
            outputs = model(batch_inputs)
            end_time = time.time()
            inference_time = end_time - start_time
            evaluation_time += inference_time
            print(f'Inference time: {inference_time:.4f} s')

            voxel_output = outputs['voxel_logits'] 
            query_output = outputs['query_logits'] 
            

            preds = panoptic_segmentation_inference(voxel_output, query_output, thing_ids=cfg.thing_ids)
            panoptic_seg = preds['panoptic_seg']
            semantic_pred = preds['semantic_perd']

            panoptic_seg_np = panoptic_seg.detach().cpu().numpy()
            instance_seg = panoptic_seg_np.copy()

            thing_ids = {segment['id'] for segment in preds['segments_info'][0] if segment['isthing']}
            mask = np.isin(instance_seg, list(thing_ids), invert=True)
            instance_seg[mask] = 0
            

            
            output_dict = {
                'panoptic_seg': panoptic_seg_np.astype(np.uint8),
                'semantic_perd': semantic_pred.detach().cpu().numpy().astype(np.uint8),
                'instance_seg': instance_seg.astype(np.uint8),
                'segments_info' : preds['segments_info'],
                'semantic_label': batch_inputs['semantic_label'].detach().cpu().numpy().astype(np.uint8), 
                'instance_label': batch_inputs['instance_label'].detach().cpu().numpy().astype(np.uint8) 
                
                }

            file_path = osp.join(args.inference_output_dir, f"{batch_inputs['img_metas']['frame_id'][0]}.pkl")
            with open(file_path, 'wb') as f:
                pickle.dump(output_dict, f)
                print(f'Saved to {file_path}')
            count += 1


 
    assert count > 0, "No inference was performed."
    if args.measure_time:
        print(f"Processed {count} frames in {evaluation_time:.4f} s")
        print(f"Average inference time/frame: {(evaluation_time / count):.4f} s")


        
def move_to_device(data, device):
    """ Recursively moves all tensors in a dictionary, list, or tuple to a given device. """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    return data 

def convert_to_numpy(data):
    """
    Recursively moves all torch tensors in a dictionary (including nested dictionaries)
    to CPU and converts them to numpy arrays.
    """
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    elif isinstance(data, dict):
        return {key: convert_to_numpy(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_to_numpy(item) for item in data]
    elif isinstance(data, tuple):
        return tuple(convert_to_numpy(item) for item in data)
    else:
        return data 


if __name__ == '__main__':
    main()
