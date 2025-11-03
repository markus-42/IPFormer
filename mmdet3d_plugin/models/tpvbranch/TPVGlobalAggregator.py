from mmdet3d.models.builder import BACKBONES
import torch
from mmcv.runner import BaseModule
from mmdet3d.models import builder
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
def get_shape(x):
    """Returns the shape and data type of various data types safely, formatted for better readability."""
    
    if isinstance(x, torch.Tensor):
        return f"(Type: torch.Tensor, Shape: {tuple(x.shape)}, Dtype: {x.dtype})"
    
    elif isinstance(x, np.ndarray):
        return f"(Type: np.ndarray, Shape: {x.shape}, Dtype: {x.dtype})"
    
    elif isinstance(x, list):
        if len(x) == 0:
            return "(Type: list, Length: 0)"
        
        if len(x) < 10: 
            elements_info = "\n".join([f"   - [{i}] {get_shape(elem)}" for i, elem in enumerate(x)])
            return f"(Type: list, Length: {len(x)}):\n{elements_info}"
        else:
            first_elem = x[0]
            return f"(Type: list[{type(first_elem).__name__}], Length: {len(x)})"
    
    elif isinstance(x, tuple):
        if len(x) == 0:
            return "(Type: tuple, Length: 0)"
        elif len(x) < 10: 
            return f"(Type: tuple, Length: {len(x)}):\n" + "\n".join(
                [f"   - [{i}] {get_shape(x[i])}" for i in range(len(x))]
            )
    
    elif isinstance(x, dict):
        formatted_dict = "\n".join([f"   - {key}: {get_shape(value)}" for key, value in x.items()])
        return f"(Type: dict, Keys: {list(x.keys())}):\n{formatted_dict}"
    
    elif isinstance(x, set):
        return f"(Type: set, Size: {len(x)})"
    
    elif isinstance(x, int):
        return f"(Type: int, Value: {x})"
    elif isinstance(x, float):
        return f"(Type: float, Value: {x})"
    elif isinstance(x, bool):
        return f"(Type: bool, Value: {x})"
    elif isinstance(x, str):
        return f"(Type: str, Length: {len(x)})"
    elif x is None:
        return f"(Type: NoneType)"
    
    return f"(Type: {type(x).__name__}, Unknown Shape)"
class TPVPooler(BaseModule):
    def __init__(
        self,
        embed_dims=128,
        split=[8,8,8],
        grid_size=[128, 128, 16],
    ):
        super().__init__()
        self.pool_xy = nn.MaxPool3d(
            kernel_size=[1, 1, grid_size[2]//split[2]],
            stride=[1, 1, grid_size[2]//split[2]], padding=0
        )

        self.pool_yz = nn.MaxPool3d(
            kernel_size=[grid_size[0]//split[0], 1, 1],
            stride=[grid_size[0]//split[0], 1, 1], padding=0
        )

        self.pool_zx = nn.MaxPool3d(
            kernel_size=[1, grid_size[1]//split[1], 1],
            stride=[1, grid_size[1]//split[1], 1], padding=0
        )

        in_channels = [int(embed_dims * s) for s in split]
        out_channels = [int(embed_dims) for s in split]

        self.mlp_xy = nn.Sequential(
            nn.Conv2d(in_channels[2], out_channels[2], kernel_size=1, stride=1), 
            nn.ReLU(), 
            nn.Conv2d(out_channels[2], out_channels[2], kernel_size=1, stride=1))
        
        self.mlp_yz = nn.Sequential(
            nn.Conv2d(in_channels[0], out_channels[0], kernel_size=1, stride=1), 
            nn.ReLU(), 
            nn.Conv2d(out_channels[0], out_channels[0], kernel_size=1, stride=1))
        
        self.mlp_zx = nn.Sequential(
            nn.Conv2d(in_channels[1], out_channels[1], kernel_size=1, stride=1), 
            nn.ReLU(), 
            nn.Conv2d(out_channels[1], out_channels[1], kernel_size=1, stride=1))
    
    def forward(self, x):
        tpv_xy = self.mlp_xy(self.pool_xy(x).permute(0, 4, 1, 2, 3).flatten(start_dim=1, end_dim=2))
        tpv_yz = self.mlp_yz(self.pool_yz(x).permute(0, 2, 1, 3, 4).flatten(start_dim=1, end_dim=2))
        tpv_zx = self.mlp_zx(self.pool_zx(x).permute(0, 3, 1, 2, 4).flatten(start_dim=1, end_dim=2))

        tpv_list = [tpv_xy, tpv_yz, tpv_zx]

        return tpv_list

@BACKBONES.register_module()
class TPVGlobalAggregator(BaseModule):
    def __init__(
        self,
        embed_dims=128,
        split=[8,8,8],
        grid_size=[128, 128, 16],
        global_encoder_backbone=None,
        global_encoder_neck=None,
    ):
        super().__init__()

        self.tpv_pooler = TPVPooler(
            embed_dims=embed_dims, split=split, grid_size=grid_size
        )

        self.global_encoder_backbone = builder.build_backbone(global_encoder_backbone)
        self.global_encoder_neck = builder.build_neck(global_encoder_neck)
    
    def forward(self, x):
        """
        xy: [b, c, h, w, z] -> [b, c, h, w]
        yz: [b, c, h, w, z] -> [b, c, w, z]
        zx: [b, c, h, w, z] -> [b, c, h, z]
        """
        x_3view = self.tpv_pooler(x)

        x_3view = self.global_encoder_backbone(x_3view)

        tpv_list = []
        for x_tpv in x_3view:
            x_tpv = self.global_encoder_neck(x_tpv)
            if not isinstance(x_tpv, torch.Tensor):
                x_tpv = x_tpv[0]
            tpv_list.append(x_tpv)
        tpv_list[0] = F.interpolate(tpv_list[0], size=(128, 128), mode='bilinear').unsqueeze(-1)
        tpv_list[1] = F.interpolate(tpv_list[1], size=(128, 16), mode='bilinear').unsqueeze(2)
        tpv_list[2] = F.interpolate(tpv_list[2], size=(128, 16), mode='bilinear').unsqueeze(3)


        return tpv_list
    