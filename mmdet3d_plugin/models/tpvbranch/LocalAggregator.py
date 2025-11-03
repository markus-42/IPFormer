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
@BACKBONES.register_module()
class LocalAggregator(BaseModule):
    def __init__(
        self,
        local_encoder_backbone=None,
        local_encoder_neck=None,
    ):
        super().__init__()
        self.local_encoder_backbone = builder.build_backbone(local_encoder_backbone)
        self.local_encoder_neck = builder.build_neck(local_encoder_neck)
    
    def forward(self, x):
        x_list = self.local_encoder_backbone(x)
        output = self.local_encoder_neck(x_list)
        output = output[0]

        return output