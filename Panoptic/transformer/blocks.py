import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.autograd import Function
from dfa3D import ext_loader
ext_module = ext_loader.load_ext(
    '_ext',
    ['wms_deform_attn_forward', 'wms_deform_attn_backward']
)
from functools import reduce
import torch
import torch.nn as nn
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.ops import MultiScaleDeformableAttention
from typing import Optional
from mmdet3d_plugin.models.img2bev.transformer_utils.deformable_cross_attention import MSDeformableAttention3D_DFA3D

class SelfAttentionLayer(nn.Module):
    """
    Standard self-attention layer for dense tensors.
    """
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, q_embed, attn_mask=None, padding_mask=None, query_pos=None):
        q = k = q_embed if query_pos is None else q_embed + query_pos
        q_embed2 = self.self_attn(q, k, value=q_embed, attn_mask=attn_mask, key_padding_mask=padding_mask)[0]
        q_embed = q_embed + self.dropout(q_embed2)
        q_embed = self.norm(q_embed)
        return q_embed

class CrossAttentionLayer(nn.Module):
    """
    Standard cross-attention layer for dense tensors.
    """
    def __init__(self, d_model, nhead, dropout=0.0):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu
        self.value_proj = nn.Linear(d_model, d_model)
        self._reset_parameters()
    
    def _reset_parameters(self):
        for name, p in self.named_parameters():
            if "weight" in name and p.dim() > 1 and isinstance(p, nn.Linear):
                nn.init.xavier_uniform_(p) 
    
    def forward(self, q_embed, bb_feat, attn_mask=None, padding_mask=None, pos=None, query_pos=None):
        q_embed = self.norm(q_embed)
        value_feats = self.value_proj(bb_feat)
        value_feats = self.norm(value_feats) 
        q_embed2 = self.multihead_attn(
            query=q_embed if query_pos is None else q_embed + query_pos,
            key=bb_feat if pos is None else bb_feat + pos,
            value=value_feats if pos is None else value_feats + pos,
            attn_mask=attn_mask,
            key_padding_mask=padding_mask,
        )[0]
        q_embed = q_embed + self.dropout(q_embed2)
        return q_embed

class FFNLayer(nn.Module):
    """
    Standard feed-forward network layer.
    """
    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0):
        super().__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.activation = F.relu
        self._reset_parameters()
    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, tgt):
        tgt = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

class MLP(nn.Module):
    """
    Multi-layer perceptron (MLP) for dense computations.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList([nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim])])
        
    
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x
    
class InstanceNet(nn.Module):
    def __init__(self):
        super(InstanceNet, self).__init__()
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 384)
        self.pool = nn.MaxPool1d(kernel_size=2621, stride=2621) 

        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1.bias)

        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
         
        x = torch.relu(self.fc1(x)) 

        x = self.fc2(x)             
        x = x.permute(0, 2, 1)       
        x = self.pool(x)            
        x = x.permute(0, 2, 1)       

        return x
        

class ReferenceNet(nn.Module):
    def __init__(self, num_input=128, num_output=100):
        """
        Args:
        - num_input (int): Feature dimension of the input (default: 128).
        - num_output (int): Number of final output instances (default: 100).
        """
        super(ReferenceNet, self).__init__()

        self.fc1 = nn.Linear(num_input, 256)

        pool_kernel_size = 2621 
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel_size, stride=pool_kernel_size)

        self.fc2 = nn.Linear(256, 3)

        self.sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.fc1.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.zeros_(self.fc1.bias)
        nn.init.xavier_uniform_(self.fc2.weight, gain=1.0)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        """
        x: Tensor of shape (1, 262144, 128)
        Returns: Tensor of shape (1, 100, 3)
        """
        x = torch.relu(self.fc1(x)) 

        x = x.permute(0, 2, 1) 
        x = self.pool(x) 
        x = x.permute(0, 2, 1) 

        x = self.fc2(x) 

        return self.sigmoid(x) 
