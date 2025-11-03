import math
import torch
from torch import nn

class PositionEmbeddingSine(nn.Module):
    """
    Standard 3D positional encoding for dense voxel grids.
    """
    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, x):
        """
        x: Tensor of shape (B, C, D, H, W)
        """
        B, C, D, H, W = x.shape
        device = x.device
        
        z_embed = torch.linspace(0, D - 1, D, device=device).view(1, D, 1, 1).expand(B, D, H, W)
        y_embed = torch.linspace(0, H - 1, H, device=device).view(1, 1, H, 1).expand(B, D, H, W)
        x_embed = torch.linspace(0, W - 1, W, device=device).view(1, 1, 1, W).expand(B, D, H, W)

        if self.normalize:
            eps = 1e-6
            x_embed = x_embed / (W - 1 + eps) * self.scale
            y_embed = y_embed / (H - 1 + eps) * self.scale
            z_embed = z_embed / (D - 1 + eps) * self.scale
        
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)
        
        pos_x = x_embed[..., None] / dim_t
        pos_y = y_embed[..., None] / dim_t
        pos_z = z_embed[..., None] / dim_t
        
        pos_x = torch.stack((pos_x[..., 0::2].sin(), pos_x[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[..., 0::2].sin(), pos_y[..., 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[..., 0::2].sin(), pos_z[..., 1::2].cos()), dim=-1).flatten(-2)
        
        pos = torch.cat((pos_x, pos_y, pos_z), dim=-1).permute(0, 4, 1, 2, 3) 
        return pos
