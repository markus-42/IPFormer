

import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS
from mmdet.models.utils import build_transformer
from mmcv.cnn.bricks.transformer import build_positional_encoding
import math


@HEADS.register_module()
class VoxFormerHead(nn.Module):
    def __init__(
        self,
        *args,
        volume_h,
        volume_w,
        volume_z,
        data_config,
        point_cloud_range,
        embed_dims,
        cross_transformer,
        self_transformer,
        positional_encoding,
        mlp_prior=False,
        panoptic_query_init,
        instance_cross_attention_type,
        **kwargs
    ):
        super().__init__()
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z
        self.embed_dims = embed_dims
        
        self.data_config = data_config
        self.point_cloud_range = point_cloud_range
        self.volume_embed = nn.Embedding((self.volume_h) * (self.volume_w) * (self.volume_z), self.embed_dims)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.cross_transformer = build_transformer(cross_transformer)
        self.self_transformer = build_transformer(self_transformer)

        self.panoptic_query_init = panoptic_query_init
        self.instance_cross_attention_type = instance_cross_attention_type

        if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init":
            self.positional_encoding_insts = build_positional_encoding(positional_encoding)
            self.self_transformer_insts = build_transformer(self_transformer)
        
        if self.instance_cross_attention_type == "deformable" or self.instance_cross_attention_type == "3Ddeformable":
            self.insts_ref_positional_encoding = PositionEmbeddingSineFlattened(normalize=True)
            self.positional_encoding_insts_ref = build_positional_encoding(positional_encoding)
            self.self_transformer_insts_ref = build_transformer(self_transformer)

            


        image_grid = self.create_grid()
        self.register_buffer('image_grid', image_grid)
        vox_coords, ref_3d = self.get_voxel_indices()
        self.register_buffer('vox_coords', vox_coords)
        self.register_buffer('ref_3d', ref_3d)

        if mlp_prior:
            self.mlp_prior = nn.Sequential(
                nn.Linear(self.embed_dims, self.embed_dims//2),
                nn.LayerNorm(self.embed_dims//2),
                nn.LeakyReLU(),
                nn.Linear(self.embed_dims//2, self.embed_dims)
            )
        else:
            self.mlp_prior = None
            self.mask_embed = nn.Embedding(1, self.embed_dims)

    def get_voxel_indices(self):
        xv, yv, zv = torch.meshgrid(
            torch.arange(self.volume_h), torch.arange(self.volume_w),torch.arange(self.volume_z), 
            indexing='ij')
        
        idx = torch.arange(self.volume_h * self.volume_w * self.volume_z)
        vox_coords = torch.cat([xv.reshape(-1, 1), yv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)

        ref_3d = torch.cat(
            [(xv.reshape(-1, 1) + 0.5) / self.volume_h, 
             (yv.reshape(-1, 1) + 0.5) / self.volume_w, 
             (zv.reshape(-1, 1) + 0.5) / self.volume_z], dim=-1
        )

        return vox_coords, ref_3d

    def create_grid(self):
        ogfH, ogfW = self.data_config['input_size']
        xs = torch.linspace(0, ogfW - 1, ogfW, dtype=torch.float).view(1, 1, ogfW).expand(1, ogfH, ogfW)
        ys = torch.linspace(0, ogfH - 1, ogfH, dtype=torch.float).view(1, ogfH, 1).expand(1, ogfH, ogfW)

        grid = torch.stack((xs, ys), 1)
        return nn.Parameter(grid, requires_grad=False)
    
    def forward(self, mlvl_feats, proposal, cam_params, lss_volume=None, img_metas=None,  **kwargs):
        """ Forward funtion.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            img_metas: Meta information
            depth: Pre-estimated depth map, (B, 1, H_d, W_d)
            cam_params: Transformation matrix, (rots, trans, intrins, post_rots, post_trans, bda)
        """
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype, device = mlvl_feats[0].dtype, mlvl_feats[0].device        

        volume_queries = self.volume_embed.weight.to(dtype)
        if lss_volume is not None:
            assert lss_volume.shape[0] == 1
            lss_volume_flatten = lss_volume.flatten(2).squeeze(0).permute(1, 0)
            volume_queries = volume_queries + lss_volume_flatten

        if proposal.sum() < 2:
            proposal = torch.ones_like(proposal)
        bev_pos_cross_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=volume_queries.device).to(dtype)).to(dtype)
        bev_pos_self_attn = self.positional_encoding(torch.zeros((bs, 512, 512), device=volume_queries.device).to(dtype)).to(dtype)

        vox_coords, ref_3d = self.vox_coords.clone(), self.ref_3d.clone()
        unmasked_idx = torch.nonzero(proposal.reshape(-1) > 0).view(-1)
        masked_idx = torch.nonzero(proposal.reshape(-1) == 0).view(-1)
        seed_feats = self.cross_transformer.get_vox_features(
            mlvl_feats,
            volume_queries,
            self.volume_h,
            self.volume_w,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=None,
            bev_pos=bev_pos_cross_attn,
            img_metas=img_metas,
            prev_bev=None,
            cam_params=cam_params,
            **kwargs
        )

        vox_feats = torch.empty((self.volume_h, self.volume_w, self.volume_z, self.embed_dims), device=volume_queries.device)
        vox_feats_flatten = vox_feats.reshape(-1, self.embed_dims)
        vox_feats_flatten[vox_coords[unmasked_idx, 3], :] = seed_feats[0]

        if self.mlp_prior is None:
            vox_feats_flatten[vox_coords[masked_idx, 3], :] = self.mask_embed.weight.view(1, self.embed_dims).expand(masked_idx.shape[0], self.embed_dims).to(dtype)
        else:
            vox_feats_flatten[vox_coords[masked_idx, 3], :] = self.mlp_prior(lss_volume_flatten[masked_idx, :])
        
        vox_feats_diff = self.self_transformer.diffuse_vox_features(
            mlvl_feats,
            vox_feats_flatten,
            512,
            512,
            ref_3d=ref_3d,
            vox_coords=vox_coords,
            unmasked_idx=unmasked_idx,
            grid_length=None,
            bev_pos=bev_pos_self_attn,
            img_metas=img_metas,
            prev_bev=None,
            cam_params=cam_params,
            **kwargs
        )

        vox_feats_diff_b = vox_feats_diff.reshape(self.volume_h, self.volume_w, self.volume_z, self.embed_dims)
        vox_feats_diff = vox_feats_diff_b.permute(3, 0, 1, 2).unsqueeze(0) 

        if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init":
            densfied_3d_dca_output = torch.zeros((self.volume_h, self.volume_w, self.volume_z, self.embed_dims), device=volume_queries.device)
            densfied_3d_dca_output_flatten = densfied_3d_dca_output.reshape(-1, self.embed_dims)
            densfied_3d_dca_output_flatten[vox_coords[unmasked_idx, 3], :] = seed_feats[0]

            densfied_3d_dca_output_flatten_pe = self.positional_encoding_insts(torch.zeros((bs, 512, 512), device=volume_queries.device).to(dtype)).to(dtype)
            densfied_3d_dca_output_flatten = self.self_transformer_insts.diffuse_vox_features(
                mlvl_feats,
                densfied_3d_dca_output_flatten,
                512,
                512,
                ref_3d=ref_3d,
                vox_coords=vox_coords,
                unmasked_idx=unmasked_idx,
                grid_length=None,
                bev_pos=densfied_3d_dca_output_flatten_pe,
                img_metas=img_metas,
                prev_bev=None,
                cam_params=cam_params,
                **kwargs
            )
        elif self.panoptic_query_init == "query_from_visible_and_invisible":
            densfied_3d_dca_output_flatten = vox_feats_diff_b.reshape(-1, self.embed_dims).unsqueeze(0)



        if self.instance_cross_attention_type == "deformable" or self.instance_cross_attention_type == "3Ddeformable":
            densfied_3d_dca_output_ref = torch.zeros((self.volume_h, self.volume_w, self.volume_z, self.embed_dims), device=volume_queries.device)
            densfied_3d_dca_output_flatten_ref = densfied_3d_dca_output_ref.reshape(-1, self.embed_dims)
            densfied_3d_dca_output_flatten_ref[vox_coords[unmasked_idx, 3], :] = seed_feats[0]

            densfied_3d_dca_output_flatten_ref_pe = self.insts_ref_positional_encoding(densfied_3d_dca_output_flatten_ref)
            densfied_3d_dca_output_flatten_ref_pe = densfied_3d_dca_output_flatten_ref + densfied_3d_dca_output_flatten_ref_pe
            
            densfied_3d_dca_output_flatten_pe2 = self.positional_encoding_insts_ref(torch.zeros((bs, 512, 512), device=volume_queries.device).to(dtype)).to(dtype)
            densfied_3d_dca_output_flatten_ref_pe = self.self_transformer_insts_ref.diffuse_vox_features(
                mlvl_feats,
                densfied_3d_dca_output_flatten_ref_pe,
                512,
                512,
                ref_3d=ref_3d,
                vox_coords=vox_coords,
                unmasked_idx=unmasked_idx,
                grid_length=None,
                bev_pos=densfied_3d_dca_output_flatten_pe2,
                img_metas=img_metas,
                prev_bev=None,
                cam_params=cam_params,
                **kwargs
            )


        if self.instance_cross_attention_type == "deformable"  or self.instance_cross_attention_type == "3Ddeformable":

            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init":
                return vox_feats_diff, densfied_3d_dca_output_flatten, densfied_3d_dca_output_flatten_ref_pe
            else:
                return vox_feats_diff

        else:    
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                return vox_feats_diff, densfied_3d_dca_output_flatten
            else:
                return vox_feats_diff
        



class PositionEmbeddingSineFlattened(nn.Module):
    """
    3D Positional Encoding for flattened input (D*W*H, hidden_dim).
    Automatically adjusts num_pos_feats and adds padding if needed.
    """
    def __init__(self, grid_shape=(128, 128, 16), hidden_dim=128, temperature=10000, normalize=False, scale=None):
        """
        grid_shape: Tuple (D, H, W) defining the voxel grid dimensions.
        hidden_dim: Size of input features (must be at least divisible by 3).
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        self.normalize = normalize
        self.grid_shape = grid_shape 
        
        self.num_pos_feats = hidden_dim // 3 
        self.extra_dims = hidden_dim - (self.num_pos_feats * 3) 

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

        self.register_buffer("positional_encoding", self.create_positional_encoding())

    def create_positional_encoding(self):
        """
        Creates the positional encodings for the entire 3D grid.
        Returns a tensor of shape (D*H*W, hidden_dim).
        """
        D, H, W = self.grid_shape
        N = D * H * W 
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        z_embed = torch.arange(D, dtype=torch.float, device=device).repeat_interleave(H * W)
        y_embed = torch.arange(H, dtype=torch.float, device=device).repeat(W).repeat(D)
        x_embed = torch.arange(W, dtype=torch.float, device=device).repeat(H * D)

        if self.normalize:
            eps = 1e-6
            z_embed = z_embed / (D - 1 + eps) * self.scale
            y_embed = y_embed / (H - 1 + eps) * self.scale
            x_embed = x_embed / (W - 1 + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float, device=device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t 
        pos_y = y_embed[:, None] / dim_t
        pos_z = z_embed[:, None] / dim_t

        pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=-1).flatten(-2)
        pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=-1).flatten(-2)
        pos_z = torch.stack((pos_z[:, 0::2].sin(), pos_z[:, 1::2].cos()), dim=-1).flatten(-2)

        pos_encoding = torch.cat((pos_x, pos_y, pos_z), dim=-1) 

        if self.extra_dims > 0:
            pad_tensor = torch.zeros(N, self.extra_dims, device=device)
            pos_encoding = torch.cat([pos_encoding, pad_tensor], dim=-1)

        return pos_encoding 

    def forward(self, x):
        """
        x: Tensor of shape (N, hidden_dim) where N = D*H*W.
        Returns a positional encoding tensor of shape (N, hidden_dim).
        """
        N, hidden_dim = x.shape
        assert hidden_dim == self.hidden_dim, f"Input hidden_dim ({hidden_dim}) must match initialized hidden_dim ({self.hidden_dim})"

        return self.positional_encoding 


