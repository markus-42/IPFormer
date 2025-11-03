import torch
from torch import nn
from Panoptic.transformer.position_encoding import PositionEmbeddingSine
from Panoptic.transformer.blocks import SelfAttentionLayer, CrossAttentionLayer, FFNLayer, MLP, InstanceNet, ReferenceNet 
from mmdet.models import HEADS  
from mmcv.runner import BaseModule
import torch.nn.functional as F

@HEADS.register_module()
class InstanceTransformer(BaseModule):
    def __init__(
        self,
        panoptic_query_init,
        cross_attention_type,
        self_attention_type,
        num_classes,
        dustbin_class,
        in_channels=128, 
        hidden_dim=384,
        num_queries=100,
        nheads=8,
        dropout=0.0,
        dim_feedforward=2048,
        dec_layers=3,   
        scene_size=[128, 128, 16],
        
    ):
        super().__init__()
        self.scene_size = scene_size
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.nheads = nheads
        self.cross_attention_type = cross_attention_type
        self.self_attention_type = self_attention_type
        self.panoptic_query_init = panoptic_query_init
        self.num_classes=num_classes
        self.dustbin_class=dustbin_class
    

        self.pe_layer = PositionEmbeddingSine(num_pos_feats=hidden_dim // 3, normalize=True)
        
        if self.panoptic_query_init == "query_random_init":
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        elif self.panoptic_query_init == "query_from_context":
            self.InstanceNet = InstanceNet()
            self.query_embed = nn.Embedding(num_queries, hidden_dim)
        elif self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
            self.InstanceNet = InstanceNet()
            self.query_feat = nn.Embedding(scene_size[0] * scene_size[1] * scene_size[2] , in_channels)
            self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.input_proj = nn.Conv3d(in_channels, hidden_dim, kernel_size=1)
        self.spatial_shapes = None
        self.level_start_index = None

        self.transformer_self_attention_layers = nn.ModuleList()
        self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        if self.cross_attention_type == "deformable" or self.cross_attention_type == "3Ddeformable":
            if self.panoptic_query_init == "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init"  or self.panoptic_query_init == "query_from_visible_and_invisible":
                self.ref_net = ReferenceNet(in_channels, self.num_queries)
            else:
                D,H,W = self.scene_size
                self.ref_pts = nn.Parameter(torch.rand(1, D*H*W, 3)) 

        
        for _ in range(dec_layers):
            
            self.transformer_cross_attention_layers.append(
                CrossAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout)
            )

            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(d_model=hidden_dim, nhead=nheads, dropout=dropout)
            )

            self.transformer_ffn_layers.append(
                FFNLayer(d_model=hidden_dim, dim_feedforward=dim_feedforward, dropout=dropout)
            )

        
        if self.dustbin_class:
            self.class_embed = nn.Linear(hidden_dim, num_classes + 1) 
        else:
            self.class_embed = nn.Linear(hidden_dim, num_classes) 

        self.mask_embed = MLP(hidden_dim, hidden_dim, hidden_dim, 3)




    def forward(self, voxel_feats, instance_proposals=None, instance_proposals_ref=None):

        """
        voxel_feats: Tensor of shape (B, C, D, H, W) = (1, 128, 128, 128, 16)
        """
        B, C, D, H, W = voxel_feats.shape

        self.spatial_shapes = torch.tensor([[D, H, W]], dtype=torch.long, device=voxel_feats.device) 
        self.level_start_index = torch.tensor([0], dtype=torch.long, device=voxel_feats.device)      

        voxel_feats = self.input_proj(voxel_feats) 
        
        pos_encoding = self.pe_layer(voxel_feats) 

        voxel_feats = voxel_feats.flatten(2).permute(0, 2, 1) 
        pos_encoding = pos_encoding.flatten(2).permute(0, 2, 1) 
        
        if self.panoptic_query_init == "query_random_init":
            query_feat = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1) 
            query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        elif self.panoptic_query_init == "query_from_context":
            query_feat =  self.InstanceNet(instance_proposals) 
            query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)

        elif self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
            query_feat = self.query_feat.weight.unsqueeze(0).expand(B, -1, -1)  
            query_feat = self.InstanceNet(query_feat + instance_proposals) 
            query_embed = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        
        

        
        aux_outputs = []
        for i in range(len(self.transformer_self_attention_layers)):
            mask_embed = self.mask_embed(query_feat) 
            outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, voxel_feats) 
            outputs_mask = outputs_mask.contiguous().reshape(B, D, H, W, self.num_queries)
                
            query_feat = self.transformer_cross_attention_layers[i](
            q_embed=query_feat, 
            bb_feat=voxel_feats, 
            pos=pos_encoding, 
            query_pos=query_embed
            )
           
            
            query_feat = self.transformer_self_attention_layers[i](q_embed=query_feat ,query_pos=query_embed)
            
            query_feat = self.transformer_ffn_layers[i](query_feat)
        outputs_class = self.class_embed(query_feat) 
        mask_embed = self.mask_embed(query_feat) 
    
        outputs_mask = torch.einsum("bqc,bpc->bpq", mask_embed, voxel_feats) 
        outputs_mask = outputs_mask.contiguous().reshape(B, D, H, W, self.num_queries)

    
        outputs_mask = self.interpolate_(outputs_mask) 

        
   
        return {"query_logits": outputs_class,
                "voxel_logits": outputs_mask,
                "aux_outputs": aux_outputs
                }
    


    def interpolate_(self, x ):
        scene_size = (256, 256, 32)
        x = x.permute(0, 4, 1, 2, 3)  

        x = F.interpolate(
            x, size=scene_size, mode="trilinear", align_corners=False
        )  
        x=x.permute(0, 2, 3, 4, 1)

        return x 
    
    def compute_ssc_logits(self,query_probs, voxel_probs):
        """
        Compute SSC logits from query logits and voxel logits.

        Args:
            query_logits: Tensor of shape (B, Q, C) - per-query class probabilities
            voxel_logits: Tensor of shape (B, D, H, W, Q) - per-voxel query probabilties (after MLp)

        Returns:
            ssc_logits: Tensor of shape (B, C, D, H, W) - per-voxel class probabilities
        """
        
        
        ssc_logits = torch.matmul(voxel_probs, query_probs) 
        ssc_logits = ssc_logits.permute(0, 4, 1, 2, 3) 

        return ssc_logits


    def compute_attn_mask(self, outputs_mask, nheads):
        """
        Computes attention mask for cross-attention
        
        Args:
            outputs_mask (torch.Tensor): matmul(voxel_logits after MLP)  (B, num_queries, D, H, W).
            nheads (int): Number of attention heads.

        Returns:
            torch.Tensor: Attention mask of shape (B * nheads, num_queries, D*H*W).
        """
        
        B, D, H, W, num_queries = outputs_mask.shape 

        keep_mask = (outputs_mask.sigmoid() > 0.5).detach().float() 

        attn_mask = ~keep_mask.bool().permute(0, 4, 1, 2, 3) 

        attn_mask = attn_mask.unsqueeze(1).expand(-1, nheads, -1, -1, -1, -1) 

        attn_mask = attn_mask.reshape(B * nheads, num_queries, D*H*W) 

        return attn_mask
