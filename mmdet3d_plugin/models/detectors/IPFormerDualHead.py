import torch
from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
from mmdet3d.models import builder
import sys
import os
import torch.nn.functional as F
import numpy as np
from Panoptic.losses import (
    ce_ssc_loss,
    frustum_proportion_loss,
    geo_scal_loss,
    sem_scal_loss,
    HungarianMatcher,
    inst_label_cls_loss,
    inst_mask_loss
)

import torch
from Panoptic.transformer.transformer_utils import panoptic_segmentation_inference


@DETECTORS.register_module()
class IPFormerDualHead(BaseModule):
    def __init__(
        self,
        img_backbone,
        img_neck,
        depth_net,
        img_view_transformer,
        proposal_layer,
        VoxFormer_head,
        panoptic_query_init,
        num_classes,
        dustbin_class,
        use_aux_losses,
        instance_cross_attention,
        occ_encoder_backbone=None,
        occ_encoder_neck=None,
        instance_transformer=None,
        depth_loss=False,
        scene_size=[256, 256, 32],
        thing_ids=[1, 2, 3, 4, 5, 6, 7, 8],
        loss_weights=None,
        criterions=None,
        aux_losses=None,
        pts_bbox_head=None,
        ssc_only=False,
        dual_head=False,
        freeze_ssc=False,
        ssc_factor=1,
        remap_lut=None,
        class_freq=None,
        **kwargs
    ):
        super().__init__()
        self.class_freq = torch.tensor(class_freq, dtype=torch.float) if class_freq is not None else torch.ones(num_classes)
        
        if remap_lut is not None:
            self.remap_lut = torch.tensor(remap_lut, dtype=torch.uint8)
            self.mapped_freq = torch.ones(num_classes, dtype=torch.float)
            for kitti_id, semkitti_id in enumerate(self.remap_lut):
                self.mapped_freq[semkitti_id] = self.class_freq[kitti_id].float()
            self.class_freq = self.mapped_freq
        self.class_weights = 1 / torch.log(self.class_freq + 1e-6)

        self.num_classes = num_classes
        self.img_backbone = builder.build_backbone(img_backbone)
        self.img_neck = builder.build_neck(img_neck)
        self.thing_ids = thing_ids
        self.depth_net = builder.build_neck(depth_net)
        if img_view_transformer is not None:
            self.img_view_transformer = builder.build_neck(img_view_transformer)
        self.proposal_layer = builder.build_head(proposal_layer)
        self.VoxFormer_head = builder.build_head(VoxFormer_head)

        if occ_encoder_backbone is not None:
            self.occ_encoder_backbone = builder.build_backbone(occ_encoder_backbone)
        if occ_encoder_neck is not None:
            self.occ_encoder_neck = builder.build_neck(occ_encoder_neck)
        
        self.ssc_only = ssc_only
        self.dual_head = dual_head
        self.ssc_factor = ssc_factor
        self.instance_transformer = builder.build_head(instance_transformer)
        if self.ssc_only:
            for param in self.instance_transformer.parameters():
                param.requires_grad = False

            for attr in [
                'self_transformer_insts',
                'positional_encoding_insts',
                'insts_ref_positional_encoding',
                'positional_encoding_insts_ref',
                'self_transformer_insts_ref'
            ]:
                if hasattr(self.VoxFormer_head, attr):
                    module = getattr(self.VoxFormer_head, attr)
                    for param in module.parameters():
                        param.requires_grad = False
        else:
            for param in self.instance_transformer.parameters():
                param.requires_grad = True

            for attr in [
                'self_transformer_insts',
                'positional_encoding_insts',
                'insts_ref_positional_encoding',
                'positional_encoding_insts_ref',
                'self_transformer_insts_ref'
            ]:
                if hasattr(self.VoxFormer_head, attr):
                    module = getattr(self.VoxFormer_head, attr)
                    for param in module.parameters():
                        param.requires_grad = True


        self.panoptic_query_init=panoptic_query_init
        self.instance_cross_attention=instance_cross_attention
        self.dustbin_class=dustbin_class
        self.use_aux_losses=use_aux_losses
        self.aux_losses=aux_losses

        self.depth_loss = depth_loss
        self.scene_size = scene_size

        self.loss_weight_dict = {
            'ce_ssc': 1,
            'sem_scal': 1,
            'geo_scal': 1,
            'frustum': 1,
            'loss_ce_inst': 1,
            'loss_mask_inst': 40,
            'loss_dice_inst': 1,
            'loss_iou_inst': 0,
            'loss_unmatched': 1
        } if loss_weights is None else loss_weights

        self.aux_loss_weights = [0.33, 0.66, 1.0] if use_aux_losses else None

        self.loss_weight_dict_matcher = {
            'loss_ce_inst': 1,
            'loss_mask_inst': 40,
            'loss_dice_inst': 1
            }

        self.matcher = HungarianMatcher(
            cost_class=self.loss_weight_dict_matcher['loss_ce_inst'],
            cost_mask=self.loss_weight_dict_matcher['loss_mask_inst'],
            cost_dice=self.loss_weight_dict_matcher['loss_dice_inst']
        )
        self.criterions = criterions

        self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if not self.ssc_only and not self.dual_head:
            for param in self.pts_bbox_head.parameters():
                param.requires_grad = False

        self.freeze_ssc = freeze_ssc
        if self.freeze_ssc:
            for param in self.parameters():
                param.requires_grad = False

            for param in self.instance_transformer.parameters():
                param.requires_grad = True

            for attr in [
                'self_transformer_insts',
                'positional_encoding_insts',
                'insts_ref_positional_encoding',
                'positional_encoding_insts_ref',
                'self_transformer_insts_ref'
            ]:
                if hasattr(self.VoxFormer_head, attr):
                    module = getattr(self.VoxFormer_head, attr)
                    for param in module.parameters():
                        param.requires_grad = True



    def image_encoder(self, img):
        imgs = img
        B, N, C, imH, imW = imgs.shape   
        imgs = imgs.view(B * N, C, imH, imW)

        x = self.img_backbone(imgs)

        if self.img_neck is not None:
            x = self.img_neck(x)
            if type(x) in [list, tuple]:
                x = x[0]
        
        _, output_dim, ouput_H, output_W = x.shape
        x = x.view(B, N, output_dim, ouput_H, output_W)
        
        return x
    
    def context_gen_and_proposal_init(self, img_inputs, img_metas):
        
        img_feats = self.image_encoder(img_inputs[0])
        
        img_transforms = self.depth_net.get_mlp_input(*img_inputs[1:7])

        context, depth_prob = self.depth_net([img_feats] + img_inputs[1:7] + [img_transforms], img_metas)

        if hasattr(self, 'img_view_transformer'):
            context_voxels = self.img_view_transformer(context, depth_prob, img_inputs[1:7])
        else:
            context_voxels = None

        voxel_mask_from_depth = self.proposal_layer(img_inputs[1:7], img_metas)
        
        if self.instance_cross_attention == "deformable" or self.instance_cross_attention == "3Ddeformable":
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                voxel_proposals, instance_proposals, instance_proposals_ref = self.VoxFormer_head(
                    [context],
                    voxel_mask_from_depth,
                    cam_params=img_inputs[1:7],
                    lss_volume=context_voxels,
                    img_metas=img_metas,
                    mlvl_dpt_dists=[depth_prob.unsqueeze(1)]
                )
            else:
                voxel_proposals = self.VoxFormer_head(
                    [context],
                    voxel_mask_from_depth,
                    cam_params=img_inputs[1:7],
                    lss_volume=context_voxels,
                    img_metas=img_metas,
                    mlvl_dpt_dists=[depth_prob.unsqueeze(1)]
                    )


        else:    
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                voxel_proposals, instance_proposals = self.VoxFormer_head(
                    [context],
                    voxel_mask_from_depth,
                    cam_params=img_inputs[1:7],
                    lss_volume=context_voxels,
                    img_metas=img_metas,
                    mlvl_dpt_dists=[depth_prob.unsqueeze(1)]
                )
            else:
                voxel_proposals = self.VoxFormer_head(
                    [context],
                    voxel_mask_from_depth,
                    cam_params=img_inputs[1:7],
                    lss_volume=context_voxels,
                    img_metas=img_metas,
                    mlvl_dpt_dists=[depth_prob.unsqueeze(1)]
                    )

        
        if self.instance_cross_attention == "deformable" or self.instance_cross_attention == "3Ddeformable":
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init":
                return voxel_proposals, depth_prob, instance_proposals, instance_proposals_ref
            else:
                return voxel_proposals, depth_prob

        else:
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                return voxel_proposals, depth_prob, instance_proposals
            else:
                return voxel_proposals, depth_prob
            

        
    def local_and_global_enc(self, x):
        if hasattr(self, 'occ_encoder_backbone'):
            x = self.occ_encoder_backbone(x)
        
        if hasattr(self, 'occ_encoder_neck'):
            x = self.occ_encoder_neck(x)
        
        return x

    def forward_train(self, data_dict):

        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        if self.instance_cross_attention == "deformable" or self.instance_cross_attention == "3Ddeformable":
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                voxel_proposals, depth_prob, instance_proposals, instance_proposals_ref = self.context_gen_and_proposal_init(img_inputs, img_metas)
            else:
                voxel_proposals, depth_prob = self.context_gen_and_proposal_init(img_inputs, img_metas)

        else:
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                voxel_proposals, depth_prob, instance_proposals = self.context_gen_and_proposal_init(img_inputs, img_metas)
            else:
                voxel_proposals, depth_prob = self.context_gen_and_proposal_init(img_inputs, img_metas)

        voxel_feats = self.local_and_global_enc(voxel_proposals)
        voxel_feats_ssc=voxel_feats
        if len(voxel_feats_ssc) > 1:
            voxel_feats_ssc = [voxel_feats_ssc[0]]
        if type(voxel_feats_ssc) is not list:
            voxel_feats_ssc = [voxel_feats_ssc]
        ssc_output = self.pts_bbox_head(
            voxel_feats=voxel_feats_ssc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=data_dict["semantic_label"]
        )
        if self.instance_cross_attention == "deformable" or self.instance_cross_attention == "3Ddeformable":
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                prediction = self.instance_transformer(voxel_feats, instance_proposals, instance_proposals_ref)
            else:
                prediction = self.instance_transformer(voxel_feats)

        else:
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                prediction = self.instance_transformer(voxel_feats, instance_proposals)
            else:
                prediction = self.instance_transformer(voxel_feats)
        
        instance_masks = prediction["voxel_logits"]
        class_predictions = prediction["query_logits"]
    

        if self.use_aux_losses:
            return {
                'voxel_logits': instance_masks,
                'query_logits': class_predictions,
                'ssc_logits': ssc_output['output_voxels'],
                'aux_outputs': prediction["aux_outputs"],
                'depth': depth_prob
            }    
        else:
            return {
                'voxel_logits': instance_masks,
                'query_logits': class_predictions,
                'ssc_logits': ssc_output['output_voxels'],
                'depth': depth_prob
            }

    
    def forward_test(self, data_dict):
        img_inputs = data_dict['img_inputs']
        img_metas = data_dict['img_metas']
        
        if self.instance_cross_attention == "deformable" or self.instance_cross_attention == "3Ddeformable":
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                voxel_proposals, depth_prob, instance_proposals, instance_proposals_ref = self.context_gen_and_proposal_init(img_inputs, img_metas)
            else:
                voxel_proposals, depth_prob = self.context_gen_and_proposal_init(img_inputs, img_metas)

        else:
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                voxel_proposals, depth_prob, instance_proposals = self.context_gen_and_proposal_init(img_inputs, img_metas)
            else:
                voxel_proposals, depth_prob = self.context_gen_and_proposal_init(img_inputs, img_metas)

        voxel_feats = self.local_and_global_enc(voxel_proposals)
        voxel_feats_ssc=voxel_feats
        if len(voxel_feats_ssc) > 1:
            voxel_feats_ssc = [voxel_feats_ssc[0]]
        if type(voxel_feats_ssc) is not list:
            voxel_feats_ssc = [voxel_feats_ssc]
        ssc_output = self.pts_bbox_head(
            voxel_feats=voxel_feats_ssc,
            img_metas=img_metas,
            img_feats=None,
            gt_occ=data_dict["semantic_label"]
        )
        if self.instance_cross_attention == "deformable" or self.instance_cross_attention == "3Ddeformable":
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                prediction = self.instance_transformer(voxel_feats, instance_proposals, instance_proposals_ref)
            else:
                prediction = self.instance_transformer(voxel_feats)

        else:
            if self.panoptic_query_init ==  "query_from_context" or self.panoptic_query_init == "query_from_context_plus_random_init" or self.panoptic_query_init == "query_from_visible_and_invisible":
                prediction = self.instance_transformer(voxel_feats, instance_proposals)
            else:
                prediction = self.instance_transformer(voxel_feats)
        instance_masks = prediction["voxel_logits"]
        class_predictions = prediction["query_logits"]
    
        if self.use_aux_losses:
            return {
                'voxel_logits': instance_masks,
                'query_logits': class_predictions,
                'ssc_logits': ssc_output['output_voxels'],
                'aux_outputs': prediction["aux_outputs"],
                'depth': depth_prob
            }    
        else:
            return {
                'voxel_logits': instance_masks,
                'query_logits': class_predictions,
                'ssc_logits': ssc_output['output_voxels'],
                'depth': depth_prob
            }


    def forward(self, data_dict):
        if self.training:
            return self.forward_train(data_dict)
        else:
            return self.forward_test(data_dict)
        


    def loss(self, preds, target):
        loss_map = {
            'ce_ssc': ce_ssc_loss,
            'sem_scal': sem_scal_loss,
            'geo_scal': geo_scal_loss,
            'frustum': frustum_proportion_loss,
            'loss_ce_inst': inst_label_cls_loss,
            'loss_mask_and_dice_inst': inst_mask_loss
        }
        

        bs = preds['ssc_logits'].shape[0]
        target['class_weights'] = self.class_weights.type_as(preds['ssc_logits'])


        indices = self.matcher(
            preds, target 
        )
    
        losses = {}
        if self.ssc_only or self.dual_head:
            ssc_losses = self.pts_bbox_head.loss(
                output_voxels=preds['ssc_logits'],
                target_voxels=target['semantic_label'],
            )
            losses['loss_ce_ssc']=ssc_losses['loss_voxel_ce'] * self.ssc_factor
            losses['loss_sem_scal']=ssc_losses['loss_voxel_sem_scal'] * self.ssc_factor
            losses['loss_geo_scal']=ssc_losses['loss_voxel_geo_scal'] * self.ssc_factor

        if self.use_aux_losses and 'aux_outputs' in preds and not self.ssc_only:
            for i, pred in enumerate(preds['aux_outputs']):
                aux_weight = self.aux_loss_weights[i]

                for loss in self.aux_losses:
                    if loss == 'loss_ce_inst':
                        losses[f'loss_ce_inst_{i}'] = aux_weight * loss_map[loss](
                            pred, target, indices, self.dustbin_class) * self.loss_weight_dict[loss]
                        
                    elif loss == 'loss_mask_and_dice_inst':
                        mask_losses = loss_map[loss](pred, target, indices, self.dustbin_class)
                        losses[f'loss_mask_inst_{i}'] = aux_weight * mask_losses['loss_mask'] * self.loss_weight_dict['loss_mask_inst'] 
                        losses[f'loss_dice_inst_{i}'] = aux_weight * mask_losses['loss_dice'] * self.loss_weight_dict['loss_dice_inst'] 
                        losses[f'loss_iou_inst_{i}'] = aux_weight * mask_losses['loss_iou'] * self.loss_weight_dict['loss_iou_inst']
                    else:
                        losses['loss_' + loss] = aux_weight * loss_map[loss](preds, target)  * self.loss_weight_dict[loss]
        
        for loss in self.criterions:
            if not self.ssc_only:    
                if loss == 'loss_ce_inst':
                    losses[loss] = loss_map[loss](
                        preds, target, indices, self.dustbin_class) * self.loss_weight_dict[loss]
                    
                elif loss == 'loss_mask_and_dice_inst':
                    mask_losses = loss_map[loss](preds, target, indices, self.dustbin_class)
                    losses['loss_mask_inst'] = mask_losses['loss_mask'] * self.loss_weight_dict['loss_mask_inst']
                    losses['loss_dice_inst'] = mask_losses['loss_dice'] * self.loss_weight_dict['loss_dice_inst'] 
                
                elif loss == 'depth':
                    losses['loss_' + loss] = self.depth_net.get_depth_loss(target['img_metas']['gt_depths'],preds['depth'])
                else:
                    losses['loss_' + loss] = loss_map[loss](preds, target) * self.loss_weight_dict[loss]
                    
        return losses
    
 
