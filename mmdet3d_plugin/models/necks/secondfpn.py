import numpy as np
import torch
from mmcv.cnn import build_conv_layer, build_norm_layer, build_upsample_layer
from mmcv.runner import BaseModule, auto_fp16
from torch import nn as nn

from mmdet.models import NECKS

@NECKS.register_module()
class SECONDFPN2(BaseModule):
    def __init__(self,
                 in_channels=[128, 128, 256],
                 out_channels=[256, 256, 256],
                 upsample_strides=[1, 2, 4],
                 norm_cfg=dict(type='BN', eps=1e-3, momentum=0.01),
                 upsample_cfg=dict(type='deconv', bias=False),
                 conv_cfg=dict(type='Conv2d', bias=False),
                 use_conv_for_no_stride=False,
                 init_cfg=None):
        super(SECONDFPN2, self).__init__(init_cfg=init_cfg)
        assert len(out_channels) == len(upsample_strides) == len(in_channels)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample_strides = upsample_strides 
        self.fp16_enabled = False

        deblocks = []
        for i, out_channel in enumerate(out_channels):
            stride = upsample_strides[i]
            if stride > 1 or (stride == 1 and not use_conv_for_no_stride):
                upsample_layer = build_upsample_layer(
                    upsample_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)
            else:
                stride = np.round(1 / stride).astype(np.int64)
                upsample_layer = build_conv_layer(
                    conv_cfg,
                    in_channels=in_channels[i],
                    out_channels=out_channel,
                    kernel_size=stride,
                    stride=stride)

            deblock = nn.Sequential(
                upsample_layer,
                build_norm_layer(norm_cfg, out_channel)[1],
                nn.ReLU(inplace=True)
            )
            deblocks.append(deblock)
        self.deblocks = nn.ModuleList(deblocks)

        if init_cfg is None:
            self.init_cfg = [
                dict(type='Kaiming', layer='ConvTranspose2d'),
                dict(type='Constant', layer='NaiveSyncBatchNorm2d', val=1.0)
            ]

    @auto_fp16()
    def forward(self, x):
        """x: list[Tensor], each (N,C_i,H_i,W_i)"""
        assert len(x) == len(self.in_channels)

        ups = [deblock(x[i]) for i, deblock in enumerate(self.deblocks)]

        ref_idx = None
        for i, s in enumerate(self.upsample_strides):
            if s == 1:
                ref_idx = i
                break
        if ref_idx is None:
            ref_idx = int(np.argmax([u.shape[-2] * u.shape[-1] for u in ups]))

        H_ref, W_ref = ups[ref_idx].shape[-2], ups[ref_idx].shape[-1]

        aligned = []
        for u in ups:
            h, w = u.shape[-2], u.shape[-1]
            dh, dw = H_ref - h, W_ref - w

            if abs(dh) <= 1 and abs(dw) <= 1:
                pad_h = max(dh, 0)
                pad_w = max(dw, 0)
                if pad_h or pad_w:
                    u = torch.nn.functional.pad(u, (0, pad_w, 0, pad_h)) 
                if dh < 0 or dw < 0:
                    u = u[..., :H_ref, :W_ref]
            else:
                u = torch.nn.functional.interpolate(
                    u, size=(H_ref, W_ref), mode='bilinear', align_corners=False
                )
            aligned.append(u)

        out = torch.cat(aligned, dim=1) if len(aligned) > 1 else aligned[0]
        return [out]
