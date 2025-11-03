from functools import reduce

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Dict


def generate_grid(grid_shape, value=None, offset=0, normalize=False):
    """
    Args:
        grid_shape: The (scaled) shape of grid.
        value: The (unscaled) value the grid represents.
    Returns:
        Grid coordinates of shape [len(grid_shape), *grid_shape]
    """
    if value is None:
        value = grid_shape
    grid = []
    for i, (s, val) in enumerate(zip(grid_shape, value)):
        g = torch.linspace(offset, val - 1 + offset, s, dtype=torch.float)
        if normalize:
            g /= s - 1
        shape_ = [1 for _ in grid_shape]
        shape_[i] = s
        g = g.reshape(1, *shape_).expand(1, *grid_shape)
        grid.append(g)
    return torch.cat(grid, dim=0)


def cumprod(xs):
    return reduce(lambda x, y: x * y, xs)


def flatten_fov_from_voxels(x3d, fov_mask):
    assert x3d.shape[0] == 1
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
    return x3d.flatten(2)[..., fov_mask].transpose(1, 2)


def index_fov_back_to_voxels(x3d, fov, fov_mask):
    assert x3d.shape[0] == fov.shape[0] == 1
    if fov_mask.dim() == 2:
        assert fov_mask.shape[0] == 1
        fov_mask = fov_mask.squeeze()
    fov_concat = torch.zeros_like(x3d).flatten(2)
    fov_concat[..., fov_mask] = fov.transpose(1, 2)
    return torch.where(fov_mask, fov_concat, x3d.flatten(2)).reshape(*x3d.shape)


def interpolate_flatten(x, src_shape, dst_shape, mode='nearest'):
    """Inputs & returns shape as [bs, n, (c)]
    """
    if len(x.shape) == 3:
        bs, n, c = x.shape
        x = x.transpose(1, 2)
    elif len(x.shape) == 2:
        bs, n, c = *x.shape, 1
    assert cumprod(src_shape) == n
    x = F.interpolate(
        x.reshape(bs, c, *src_shape).float(), dst_shape, mode=mode,
        align_corners=False).flatten(2).transpose(1, 2).to(x.dtype)
    if c == 1:
        x = x.squeeze(2)
    return x


def flatten_multi_scale_feats(feats):
    feat_flatten = torch.cat([nchw_to_nlc(feat) for feat in feats], dim=1)
    shapes = torch.stack([torch.tensor(feat.shape[2:]) for feat in feats]).to(feat_flatten.device)
    return feat_flatten, shapes


def get_level_start_index(shapes):
    return torch.cat((shapes.new_zeros((1, )), shapes.prod(1).cumsum(0)[:-1]))


def nlc_to_nchw(x, shape):
    """Convert [N, L, C] shape tensor to [N, C, H, W] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, L, C] before conversion.
        shape (Sequence[int]): The height and width of output feature map.
    Returns:
        Tensor: The output tensor of shape [N, C, H, W] after conversion.
    """
    B, L, C = x.shape
    assert L == cumprod(shape), 'The seq_len does not match H, W'
    return x.transpose(1, 2).reshape(B, C, *shape).contiguous()


def nchw_to_nlc(x):
    """Flatten [N, C, H, W] shape tensor to [N, L, C] shape tensor.
    Args:
        x (Tensor): The input tensor of shape [N, C, H, W] before conversion.
    Returns:
        Tensor: The output tensor of shape [N, L, C] after conversion.
        tuple: The [H, W] shape.
    """
    return x.flatten(2).transpose(1, 2).contiguous()


def pix2cam(p_pix, depth, K):
    p_pix = torch.cat([p_pix * depth, depth], dim=1) 
    return K.inverse() @ p_pix.flatten(2)


def cam2vox(p_cam, E, vox_origin, vox_size, offset=0.5):
    p_wld = E.inverse() @ F.pad(p_cam, (0, 0, 0, 1), value=1)
    p_vox = (p_wld[:, :-1].transpose(1, 2) - vox_origin.unsqueeze(1)) / vox_size - offset
    return p_vox


def pix2vox(p_pix, depth, K, E, vox_origin, vox_size, offset=0.5, downsample_z=1):
    """
    Pixel to Camera Space (pix2cam): The pix2cam function takes pixel coordinates (p_pix), multiplies them by their corresponding depth values (depth), and then transforms these points into camera space using the camera intrinsic matrix (K). The depth is concatenated to the pixel coordinates to form homogeneous coordinates before applying the transformation.

    Camera to Voxel Space (cam2vox): The cam2vox function takes the camera space coordinates (p_cam) and transforms them into world space using the extrinsic matrix (E). Then, it converts these world space coordinates into voxel space coordinates by adjusting them based on the voxel grid's origin (vox_origin) and size (vox_size). An offset is applied to center the points within the voxels.

    Downsampling (Optional): If the downsample_z parameter is not equal to 1, the function downsamples the z-coordinate of the voxel points. This is useful for adjusting the resolution of the voxel grid in the z-dimension.

    Final Output (vol_pts): The final output, vol_pts, is the voxel space coordinates obtained from the original pixel coordinates. These coordinates are converted to long integers, which is a common practice when dealing with indices or discrete grid locations in programming.
    """
    p_cam = pix2cam(p_pix, depth, K)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size, offset)
    if downsample_z != 1:
        p_vox[..., -1] /= downsample_z
    return p_vox


def cam2pix(p_cam, K, image_shape):
    """
    Return:
        p_pix: (bs, H*W, 2)
    """
    p_pix = K @ p_cam / p_cam[:, 2] 
    p_pix = p_pix[:, :2].transpose(1, 2) / (torch.tensor(image_shape[::-1]).to(p_pix) - 1)
    return p_pix


def vox2pix(p_vox, K, E, vox_origin, vox_size, image_shape, scene_shape):
    p_vox = p_vox.squeeze(2) * torch.tensor(scene_shape).to(p_vox) * vox_size + vox_origin
    p_cam = E @ F.pad(p_vox.transpose(1, 2), (0, 0, 0, 1), value=1)
    return cam2pix(p_cam[:, :-1], K, image_shape).clamp(0, 1)


def volume_rendering(
        volume,
        image_grid,
        K,
        E,
        vox_origin,
        vox_size,
        image_shape,
        depth_args=(2, 50, 1),
):
    depth = torch.arange(*depth_args).to(image_grid) 
    p_pix = F.pad(image_grid, (0, 0, 0, 0, 0, 1), value=1) 
    p_pix = p_pix.unsqueeze(-1) * depth.reshape(1, 1, 1, 1, -1)

    p_cam = K.inverse() @ p_pix.flatten(2)
    p_vox = cam2vox(p_cam, E, vox_origin, vox_size)
    p_vox = p_vox.reshape(1, *image_shape, depth.size(0), -1) 
    p_vox = p_vox / (torch.tensor(volume.shape[-3:]) - 1).to(p_vox)

    return F.grid_sample(volume, torch.flip(p_vox, dims=[-1]) * 2 - 1, padding_mode='zeros'), depth


def render_depth(volume, image_grid, K, E, vox_origin, vox_size, image_shape, depth_args):
    sigmas, z = volume_rendering(volume, image_grid, K, E, vox_origin, vox_size, image_shape,
                                 depth_args)
    beta = z[1] - z[0]
    T = torch.exp(-torch.cumsum(F.pad(sigmas[..., :-1], (1, 0)) * beta, dim=-1))
    alpha = 1 - torch.exp(-sigmas * beta)
    depth_map = torch.sum(T * alpha * z, dim=-1).reshape(1, *image_shape)
    depth_map = depth_map 
    return depth_map


def inverse_warp(img, image_grid, depth, pose, K, padding_mode='zeros'):
    """
    img: (B, 3, H, W)
    image_grid: (B, 2, H, W)
    depth: (B, H, W)
    pose: (B, 3, 4)
    """
    p_cam = pix2cam(image_grid, depth.unsqueeze(1), K)
    p_cam = (pose @ F.pad(p_cam, (0, 0, 0, 1), value=1))[:, :3]
    p_pix = cam2pix(p_cam, K, img.shape[2:])
    p_pix = p_pix.reshape(*depth.shape, 2) * 2 - 1
    projected_img = F.grid_sample(img, p_pix, padding_mode=padding_mode)
    valid_mask = p_pix.abs().max(dim=-1)[0] <= 1
    return projected_img, valid_mask

def mask_wise_merge(
        voxel_logits: Tensor, 
        query_logits: Tensor, 
        ssc_logits: Tensor, 
        thing_classes: List, 
        frustrum_masks: Tensor,
        alpha: float = 1/3,
        beta: float = 1.0,
        query_threshold: float = 0.2,
        mask_overlap_threshold: float = 0.5,
        in_fov_threshold: float = 0.5,
        object_mask_threshold: float = 0.25
    ) -> Dict[str, Tensor]: 
    """
    This function implements the Mask-Wise Merge algo in the PanoSSC https://arxiv.org/pdf/2406.07037 to produce panoptic output.
    Alpha, and Beta is hyperparameter to calculate score for i-th mask. The default values are taken directly from the paper.
    
    Args:
        voxel_logits: Tensor, shape (bs, num_instances, 256, 256, 32), dtype float32
        query_logits: Tensor, shape (bs, num_instances, num_classes), dtype float32
        ssc_logits: Tensor, shape (bs, num_classes, 256, 256, 32), dtype float32
        thing_classes: List, list of thing classes, int
        frustrum_masks: Tensor, shape (bs, 256, 256, 32), dtype bool
        alpha: float, default 1/3
        beta: float, default 1.0
        query_threshold: float, default 0.2
        mask_overlap_threshold: float, default 0.5
        in_fov_threshold: float, default 0.5
        object_mask_threshold: float, default 0.25

    Returns:
        final_ssc_pred: Tensor, shape (bs, 256, 256, 32)
        merged_instance_pred: Tensor, shape (bs, 256, 256, 32)
        semantic_pred: Tensor, shape (bs, 256, 256, 32)
        sorted_indices: Tensor, shape (bs, num_instances)
        mask_scores: Tensor, shape (bs, num_instances)
        query_class_pred: Tensor, shape (bs, num_instances)
        query_probs: Tensor, shape (bs, num_instances)
    """
    bs, _, h, w, d = voxel_logits.shape
    device = voxel_logits.device
    
    semantic_pred = torch.argmax(torch.softmax(ssc_logits, dim=1), dim=1) 
    thing_classes_tensor = torch.tensor(thing_classes).to(device)
    things_filtered_ssc_pred = semantic_pred * (~torch.isin(semantic_pred, thing_classes_tensor)).int()
    
    query_probs, query_class_pred = torch.max(F.softmax(query_logits, dim=-1), dim=-1)
    
    voxel_probs = torch.sigmoid(voxel_logits)
    voxel_masks = (voxel_probs > object_mask_threshold) 
    mask_areas = voxel_masks.int().sum(dim=(2, 3, 4)) 
    mask_scores = (query_probs ** alpha) * ((torch.sum(voxel_probs * voxel_masks, dim=(2, 3, 4)) / mask_areas)**beta)
    sorted_indices = torch.argsort(mask_scores, dim=-1, descending=True)

    merged_instance_pred = torch.zeros((bs, h, w, d), dtype=torch.long, device=device)
    final_ssc_pred = things_filtered_ssc_pred.clone()
    for b in range(bs):
        inst_id = 1
        for idx in sorted_indices[b]:
            class_id = query_class_pred[b, idx]
            if class_id not in thing_classes:
                continue
            if mask_scores[b, idx] > query_threshold:
                mask_i = (voxel_masks[b, idx] & (final_ssc_pred[b] == 0))
                no_overlap_conflict = torch.sum(mask_i) / torch.sum(voxel_masks[b, idx]) > mask_overlap_threshold
                in_fov = torch.sum(mask_i & frustrum_masks[b]) / torch.sum(voxel_masks[b, idx]) > in_fov_threshold
                if no_overlap_conflict and in_fov:
                    final_ssc_pred[b] = torch.where(mask_i, class_id, final_ssc_pred[b])
                    merged_instance_pred[b] = torch.where(mask_i, inst_id, merged_instance_pred[b])
                    inst_id += 1
        

    return {
        'final_ssc_pred': final_ssc_pred,
        'merged_instance_pred': merged_instance_pred,
        'semantic_pred': semantic_pred,
        'sorted_indices': sorted_indices,
        'mask_scores': mask_scores,
        'query_class_pred': query_class_pred,
        'query_probs': query_probs,
        'voxel_probs': voxel_probs,
    }

def panoptic_inference(
        preds: Dict, 
        frustum_masks: Tensor, 
        thing_classes: List, 
    ) -> Dict:
    """
    Wrapper function of mask_wise_merge taking from PanoSSC. The softmax and sigmoid are applied inside the mask_wise_merge function.
    """
    return mask_wise_merge(
        voxel_logits=preds['voxel_logits'], 
        query_logits=preds['query_logits'], 
        ssc_logits=preds['ssc_logits'], 
        thing_classes=thing_classes, 
        frustrum_masks=frustum_masks,
    )

def semantic_inference(
        preds: Dict, 
        frustum_masks: Tensor, 
        thing_classes: List, 
    ) -> Tuple[Tensor, Tensor]:
    """
    Post-processing function for semantic prediction taking from panoptic_inference.
    """
    result = panoptic_inference(preds, frustum_masks, thing_classes)
    return (result['final_ssc_pred'], result['semantic_pred'])

def instance_inference(
        preds: Dict, 
        frustum_masks: Tensor, 
        thing_classes: List, 
    ) -> Dict:
    """
    Post-processing function for instance prediction taking from panoptic_inference.
    """
    result = panoptic_inference(preds, frustum_masks, thing_classes)
    keys_to_return = [
        'merged_instance_pred',
        'query_class_pred',
        'query_probs',
        'mask_scores',
        'sorted_indices',
        'voxel_probs'
    ]
    return {k: result[k] for k in keys_to_return}

