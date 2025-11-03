import torch
import numpy as np
from PIL import Image
from mmdet.datasets.builder import PIPELINES

@PIPELINES.register_module()
class LoadAnnotationOccGen():
    def __init__(self, bda_aug_conf, is_train=True, apply_bda=False, 
                 point_cloud_range=[0, -25.6, -2, 51.2, 25.6, 4.4]):
        self.bda_aug_conf = bda_aug_conf
        self.is_train = is_train
        self.point_cloud_range = torch.tensor(point_cloud_range)
        self.transform_center = (self.point_cloud_range[:3] + self.point_cloud_range[3:]) / 2

        self.apply_bda = apply_bda
    
    def sample_bda_augmentation(self):
        """Generate bda augmentation values based on bda_config."""

        rotate_bda = np.random.uniform(*self.bda_aug_conf['rot_lim'])
        scale_bda = np.random.uniform(*self.bda_aug_conf['scale_lim'])
        flip_dx = np.random.uniform() < self.bda_aug_conf['flip_dx_ratio']
        flip_dy = np.random.uniform() < self.bda_aug_conf['flip_dy_ratio']
        flip_dz = np.random.uniform() < self.bda_aug_conf['flip_dz_ratio']
        
        return rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz
    
    def forward_test(self, results):
        bda_rot = torch.eye(4).float()
        imgs, rots, trans, intrins, post_rots, post_trans, sensor2sensors = results['img_inputs']

        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, sensor2sensors)
        return results
    
    def __call__(self, results):
        if results['gt_occ'] is None:
            return self.forward_test(results)

        if isinstance(results['gt_occ'], list):
            gt_occ = [torch.tensor(x).long() for x in results['gt_occ']]
        else:
            gt_occ = torch.tensor(results['gt_occ']).long().clone()

            kitti360_to_semkitti = torch.tensor([
                0, 1, 2, 3, 4, 5, 6, 9, 10, 11,
                12, 13, 14, 15, 0, 17, 18, 19, 0, 0
            ])
            remap_mask = gt_occ < len(kitti360_to_semkitti)
            gt_occ[remap_mask] = kitti360_to_semkitti[gt_occ[remap_mask]]

            shift_voxels = torch.tensor([4, 2, -1]) 
            pad_width = (
                (max(shift_voxels[0], 0), max(-shift_voxels[0], 0)),
                (max(shift_voxels[1], 0), max(-shift_voxels[1], 0)),
                (max(shift_voxels[2], 0), max(-shift_voxels[2], 0)),
            )
            gt_occ = torch.nn.functional.pad(gt_occ, (
                pad_width[2][0], pad_width[2][1],
                pad_width[1][0], pad_width[1][1],
                pad_width[0][0], pad_width[0][1],
            ), value=255) 

            x0 = pad_width[0][0] - shift_voxels[0]
            y0 = pad_width[1][0] - shift_voxels[1]
            z0 = pad_width[2][0] - shift_voxels[2]
            gt_occ = gt_occ[x0:x0+256, y0:y0+256, z0:z0+32]

        results['gt_occ'] = gt_occ

        if self.is_train:
            if self.apply_bda:
                rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz = self.sample_bda_augmentation()
                gt_occ, bda_rot = voxel_transform(
                    gt_occ, rotate_bda, scale_bda, flip_dx, flip_dy, flip_dz, self.transform_center
                )
            else:
                bda_rot = torch.eye(4).float()
        else:
            bda_rot = torch.eye(4).float()

        imgs, rots, trans, intrins, post_rots, post_trans, sensor2sensors = results['img_inputs']
        results['img_inputs'] = (imgs, rots, trans, intrins, post_rots, post_trans, bda_rot, sensor2sensors)
        results['img_shape'] = imgs.shape[-2:]
        results['gt_occ'] = gt_occ.long()

        return results


def voxel_transform(voxel_labels, rotate_angle, scale_ratio, flip_dx, flip_dy, flip_dz, transform_center=None):
    assert transform_center is not None
    trans_norm = torch.eye(4)
    trans_norm[:3, -1] = - transform_center
    trans_denorm = torch.eye(4)
    trans_denorm[:3, -1] = transform_center
    
    rotate_degree = rotate_angle
    rotate_angle = torch.tensor(rotate_angle / 180 * np.pi)
    rot_sin = torch.sin(rotate_angle)
    rot_cos = torch.cos(rotate_angle)
    rot_mat = torch.Tensor([
        [rot_cos, -rot_sin, 0, 0],
        [rot_sin, rot_cos, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]])
    
    flip_mat = torch.eye(4)
    if flip_dx:
        flip_mat = flip_mat @ torch.Tensor([
            [-1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if flip_dy:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0], 
            [0, -1, 0, 0], 
            [0, 0, 1, 0],
            [0, 0, 0, 1]])
    
    if flip_dz:
        flip_mat = flip_mat @ torch.Tensor([
            [1, 0, 0, 0], 
            [0, 1, 0, 0], 
            [0, 0, -1, 0],
            [0, 0, 0, 1]])
    
    bda_mat = trans_denorm @ flip_mat @ rot_mat @ trans_norm
    
    voxel_labels = voxel_labels.numpy().astype(np.uint8)
    
    if not np.isclose(rotate_degree, 0):
        voxel_labels = custom_rotate_3d(voxel_labels, rotate_degree)
    
    if flip_dz:
        voxel_labels = voxel_labels[:, :, ::-1]
    
    if flip_dy:
        voxel_labels = voxel_labels[:, ::-1]
    
    if flip_dx:
        voxel_labels = voxel_labels[::-1]
    
    voxel_labels = torch.from_numpy(voxel_labels.copy()).long()
    
    return voxel_labels, bda_mat

def custom_rotate_3d(voxel_labels, rotate_degree):
    is_tensor = False
    if type(voxel_labels) is torch.Tensor:
        is_tensor = True
        voxel_labels = voxel_labels.numpy().astype(np.uint8)
    
    voxel_labels_list = []
    for height_index in range(voxel_labels.shape[-1]):
        bev_labels = voxel_labels[..., height_index]
        bev_labels = Image.fromarray(bev_labels.astype(np.uint8))
        bev_labels = bev_labels.rotate(rotate_degree, resample=Image.Resampling.NEAREST, fillcolor=255)
        bev_labels = np.array(bev_labels)
        voxel_labels_list.append(bev_labels)
    voxel_labels = np.stack(voxel_labels_list, axis=-1)
    
    if is_tensor:
        voxel_labels = torch.from_numpy(voxel_labels).long()
    
    return voxel_labels