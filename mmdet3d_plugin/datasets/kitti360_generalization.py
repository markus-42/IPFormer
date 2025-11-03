import os
import glob
import numpy as np
import torch
import pickle
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet.datasets.pipelines import Compose

@DATASETS.register_module()
class KITTI360DatasetGen(Dataset):
    def __init__(
        self,
        data_root,
        stereo_depth_root,
        ann_file,
        preprocess_root,
        pipeline,
        split,
        camera_used,
        occ_size,
        pc_range,
        thing_ids,
        test_mode=False,
        load_continuous=False
    ):
        super().__init__()
        
        self.splits = {
            "train": [
                "2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync",
                "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync",
                "2013_05_28_drive_0010_sync"
            ],
            "val": ["2013_05_28_drive_0006_sync"],
            "test": ["2013_05_28_drive_0009_sync"],
            "all":[
                "2013_05_28_drive_0000_sync", "2013_05_28_drive_0002_sync", "2013_05_28_drive_0003_sync",
                "2013_05_28_drive_0004_sync", "2013_05_28_drive_0005_sync", "2013_05_28_drive_0007_sync",
                "2013_05_28_drive_0010_sync", "2013_05_28_drive_0006_sync", "2013_05_28_drive_0009_sync"
            ]
        }


        self.sequences = self.splits[split]
        self.data_root = data_root
        self.stereo_depth_root = stereo_depth_root
        self.ann_file = ann_file
        self.preprocess_root = preprocess_root
        self.test_mode = test_mode
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.thing_ids = thing_ids
        self.class_names = [
            'unlabeled',  
            'car',        
            'bicycle',    
            'motorcycle', 
            'truck',      
            'other-vehicle', 
            'person',     
            'ignore',     
            'ignore',     
            'road',       
            'parking',    
            'sidewalk',   
            'other-ground',# 12
            'building',   
            'fence',      
            'vegetation', 
            'ignore',     
            'terrain',    
            'pole',       
            'traffic-sign'
        ]
        self.class_meta = {
            'class_names': self.class_names,
            'class_weights': torch.ones(len(self.class_names))
        }

        self.camera_map = {'left': '00', 'right': '01'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]

        self.data_infos = self.load_annotations()
        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def load_annotations(self):
        scans = []
        for sequence in self.sequences:
            P0, P1, Tr = self.read_calib()
            proj_matrix_0 = P0 @ Tr
            proj_matrix_1 = P1 @ Tr

            voxel_base_path = os.path.join(self.ann_file, sequence)
            gt_base_path = os.path.join(self.preprocess_root, "instance_labels_v2", sequence)
            img_base_path = os.path.join(self.data_root, 'data_2d_raw', sequence)

            img_pattern = os.path.join(img_base_path, 'image_00', 'data_rect', '*.png')
            for img_path in sorted(glob.glob(img_pattern)):
                img_id = os.path.basename(img_path).split('.')[0]

                img_00_path = os.path.join(img_base_path, 'image_00', 'data_rect', img_id + '.png')
                img_01_path = os.path.join(img_base_path, 'image_01', 'data_rect', img_id + '.png')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')
                stereo_depth_path = os.path.join(self.stereo_depth_root, "sequences",sequence, img_id + '.npy')
                gt_path = os.path.join(gt_base_path, img_id + '_1_1.pkl')
                if not os.path.exists(gt_path):
                    continue 

                scans.append({
                    "img_00_path": img_00_path,
                    "img_01_path": img_01_path,
                    "sequence": sequence,
                    "frame_id": img_id,
                    "P00": P0,
                    "P01": P1,
                    "T_velo_2_cam": Tr,
                    "proj_matrix_00": proj_matrix_0,
                    "proj_matrix_01": proj_matrix_1,
                    "voxel_path": voxel_path if os.path.exists(voxel_path) else None,
                    "stereo_depth_path": stereo_depth_path,
                    "gt_path": gt_path if os.path.exists(gt_path) else None
                })
        return scans

    def get_data_info(self, index):
        info = self.data_infos[index]
        input_dict = dict(
            occ_size=np.array(self.occ_size),
            pc_range=np.array(self.pc_range),
            sequence=info['sequence'],
            frame_id=info['frame_id'],
            class_meta=self.class_meta
        )

        image_paths, lidar2img_rts, cam_intrinsics, lidar2cam_rts = [], [], [], []
        for cam_type in self.camera_used:
            image_paths.append(info[f'img_{cam_type}_path'])
            lidar2img_rts.append(info[f'proj_matrix_{cam_type}'])
            cam_intrinsics.append(info[f'P{cam_type}'])
            lidar2cam_rts.append(info['T_velo_2_cam'])

        input_dict.update(dict(
            img_filename=image_paths,
            lidar2img=lidar2img_rts,
            cam_intrinsic=cam_intrinsics,
            lidar2cam=lidar2cam_rts,
            focal_length=info['P00'][0, 0],
            baseline=self.dynamic_baseline(info),
            stereo_depth_path=info['stereo_depth_path'],
            gt_occ=self.get_ann_info(index, key='voxel_path')
        ))

        gt_data = self.get_ann_info(index)
        if gt_data:
            input_dict.update(gt_data)

        return input_dict

    def get_ann_info(self, index, key=None):
        def shift_and_pad(tensor, shift_voxels, pad_value=0, out_shape=(256, 256, 32)):
            """
            Shift a voxel tensor using pad and slice.
            Args:
                tensor: Tensor of shape [D, H, W]
                shift_voxels: (x, y, z)
                pad_value: int
                out_shape: (D, H, W)
            Returns:
                Tensor of shape [D, H, W] with dtype preserved
            """
            sx, sy, sz = shift_voxels
            pad_width = (
                (max(sz, 0), max(-sz, 0)), 
                (max(sy, 0), max(-sy, 0)), 
                (max(sx, 0), max(-sx, 0))  
            )
            padded = torch.nn.functional.pad(tensor, (
                pad_width[0][0], pad_width[0][1], 
                pad_width[1][0], pad_width[1][1], 
                pad_width[2][0], pad_width[2][1]  
            ), value=pad_value)

            x_start = pad_width[2][0] - sx
            y_start = pad_width[1][0] - sy
            z_start = pad_width[0][0] - sz

            return padded[x_start:x_start+out_shape[0],
                        y_start:y_start+out_shape[1],
                        z_start:z_start+out_shape[2]].type(tensor.dtype)

        info = self.data_infos[index]
        if key == 'voxel_path':
            return None if info[key] is None else np.load(info[key])

        pkl_path = info.get("gt_path", None)
        if not pkl_path or not os.path.exists(pkl_path):
            return None

        with open(pkl_path, 'rb') as handle:
            data = pickle.load(handle)

        semantic_label = torch.from_numpy(data["semantic_labels"]).type(torch.uint8)
        instance_label = torch.from_numpy(data["instance_labels"]).type(torch.uint8)

        remap_lut = torch.tensor([
            0, 
            1, 
            2, 
            3, 
            4, 
            5, 
            6, 
            9, 
            10,
            11,
            12,
            13,
            14,
            15,
            17,
            18,
            19,
            0, 
            0  
        ], dtype=torch.uint8)

        semantic_label_long = semantic_label.clone().long()
        mask = (semantic_label_long < len(remap_lut)) & (semantic_label_long != 255)
        semantic_label[mask] = remap_lut[semantic_label_long[mask]]
        semantic_label = semantic_label.type(torch.uint8)

        shift_voxels = (4, 2, -1)
        target_shape = (256, 256, 32)
        semantic_pad_value = 255 
        instance_pad_value = 0

        semantic_label = shift_and_pad(semantic_label, shift_voxels, pad_value=semantic_pad_value, out_shape=target_shape)
        instance_label = shift_and_pad(instance_label, shift_voxels, pad_value=instance_pad_value, out_shape=target_shape)

        mask_label = self.prepare_mask_label(semantic_label, instance_label)
        return {
            "semantic_label": semantic_label,
            "instance_label": instance_label,
            "mask_label": mask_label
        }

    def prepare_mask_label(self, semantic_label, instance_label):
        mask_semantic = self.prepare_target(semantic_label, ignore_labels=[0, 255])
        stuff_mask = [t not in self.thing_ids for t in mask_semantic["labels"]]
        labels = [mask_semantic["labels"][stuff_mask]]
        masks = [mask_semantic["masks"][stuff_mask]]
        inst = self.prepare_instance_target(semantic_label, instance_label, ignore_label=0)
        if inst:
            labels.append(inst["labels"])
            masks.append(inst["masks"])
        return {
            "labels": torch.cat(labels, dim=0),
            "masks": torch.cat(masks, dim=0).permute(1, 2, 3, 0)
        }

    def prepare_target(self, target, ignore_labels):
        uids = torch.unique(target)
        uids = torch.tensor([uid for uid in uids if uid not in ignore_labels])
        if len(uids) == 0:
            return {"labels": torch.tensor([], dtype=torch.uint8), "masks": torch.tensor([], dtype=torch.bool)}
        masks = [target == uid for uid in uids]
        return {"labels": uids, "masks": torch.stack(masks, dim=0)}

    def prepare_instance_target(self, semantic_target, instance_target, ignore_label):
        ids = torch.unique(instance_target)
        ids = ids[ids != ignore_label]
        if len(ids) == 0:
            return None
        masks = [instance_target == i for i in ids]
        labels = [semantic_target[instance_target == i][0] for i in ids]
        return {"labels": torch.tensor(labels, dtype=torch.uint8), "masks": torch.stack(masks)}

    @staticmethod
    def read_calib(calib_path=None):
        """
        Tr transforms a point from velodyne coordinates into the 
        left rectified camera coordinate system.
        In order to map a point X from the velodyne scanner to a 
        point x in the i'th image plane, you thus have to transform it like:
        x = Pi * Tr * X
        """
        P2 = np.array([
            [552.554261, 0.000000, 682.049453, 0.000000],
            [0.000000, 552.554261, 238.769549, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000]
        ]).reshape(3, 4)

        P3 = np.array([
            [552.554261, 0.000000, 682.049453, -328.318735],
            [0.000000, 552.554261, 238.769549, 0.000000],
            [0.000000, 0.000000, 1.000000, 0.000000]
        ]).reshape(3, 4)

        cam2velo = np.array([
            [0.04307104361, -0.08829286498, 0.995162929, 0.8043914418],
            [-0.999004371, 0.007784614041, 0.04392796942, 0.2993489574],
            [-0.01162548558, -0.9960641394, -0.08786966659, -0.1770225824],
            [0, 0, 0, 1]
        ]).reshape(4, 4)

        velo2cam = np.linalg.inv(cam2velo)

        calib_out = {}
        calib_out["P2"] = np.identity(4)
        calib_out["P3"] = np.identity(4)
        calib_out["P2"][:3, :4] = P2
        calib_out["P3"][:3, :4] = P3
        calib_out["Tr"] = np.identity(4)
        calib_out["Tr"][:3, :4] = velo2cam[:3, :4]
        return calib_out["P2"], calib_out["P3"], calib_out["Tr"]

    def dynamic_baseline(self, info):
        P1 = info['P01']
        P0 = info['P00']
        return P1[0, 3] / (-P1[0, 0]) - P0[0, 3] / (-P0[0, 0])

    def __len__(self):
        return len(self.data_infos)

    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def prepare_train_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        return self.pipeline(input_dict)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        if input_dict is None:
            return None
        return self.pipeline(input_dict)

    def _rand_another(self, idx):
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def _set_group_flag(self):
        self.flag = np.zeros(len(self), dtype=np.uint8)
