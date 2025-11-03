import os
import glob
import numpy as np
import torch
from mmdet.datasets import DATASETS
from torch.utils.data import Dataset
from mmdet.datasets.pipelines import Compose
import pickle
@DATASETS.register_module()
class SemanticKITTIDataset(Dataset):
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

        self.load_continuous = load_continuous
            
        self.splits = {
            "train": ["00", "01", "02", "03", "04", "05", "06", "07","09", "10"],
            "val": ["08"],
            "test": ["08"],
            "test_submit": ["11", "12", "13", "14", "15", "16", "17", "18", "19", "20", "21"],
        }
        self.class_freq = torch.tensor([
        5.41773033e09, 1.57835390e07, 1.25136000e05, 1.18809000e05, 6.46799000e05, 8.21951000e05,
        2.62978000e05, 2.83696000e05, 2.04750000e05, 6.16887030e07, 4.50296100e06, 4.48836500e07,
        2.26992300e06, 5.68402180e07, 1.57196520e07, 1.58442623e08, 2.06162300e06, 3.69705220e07,
        1.15198800e06, 3.34146000e05
        ])
        self.class_meta = {
        'class_weights':
        1 / torch.log(self.class_freq + 1e-6),
        'class_names':
        ('empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
         'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building', 'fence',
         'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign')
    }
        self.class_names = ['empty', 'car', 'bicycle', 'motorcycle', 'truck', 'other-vehicle', 'person', 'bicyclist',
                'motorcyclist', 'road', 'parking', 'sidewalk', 'other-ground', 'building, fence',
                'vegetation', 'trunk', 'terrain', 'pole', 'traffic-sign'],



        self.sequences = self.splits[split]

        self.data_root = data_root
        self.stereo_depth_root = stereo_depth_root
        self.preprocess_root = preprocess_root  
        self.ann_file = ann_file
        self.test_mode = test_mode
        self.data_infos = self.load_annotations(self.ann_file)
        self.thing_ids = thing_ids
        self.occ_size = occ_size
        self.pc_range = pc_range
        self.camera_map = {'left': '2', 'right': '3'}
        self.camera_used = [self.camera_map[camera] for camera in camera_used]

        if pipeline is not None:
            self.pipeline = Compose(pipeline)
        self._set_group_flag()

    def __len__(self):
        return len(self.data_infos)
    
    def prepare_train_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        modified_input_dict = input_dict.copy()
        for key in ["semantic_label", "instance_label", "mask_label"]:
            if key in modified_input_dict:
                del modified_input_dict[key]
        example = self.pipeline(input_dict)
        return example
    
    def prepare_test_data(self, index):
        """
        Training data preparation.
        Args:
            index (int): Index for accessing the target data.
        Returns:
            dict: Training data dict of the corresponding index.
        """
        input_dict = self.get_data_info(index)
        if input_dict is None:
            print('found None in training data')
            return None
        
        example = self.pipeline(input_dict)
        return example
    
    def __getitem__(self, idx):
        if self.test_mode:
            return self.prepare_test_data(idx)
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data
    
    def get_data_info(self, index):
        info = self.data_infos[index]
        '''
        sample info includes the following:
            "img_2_path": img_2_path,
            "img_3_path": img_3_path,
            "sequence": sequence,
            "P2": P2,
            "P3": P3,
            "T_velo_2_cam": T_velo_2_cam,
            "proj_matrix_2": proj_matrix_2,
            "proj_matrix_3": proj_matrix_3,
            "voxel_path": voxel_path,
        '''

        input_dict = dict(
            occ_size = np.array(self.occ_size),
            pc_range = np.array(self.pc_range),
            sequence = info['sequence'],
            frame_id = info['frame_id'],
        )

        image_paths = []
        lidar2cam_rts = []
        lidar2img_rts = []
        cam_intrinsics = []

        for cam_type in self.camera_used:
            image_paths.append(info['img_{}_path'.format(int(cam_type))])
            lidar2img_rts.append(info['proj_matrix_{}'.format(int(cam_type))])
            cam_intrinsics.append(info['P{}'.format(int(cam_type))])
            lidar2cam_rts.append(info['T_velo_2_cam'])
        
        focal_length = info['P2'][0, 0]
        baseline = self.dynamic_baseline(info)

        input_dict.update(
            dict(
                img_filename=image_paths,
                lidar2img=lidar2img_rts,
                cam_intrinsic=cam_intrinsics,
                lidar2cam=lidar2cam_rts,
                focal_length=focal_length,
                baseline=baseline
            ))
        input_dict['stereo_depth_path'] = info['stereo_depth_path']
        input_dict['class_meta'] = self.class_meta
        input_dict['gt_occ'] = self.get_ann_info(index, key='voxel_path')
        
        gt_data = self.get_ann_info(index)
        if gt_data:
            input_dict.update(gt_data) 

        return input_dict
    
    def load_annotations(self, ann_file=None):
        scans = []
        for sequence in self.sequences:
            calib = self.read_calib(
                os.path.join(self.data_root, "sequences", sequence, "calib.txt")
            )
            P2 = calib["P2"]
            P3 = calib["P3"]
            T_velo_2_cam = calib["Tr"]
            proj_matrix_2 = P2 @ T_velo_2_cam
            proj_matrix_3 = P3 @ T_velo_2_cam

            voxel_base_path = os.path.join(self.ann_file, sequence)
            img_base_path = os.path.join(self.data_root, "sequences", sequence)
            gt_base_path = os.path.join(self.preprocess_root, "instance_labels_v2", sequence)
            id_base_path = os.path.join(self.data_root, "sequences", sequence, 'voxels', '*.bin')            
            if self.load_continuous:
                id_base_path = os.path.join(self.data_root, "sequences", sequence, 'image_2', '*.png')
            else:
                id_base_path = os.path.join(self.data_root, "sequences", sequence, 'voxels', '*.bin')
            
            for id_path in glob.glob(id_base_path):
                img_id = id_path.split("/")[-1].split(".")[0]
                img_2_path = os.path.join(img_base_path, 'image_2', img_id + '.png')
                img_3_path = os.path.join(img_base_path, 'image_3', img_id + '.png')
                voxel_path = os.path.join(voxel_base_path, img_id + '_1_1.npy')
                stereo_depth_path = os.path.join(self.stereo_depth_root, "sequences", sequence, img_id + '.npy')
                gt_path = os.path.join(gt_base_path, f"{img_id}_1_1.pkl")
                if not os.path.exists(voxel_path):
                    voxel_path = None
                if not os.path.exists(gt_path):
                    gt_path = None 
                
                
                scans.append(
                    {   "img_2_path": img_2_path,
                        "img_3_path": img_3_path,
                        "sequence": sequence,
                        "frame_id": img_id,
                        "P2": P2,
                        "P3": P3,
                        "T_velo_2_cam": T_velo_2_cam,
                        "proj_matrix_2": proj_matrix_2,
                        "proj_matrix_3": proj_matrix_3,
                        "voxel_path": voxel_path,
                        "stereo_depth_path": stereo_depth_path,
                        "gt_path": gt_path
                    })
                
        return scans 
    
    def get_ann_info(self, index, key=None):
        """
        Loads ground truth (GT) information:
        - SSC GT from .npy (if `key='voxel_path'`)
        - Panoptic GT from .pkl (semantic, instance, mask_label)

        Args:
            index (int): Dataset index.
            key (str): Key to fetch SSC GT (`'voxel_path'`).

        Returns:
            dict or np.ndarray: SSC GT if `key='voxel_path'`, otherwise a dictionary with panoptic GT.
        """
        info = self.data_infos[index]

        
        if key == 'voxel_path':
            gt_occ = None if info[key] is None else np.load(info[key])
            return gt_occ 

        
        frame_id = info['frame_id']
        sequence = info['sequence']
        pkl_path = os.path.join(self.preprocess_root, "instance_labels_v2", sequence, f"{frame_id}_1_1.pkl")

        if not os.path.exists(pkl_path):
            print(f"⚠️ Warning: Missing GT file {pkl_path}")
            return None

        with open(pkl_path, "rb") as handle:
            data = pickle.load(handle)

        semantic_label = torch.from_numpy(data["semantic_labels"]).type(torch.uint8)
        instance_label = torch.from_numpy(data["instance_labels"]).type(torch.uint8)
        mask_label = self.prepare_mask_label(semantic_label, instance_label)


        return {
            "semantic_label": semantic_label,
            "instance_label": instance_label,
            "mask_label": mask_label
        }

    
    @staticmethod
    def read_calib(calib_path):
        """calib.txt: Calibration data for the cameras: P0/P1 are the 3x4 projection
            matrices after rectification. Here P0 denotes the left and P1 denotes the
            right camera. Tr transforms a point from velodyne coordinates into the
            left rectified camera coordinate system. In order to map a point X from the
            velodyne scanner to a point x in the i'th image plane, you thus have to
            transform it like:
            x = Pi * Tr * X
            - 'image_00': left rectified grayscale image sequence
            - 'image_01': right rectified grayscale image sequence
            - 'image_02': left rectified color image sequence
            - 'image_03': right rectified color image sequence
        """
        calib_all = {}
        with open(calib_path, "r") as f:
            for line in f.readlines():
                if line == "\n":
                    break
                key, value = line.split(":", 1)
                calib_all[key] = np.array([float(x) for x in value.split()])

        calib_out = {}
        calib_out["P2"] = np.identity(4) 
        calib_out["P3"] = np.identity(4) 
        calib_out["P2"][:3, :4] = calib_all["P2"].reshape(3, 4)
        calib_out["P3"][:3, :4] = calib_all["P3"].reshape(3, 4)
        calib_out["Tr"] = np.identity(4) 
        calib_out["Tr"][:3, :4] = calib_all["Tr"].reshape(3, 4) 
        
        return calib_out
    
    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)
    
    def _set_group_flag(self):
        """Set flag according to image aspect ratio.

        Images with aspect ratio greater than 1 will be set as group 1,
        otherwise group 0. In 3D datasets, they are all the same, thus are all
        zeros.
        """
        self.flag = np.zeros(len(self), dtype=np.uint8)
    
    def dynamic_baseline(self, infos):
        P3 = infos['P3']
        P2 = infos['P2']
        baseline = P3[0,3]/(-P3[0,0]) - P2[0,3]/(-P2[0,0])
        return baseline
    
    def prepare_mask_label(self, semantic_label, instance_label):
        """
        Converts semantic & instance labels into panoptic masks.
        """
        mask_semantic_label = self.prepare_target(semantic_label, ignore_labels=[0, 255])

        stuff_filtered_mask = [
            t not in self.thing_ids for t in mask_semantic_label["labels"]
        ]
        stuff_semantic_labels = mask_semantic_label["labels"][stuff_filtered_mask]
        stuff_semantic_masks = mask_semantic_label["masks"][stuff_filtered_mask]
    
        labels = [stuff_semantic_labels]
        masks = [stuff_semantic_masks]

        mask_instance_label = self.prepare_instance_target(
            semantic_target=semantic_label,
            instance_target=instance_label,
            ignore_label=0,
        )

        if mask_instance_label is not None:
            labels.append(mask_instance_label["labels"])
            masks.append(mask_instance_label["masks"])

        return {
            "labels": torch.cat(labels, dim=0),
            "masks": torch.cat(masks, dim=0).permute(1, 2, 3, 0),
        }
    def prepare_target(self, target: torch.Tensor, ignore_labels: list) -> dict:
        """
        Creates binary masks for each unique label in the target tensor.

        Args:
            target (torch.Tensor): The semantic or instance label tensor.
            ignore_labels (list): List of label values to ignore.

        Returns:
            dict: A dictionary containing:
                - "labels": A tensor of unique valid labels.
                - "masks": A tensor of corresponding binary masks.
        """
        unique_ids = torch.unique(target)
        unique_ids = torch.tensor(
            [uid for uid in unique_ids if uid not in ignore_labels]
        )

        if len(unique_ids) == 0:
            return {"labels": torch.tensor([], dtype=torch.uint8), "masks": torch.tensor([], dtype=torch.bool)}

        masks = [target == uid for uid in unique_ids]
        masks = torch.stack(masks, dim=0)

        return {"labels": unique_ids, "masks": masks}
    def prepare_instance_target(self, semantic_target: torch.Tensor, instance_target: torch.Tensor, ignore_label: int) -> dict:
        """
        Converts instance labels into instance-level masks.

        Args:
            semantic_target (torch.Tensor): Semantic label tensor.
            instance_target (torch.Tensor): Instance label tensor.
            ignore_label (int): Value to ignore in instance segmentation.

        Returns:
            dict: A dictionary containing:
                - "labels": A tensor of semantic labels per instance.
                - "masks": A tensor of instance segmentation masks.
        """
        unique_instance_ids = torch.unique(instance_target)
        unique_instance_ids = unique_instance_ids[unique_instance_ids != ignore_label] 
        
        if len(unique_instance_ids) == 0:
            return None 
        
        masks = []
        semantic_labels = []

        for instance_id in unique_instance_ids:
            masks.append(instance_target == instance_id)
            semantic_labels.append(semantic_target[instance_target == instance_id][0]) 
        
        masks = torch.stack(masks, dim=0) if masks else torch.empty(0, *instance_target.shape, dtype=torch.bool)
        semantic_labels = torch.tensor(semantic_labels, dtype=torch.uint8) if semantic_labels else torch.tensor([], dtype=torch.uint8)

        return {"labels": semantic_labels, "masks": masks}


    def test_dataset_loading(self, num_samples=2):
        """
        Test method to verify the dataset loading, GT structure, and ensure everything is correct.
        
        Args:
            num_samples (int): Number of samples to print for verification.
        """
        from torch.utils.data import DataLoader

        print("\n🔹 **Testing Dataset Loading...**")
        dataset = self
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

        for i, sample in enumerate(dataloader):
            if i >= num_samples:
                break 

            print(f"\n🟢 **Sample {i+1}/{num_samples}**")
            print(f"📌 Frame ID: {sample['frame_id'][0]} | Sequence: {sample['sequence'][0]}\n")

            print("📸 Image Paths:")
            print(f"   Left Image: {sample['img_filename'][0][0]}")
            print(f"   Right Image: {sample['img_filename'][0][1]}\n")

            print("📷 Camera Extrinsics:")
            print(f"   Projection Matrix (Left):\n{sample['lidar2img'][0][0]}")
            print(f"   Projection Matrix (Right):\n{sample['lidar2img'][0][1]}\n")

            if sample['gt_occ'] is not None:
                print(f"🟢 **SSC Ground Truth (gt_occ) Shape:** {sample['gt_occ'].shape}\n")
            else:
                print("⚠️ **Warning:** No SSC GT Found.\n")

            if "semantic_label" in sample and sample["semantic_label"] is not None:
                print(f"🟢 **Semantic Label Shape:** {sample['semantic_label'].shape}")
                print(f"   Unique Semantic Classes: {torch.unique(sample['semantic_label'])}\n")
            else:
                print("⚠️ **Warning:** No Semantic Labels Found.\n")

            if "instance_label" in sample and sample["instance_label"] is not None:
                print(f"🟢 **Instance Label Shape:** {sample['instance_label'].shape}")
                print(f"   Unique Instances: {torch.unique(sample['instance_label'])}\n")
            else:
                print("⚠️ **Warning:** No Instance Labels Found.\n")

            if "mask_label" in sample and sample["mask_label"] is not None:
                mask_labels = sample["mask_label"]["labels"]
                mask_masks = sample["mask_label"]["masks"]

                print(f"🟢 **Panoptic Mask Labels Shape:** {mask_masks.shape}")
                print(f"   Unique Panoptic Labels: {mask_labels}\n")
            else:
                print("⚠️ **Warning:** No Panoptic Masks Found.\n")

        print("\n✅ **Dataset Test Completed Successfully!**\n")
if __name__ == "__main__":
    data_root = '/home/jovyan/danit-semantickitti/kitti/dataset'
    ann_file = '/home/jovyan/danit-semantickitti/kitti/dataset/labels'
    stereo_depth_root = '/home/jovyan/danit-semantickitti/kitti/dataset/depth'
    preprocess_root = '/home/jovyan/danit-semantickitti/pasco_preprocess/kitti'
    thing_ids = [1, 2, 3, 4, 5, 6, 7, 8] 
    data_config={
    'input_size': (384, 1280),
    'resize': (0., 0.),
    'rot': (0.0, 0.0 ),
    'flip': (0.0, 0.0 ),
    'flip': False,
    'crop_h': (0.0, 0.0),
    'resize_test': 0.00,
    }
    bda_aug_conf = dict(
    rot_lim=(-22.5, 22.5),
    scale_lim=(0.95, 1.05),
    flip_dx_ratio=0.5,
    flip_dy_ratio=0.5,
    flip_dz_ratio=0
    )
    point_cloud_range = [0, -25.6, -2, 51.2, 25.6, 4.4]
    pipeline = [
        dict(type='LoadMultiViewImageFromFiles', data_config=data_config, load_stereo_depth=True,
         is_train=True, color_jitter=(0.4, 0.4, 0.4)),
        dict(type='CreateDepthFromLiDAR', data_root=data_root, dataset='kitti', load_seg=False),
        dict(type='LoadAnnotationOcc', bda_aug_conf=bda_aug_conf, apply_bda=False,
            is_train=True, point_cloud_range=point_cloud_range),
        dict(type='CollectData', keys=['img_inputs', 'gt_occ'], 
            meta_keys=['pc_range', 'occ_size', 'raw_img', 'stereo_depth', 'focal_length', 'baseline', 'img_shape', 'gt_depths']),
    ]
    dataset = SemanticKITTIDataset(
        data_root=data_root,
        stereo_depth_root=stereo_depth_root,
        ann_file=ann_file,
        preprocess_root=preprocess_root,
        pipeline=None,
        split="val",
        camera_used=["left", "right"],
        occ_size=(256, 256, 32),
        pc_range=point_cloud_range,
        thing_ids=thing_ids,
        test_mode=False
    )

    dataset.test_dataset_loading(num_samples=2)
