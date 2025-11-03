# Dataset Preparation

This guide explains how to prepare datasets for **IPFormer**. It covers **SemanticKITTI** and **SSCBench‑KITTI‑360**, including **depth map** generation and **panoptic label** preparation.

- Depth map generation procedure follows [VoxFormer](https://github.com/NVlabs/VoxFormer)
- SSCBench‑KITTI‑360 data organization follows [SSCBench](https://github.com/ai4ce/SSCBench)  
- Panoptic label generation follows [PaSCo](https://github.com/astra-vision/PaSCo)


---

## 1. SemanticKITTI

### 1.1 Download

- [KITTI Odometry](https://www.cvlibs.net/datasets/kitti/eval_odometry.php  ) raw data (RGB, Velodyne, calibration)
- [SemanticKITTI](http://www.semantic-kitti.org/dataset.html#download) SSC ground-truth

Place the dataset at:
```
./dataset/SemanticKITTI/
```

### 1.2 RGB‑to‑Depth and SSC Preprocess

Generate depth maps and prepare SSC inputs by following the [VoxFormer guide](https://github.com/NVlabs/VoxFormer/blob/main/docs/prepare_dataset.md) . In our repo, the necessary helper scripts to follow along are placed at `./preprocess/`:

```bash
bash ./preprocess/image2depth_semantickitti.sh
```
```bash
python ./preprocess/preprocess.py \
  --kitti_root ./dataset/SemanticKITTI \
  --kitti_preprocess_root ./dataset/SemanticKITTI
```

### 1.3 Expected Folder Structure

```
/semantickittii/
          |-- sequences/
          │       |-- 00/
          │       │   |-- poses.txt
          │       │   |-- calib.txt
          │       │   |-- image_2/
          │       │   |-- image_3/
          │       |   |-- voxels/
          │       |         |- 000000.bin
          │       |         |- 000000.label
          │       |         |- 000000.occluded
          │       |         |- 000000.invalid
          │       |         |- 000005.bin
          │       |         |- 000005.label
          │       |         |- 000005.occluded
          │       |         |- 000005.invalid
          │       |-- 01/
          │       |-- 02/
          │       .
          │       |-- 21/
          |-- labels/
          │       |-- 00/
          │       │   |-- 000000_1_1.npy
          │       │   |-- 000000_1_2.npy
          │       │   |-- 000005_1_1.npy
          │       │   |-- 000005_1_2.npy
          │       |-- 01/
          │       .
          │       |-- 10/
          |-- lidarseg/
          |       |-- 00/
          |       │   |-- labels/
          |       |         ├ 000001.label
          |       |         ├ 000002.label
          |       |-- 01/
          |       |-- 02/
          |       .
          |       |-- 21/
          |-- depth/sequences/
          		  |-- 00/
          		  │   |-- 000000.npy
          		  |   |-- 000001.npy
          		  |-- 01/
                  |-- 02/
                  .
                  |-- 21/
          
```

---

## 2. SSCBench‑KITTI‑360

### 2.1 Download

Follow [SSCBench](https://github.com/ai4ce/SSCBench)   to obtain KITTI‑360 data and voxelized labels. Place the dataset at:
```
./dataset/SSCBenchKITTI360/
```

For depth generation, use the helper script in this repo:
```bash
bash ./preprocess/image2depth_kitti360.sh
```

### 2.2 Expected Folder Structure

```
/SSCBenchKITTI360/
    |-- data_2d_raw
    |   	|-- 2013_05_28_drive_0000_sync # train:[0, 2, 3, 4, 5, 7, 10] + val:[6] + test:[9]
    |   	|   |-- image_00
    |   	|   |   |-- data_rect # RGB images for left camera
    |   	|   |   |   |-- 000000.png
    |   	|   |   |   |-- 000001.png
    |   	|   |   |   |-- ...
    |   	|   |   |-- timestamps.txt
    |   	|   |-- image_01
    |   	|   |   |-- data_rect # RGB images for right camera
    |   	|   |   |   |-- 000000.png
    |   	|   |   |   |-- 000001.png
    |   	|   |   |   |-- ...
    |   	|   |   |-- timestamps.txt
    |   	|   |-- voxels # voxelized point clouds
    |   	|   |   |-- 000000.bin # voxelized input
    |   	|   |   |-- 000000.invalid # voxelized invalid mask
    |   	|   |   |-- 000000.label  #voxelized label
    |   	|   |   |-- 000005.bin # calculate every 5 frames 
    |   	|   |   |-- 000005.invalid
    |   	|   |   |-- 000005.label
    |   	|   |   |-- ...
    |   	|   |-- cam0_to_world.txt
    |   	|   |-- pose.txt # car pose information
    |   	|-- ...
    |   	|-- 2013_05_28_drive_0010_sync 
    |-- labels
    |       |-- 2013_05_28_drive_0000_sync 
    |       |   |-- 000000_1_1.npy # original labels
    |       |   |-- 000000_1_8.npy # 8x downsampled labels
    |       |   |-- 000005_1_1.npy
    |       |   |-- 000005_1_8.npy
    |       |   |-- ...
    |       |-- ... 
    |       |-- 2013_05_28_drive_0010_sync
    |-- labels_half # not unified, downsampled 
    |       |-- 2013_05_28_drive_0000_sync 
    |       |   |-- 000000_1_1.npy # original labels
    |       |   |-- 000000_1_8.npy # 8x downsampled labels
    |       |   |-- 000005_1_1.npy
    |       |   |-- 000005_1_8.npy
    |       |   |-- ...
    |       |-- ... 
    |       |-- 2013_05_28_drive_0010_sync
    |-- unified # unified
    |       |-- labels
    |           |-- 2013_05_28_drive_0000_sync 
    |           |   |-- 000000_1_1.npy # original labels
    |           |   |-- 000000_1_8.npy # 8x downsampled labels
    |           |   |-- 000005_1_1.npy
    |           |   |-- 000005_1_8.npy
    |           |   |-- ...
    |           |-- ... 
    |           |-- 2013_05_28_drive_0010_sync
    |-- calibration # preprocessed downsampled labels
    |   |-- calib_cam_to_pose.txt
    |   |-- calib_cam_to_velo.txt
    |   |-- calib_sick_to_velo.txt
    |   |-- image_02.yaml
    |   |-- image_03.yaml
    |   |-- perspective.txt
    |-- depth
     		|-- sequences
     			|-- 2013_05_28_drive_0000_sync
     			|	|-- 000000.npy
     			|	|-- 000001.npy
     			|-- ...
    			|-- 2013_05_28_drive_0010_sync
```


---

## 3. Panoptic Label Generation

Panoptic labels are generated by clustering the objects of the SSC ground-truth. The following procedure is identical to [PaSCo](https://github.com/astra-vision/PaSCo), which introduced the task of Panoptic Scene Completion.

### 3.1 PSC Labels for SemanticKITTI

1. Choose a preprocess root to store panoptic labels, for example:
   ```
   ./dataset/SemanticKITTI
   ```

2. **Option A: Generate locally** using this repository’s script:
   ```bash
   # From repository root
   python ./preprocess/gen_instance_labels.py \
     --dataset semantic_kitti \
     --kitti_root ./dataset/SemanticKITTI \
     --preprocess_root ./dataset/SemanticKITTI \
     --n_process 10
   ```
   Note: `--n_process` controls CPU parallelism.

3. **Option B: Download pre‑generated labels** released by [PaSCo](https://github.com/astra-vision/PaSCo) 
   ```bash
   cd ./dataset/SemanticKITTI
   wget https://github.com/astra-vision/PaSCo/releases/download/v0.0.1/kitti_instance_label_v2.tar.gz
   tar xvf kitti_instance_label_v2.tar.gz
   ```

4. Resulting layout (labels inside `instance_labels_v2`):
   ```
   ./dataset/SemanticKITTI/
   └── instance_labels_v2
       ├── 00
       ├── 01
       ├── 02
       ├── 03
       ├── 04
       ├── 05
       ├── 06
       ├── 07
       ├── 08
       ├── 09
       └── 10
   ```

### 3.2 PSC Labels for SSCBench‑KITTI‑360

1. Choose a preprocess root to store panoptic labels, for example:
   ```
   ./dataset/SSCBenchKITTI360
   ```

2. **Option A: Generate locally** using this repository’s script:
   ```bash
   python ./preprocess/gen_instance_labels.py \
     --dataset kitti360 \
     --kitti360_root ./dataset/SSCBenchKITTI360 \
     --preprocess_root ./dataset/SSCBenchKITTI360 \
     --n_process 10
   ```
   Note: `--n_process` controls CPU parallelism.

3. **Option B: Download pre‑generated labels** released by [PaSCo](https://github.com/astra-vision/PaSCo)
   ```bash
   cd ./dataset/SSCBenchKITTI360
   mkdir -p instance_labels_v2
   cd instance_labels_v2

   wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0000_sync.tar.gz
   wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0002_sync.tar.gz
   wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0003_sync.tar.gz
   wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0004_sync.tar.gz
   wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0005_sync.tar.gz
   wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0006_sync.tar.gz
   wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0007_sync.tar.gz
   wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0009_sync.tar.gz
   wget https://github.com/astra-vision/PaSCo/releases/download/v0.1.0/2013_05_28_drive_0010_sync.tar.gz

   tar xvf *.tar.gz
   ```

4. Resulting layout:
   ```
   ./dataset/SSCBenchKITTI360/instance_labels_v2/
   ├── 2013_05_28_drive_0000_sync
   ├── 2013_05_28_drive_0002_sync
   ├── 2013_05_28_drive_0003_sync
   ├── 2013_05_28_drive_0004_sync
   ├── 2013_05_28_drive_0005_sync
   ├── 2013_05_28_drive_0006_sync
   ├── 2013_05_28_drive_0007_sync
   ├── 2013_05_28_drive_0009_sync
   └── 2013_05_28_drive_0010_sync
   ```

---


