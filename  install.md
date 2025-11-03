# Step-by-step installation instructions

IPFormer environment setup is based on [CGFormer](https://github.com/pkqbajng/CGFormer) and [OccFormer](https://github.com/zhangyp15/OccFormer).

**a. Create a conda virtual environment and activate**

python 3.8 may not be supported.

```shell
conda create -n ipformer python=3.7 -y
conda activate ipformer

```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/get-started/previous-versions/)**


```shell
pip install --no-cache-dir torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1 \
  -f https://download.pytorch.org/whl/cu113/torch_stable.html

```

We select this pytorch version because mmdet3d 0.17.1 does not support pytorch >= 1.11 and our cuda version is 11.3.


**c. Install CUDA compiler for building extensions (nvcc 11.3)**

```shell
conda install -y -c conda-forge cudatoolkit-dev=11.3

```
**d. Install system toolchain**

```shell
apt-get update
apt-get install -y build-essential cmake ninja-build libc6-dev libcrypt-dev
```

**e. Install OpenMMLab core**

```shell
pip install -U openmim
pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10/index.html
mim install mmdet==2.14.0
mim install mmsegmentation==0.14.1
```

**f. Build mmdet3d ops (OccFormer custom ops)**

Compared with the offical version, the mmdetection3d provided by [OccFormer](https://github.com/zhangyp15/OccFormer) further includes operations like bev-pooling, voxel pooling. 

```shell
cd packages
export CC=/usr/bin/gcc
export CXX=/usr/bin/g++
export CUDAHOSTCXX=/usr/bin/g++
export TORCH_CUDA_ARCH_LIST="8.6+PTX;8.0;7.5;7.0"
bash setup.sh
cd ..

```

**GPU note:** The default `TORCH_CUDA_ARCH_LIST="8.6+PTX;8.0;7.5;7.0"` builds native code for Ampere/Turing/Volta and includes PTX for forward compatibility (e.g., Ada).  
For Pascal add `;6.1`. If you hit “Unknown CUDA arch”, remove hardcoded `sm_89/sm_90` from any `setup.py` and re-run the build.


**g. Install Python requirements**

```shell
pip install yapf==0.40.0
pip install natten==0.14.6+torch1101cu113 -f https://shi-labs.com/natten/wheels
pip install -r requirements.txt
```

