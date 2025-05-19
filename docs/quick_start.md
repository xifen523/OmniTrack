# Installation guide

## 1. Clone OmniTrack to local  device
```bash
git clone https://github.com/xifen523/OmniTrack.git
cd OmniTrack
```

## 2. Set up a new virtual environment
```bash
conda create -n OmniTrack python=3.10.15 -y
conda activate OmniTrack
```

## 3. Install torch（GPU version），（Note: other versions can be installed [here](https://pytorch.org/), but not verified）
```bash
conda install pytorch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 pytorch-cuda=11.8 -c pytorch -c nvidia -y
```

## 4. Install other dependencies
```bash
pip install -r requirements.txt
```

## 5. Install causal_conv1d-1.1.1 and mamba-ssm-1.2.0 for SSM
```bash
pip install causal_conv1d-1.1.1 mamba-ssm-1.2.0
```
*if you don\'t use pip to install these wheels, you can download them from [causal_conv1d](https://pan.baidu.com/s/1gGt9navtB5BWnDhy8FLcMg?pwd=xsif ),[mamba-ssm](https://pan.baidu.com/s/1-To0SDHCxD_8ADi-Ofiv2g?pwd=ak6n) and install them manually.*


## 6. Compile some modules
```bash
cd mmcv-full-1.7.1

# for mmcv-full-1.7.1 GPU version with pip == 24.2
MMCV_WITH_OPS=1 pip install -e .
cd ..

# Compile the deformable_aggregation CUDA op
cd projects/mmdet3d_plugin/ops
python3 setup.py develop
cd ../../../

# Compile the jrdb_toolkit nms
cd jrdb_toolkit/detection_eval
python3 setup.py develop
cd ../../
###
```


## 7. Prepare the dataset
[Dataset preparation](dataset_preparation.md)

## 8. Generate anchors by K-means

```bash
python3 tools/anchor_2d_generator.py --ann_file ${pkl_path}/nuscenes_infos_train.pkl
```

## 9. Download pre-trained weights
```bash
mkdir ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```

## 10. Commence training and testing
```bash
# train
bash local_train.sh JRDB_OmniTrack

# test
bash local_test.sh JRDB_OmniTrack path/to/checkpoint
```
