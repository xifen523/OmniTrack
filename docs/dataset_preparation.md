# Dataset Preparation

## 1. prepare for JRDB2019 dataset

### 1.1 download JRDB2019 dataset

Download the JRDB2019 dataset from the official website: [https://www.jrdb.com/dataset/download/](https://jrdb.erc.monash.edu/#downloads)

### 1.2 extract JRDB2019 dataset

Extract the JRDB2019 dataset to the following directory:

```bash 
# .../OmniTrack/data/JRDB2019
JRDB2019
├── test_dataset_without_labels
│   ├── calibration
│   ├── detections
│   ├── images
│   ├── pointclouds
│   └── timestamps
│   ...
├── train_dataset_with_activity
│   ├── calibration
│   ├── detections
│   ├── images
│   ├── labels
│   ├── pointclouds
│   └── timestamps
│   ...
```

### 1.3 generate pkl for JRDB2019 dataset
```bash
cd tools/
python JRDB2019_2d_stitched_converter.py

# generate k-means anchors
python anchor_2d_generator.py --ann_file ../data/JRDB2019_2d_stitched_anno_pkls/JRDB_infos_train_v1.2.pkl

cd ..
```
### 1.4 Prepare pre-trained weights 
Download the pre-trained weights from the following link: https://download.pytorch.org/models/resnet50-19c8e357.pth
Put the pre-trained weights in the following directory:
```bash
mkdir -p ckpt
wget https://download.pytorch.org/models/resnet50-19c8e357.pth -O ckpt/resnet50-19c8e357.pth
```


# Commence training and testing
```bash
# train
bash local_train.sh JRDB_OmniTrack

# test
bash local_test.sh JRDB_OmniTrack  path/to/checkpoint
```

## 2. prepare for QuadTrack dataset
    coming soon
