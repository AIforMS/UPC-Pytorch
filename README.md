# UPC: Uncertainty-aware Pseudo-label and Consistency for Semi-supervised Medical Image Segmentationh

by [Liyun Lu](https://github.com/liyun-lu), Mengxiao Yin, Liyao Fu, Feng Yang.

## Introduction
This repository is the Pytorch implementation of "Uncertainty-aware Pseudo-label and Consistency for Semi-supervised Medical Image Segmentationh"

Our paper and code will be release soon.

## Requirements
We implemented our experiment on the super parallel computer system of Guangxi University. The specific configuration is as follows:
* Centos 7.4
* NVIDIA Tesla V100 32G
* Intel Xeon gold 6230 2.1G 20C processor

Some important required packages include:
* CUDA 10.1
* Pytorch == 1.6.0
* Python == 3.8 
* Some basic python packages such as Numpy, Scikit-image, SimpleITK, Scipy ......

# Usage

1. Clone the repo:
```
git clone https://github.com/GXU-GMU-MICCAI/UPC-Pytorch.git 
cd UPC-Pytorch
```

2. Download the Left Atrium dataset in [Google drive](https://drive.google.com/file/d/1CKEtfOGRQhjySYf4MnTgrdEOcuYbBC2t/view?usp=sharing).
Put the data in './data/'  folder
```
cd code/dataloaders
python la_heart_processing.py
```

3. Train the model
```
cd code
python train_la_upc.py
```

4. Test the model
```
python test_LA.py
```

## Citation

## Acknowledgement
Part of the code is revised from the [UA-MT](https://github.com/yulequan/UA-MT).

We thank Dr. Lequan Yu for their elegant and efficient code base.

## Note
* The repository is being updated.
* Contact: Liyun Lu (luly1061@163.com)
