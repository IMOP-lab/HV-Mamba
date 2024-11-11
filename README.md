# HV-Mamba: Harmonic Vision Mamba

### [Project page](https://github.com/IMOP-lab/HV-Mamba) | [Our laboratory home page](https://github.com/IMOP-lab) 

by Xingru Huang, Han Yang, Gaopeng Huang, Zhao Huang, Hui Guo, Lou Zhao, Huiyu Zhou, Guan Gui, Yun Lin, Minhong Sun, Jin Liu, Zhiwen Zheng, Xiaoshuai Zhang

Hangzhou Dianzi University IMOP-lab

<div align=left>
  <img src="https://github.com/IMOP-lab/HV-Mamba/blob/main/figures/HV-Mamba.png">
</div>
<p align=left>
  Figure 1: Detailed network structure of the HV-Mamba.
</p>

The structural overview of the HV-Mamba framework. The left section, designated as the HSS Module, utilizes the FFT to transmute input imagery into the frequency domain, subsequently isolating real and imaginary components. These components are processed through the VSS Block to encapsulate long-range dependencies. Following transformation, the iFFT restores the processed frequency-domain information to the spatial domain. The right section consists of the Full-Spectrum Recursive Residual Encoder, comprising multiple residual blocks and HCF Attention, optimized for feature extraction and detail enhancement. Each layer integrates the SpaC Module to refine feature expression, thereby facilitating high-precision segmentation of small targets within breast cancer imagery. 

## Installation
The experimental environment consisted of two workstations, each equipped with dual NVIDIA RTX 3080 GPUs and 128GB RAM. Experiments were conducted using Python 3.9, PyTorch 2.0.0, and CUDA 11.8. An initial learning rate of 0.0001 was applied across all models to ensure consistency in training dynamics.

## Experiment
### Dataset
<p align=left>
  Table 1: Summary of the breast tumor datasets utilized in this experimental protocol, encompassing total instance counts alongside distributions within the training, validation, and testing cohorts.
</p>
| Datasets | Quantity | Training Set | Validation Set | Testing Set |
|---------|---------|---------|---------|---------|
| BresdtDM | 29,510   | 20,432 | 1,989 |7,089
| I-SPY 1  | 6,801   | 6,120   |  -  | 681 |
| BCMedSet | 9,928 | 8,935| - | 993|

### Baselines

We provide GitHub links pointing to the PyTorch implementation code for all networks compared in this experiment here, so you can easily reproduce all these projects.

[U-Net](https://github.com/milesial/Pytorch-UNet); [R2U-Net](https://github.com/ncpaddle/R2UNet-paddle); [PAttUNet](https://github.com/faresbougourzi/PDAtt-Unet); [DAttUNet](https://github.com/faresbougourzi/PDAtt-Unet); [CENet](https://github.com/Guzaiwang/CE-Net); [DANet](https://github.com/junfu1115/DANet), [OCNet](https://github.com/openseg-group/OCNet.pytorch); [CGNet](https://github.com/wutianyiRosun/CGNet); [ENet](https://github.com/davidtvs/PyTorch-ENet), [LEDNet](https://github.com/sczhou/LEDNet), [SegNet](https://github.com/vinceecws/SegNet_PyTorch?tab=readme-ov-file); [DconnNet](https://github.com/Zyun-Y/DconnNet); [DeepLab](https://github.com/kazuto1011/deeplab-pytorch?tab=readme-ov-file); [ICNet](https://github.com/hszhao/ICNet)



<div align=left>
  <img src="https://github.com/IMOP-lab/HV-Mamba/blob/main/figures/comparison of models.png"width=50% height=50%>
</div>
<p align=left>
  Figure 2: A comparative analysis of segmentation outputs from HV-Mamba and fifteen contemporary breast cancer segmentation architectures reveals the segmentation efficacy of each model on the BreastDM dataset.
</p>

