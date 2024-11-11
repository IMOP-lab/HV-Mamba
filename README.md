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

Table1: Summary of the breast tumor datasets utilized in this experimental protocol, encompassing total instance counts alongside distributions within the training, validation, and testing cohorts.

| Datasets | Quantity | Training Set | Validation Set | Testing Set |
|---------|---------|---------|---------|---------|
| BresdtDM | 29,510   | 20,432 | 1,989 |7,089
| I-SPY 1  | 6,801   | 6,120   |  -  | 681 |
| BCMedSet | 9,928 | 8,935| - | 993|

### Results

We provide GitHub links pointing to the PyTorch implementation code for all networks compared in this experiment here, so you can easily reproduce all these projects.

[U-Net](https://github.com/milesial/Pytorch-UNet); [R2U-Net](https://github.com/ncpaddle/R2UNet-paddle); [PAttUNet](https://github.com/faresbougourzi/PDAtt-Unet); [DAttUNet](https://github.com/faresbougourzi/PDAtt-Unet); [CENet](https://github.com/Guzaiwang/CE-Net); [DANet](https://github.com/junfu1115/DANet), [OCNet](https://github.com/openseg-group/OCNet.pytorch); [CGNet](https://github.com/wutianyiRosun/CGNet); [ENet](https://github.com/davidtvs/PyTorch-ENet), [LEDNet](https://github.com/sczhou/LEDNet), [SegNet](https://github.com/vinceecws/SegNet_PyTorch?tab=readme-ov-file); [DconnNet](https://github.com/Zyun-Y/DconnNet); [DeepLab](https://github.com/kazuto1011/deeplab-pytorch?tab=readme-ov-file); [ICNet](https://github.com/hszhao/ICNet)

Table 2: Comparison of Segmentation Performance Across Breast Cancer Medical Imaging Models on the DCE-MRI BreastDM Dataset

| Model          | 95HD↓            | DSC (%)↑           | G-mean (%)↑       | Kappa (%)↑         | MCC (%)↑           |
|----------------|------------------|--------------------|-------------------|--------------------|--------------------|
| U-Net          | 1.81 ± 0.81      | 74.65 ± 29.14      | 82.70 ± 29.06     | 74.60 ± 29.17      | 75.59 ± 28.62      |
| R2U-Net        | 1.81 ± 0.82      | 76.18 ± 27.84      | 82.70 ± 26.17     | 76.13 ± 27.88      | 77.12 ± 27.27      |
| R2AttU-Net     | 1.85 ± 0.71      | 74.48 ± 25.56      | 80.53 ± 24.30     | 74.43 ± 25.59      | 75.62 ± 24.98      |
| PAttUNet       | 1.88 ± 0.84      | 72.15 ± 31.01      | 80.76 ± 30.32     | 72.09 ± 31.05      | 73.01 ± 30.54      |
| DAttUNet       | 1.82 ± 0.81      | 74.62 ± 28.11      | 84.77 ± 28.38     | 74.57 ± 28.13      | 75.78 ± 27.47      |
| CENet          | 1.94 ± 0.77      | 72.85 ± 27.25      | 82.19 ± 27.24     | 72.79 ± 27.29      | 73.65 ± 26.87      |
| DANet          | 2.05 ± 0.64      | 69.95 ± 25.90      | 80.44 ± 26.17     | 69.89 ± 25.92      | 70.77 ± 25.47      |
| OCNet          | 1.99 ± 0.59      | 72.15 ± 22.62      | 84.20 ± 20.80     | 72.09 ± 22.65      | 72.96 ± 22.07      |
| CGNet          | 2.08 ± 0.60      | 66.25 ± 24.75      | 79.65 ± 24.13     | 66.18 ± 24.77      | 67.37 ± 24.10      |
| ENet           | 2.00 ± 0.70      | 68.77 ± 27.58      | 77.91 ± 27.59     | 68.72 ± 27.59      | 70.05 ± 26.86      |
| LEDNet         | 2.36 ± 0.64      | 51.14 ± 27.96      | 72.12 ± 35.99     | 51.05 ± 27.94      | 53.62 ± 28.10      |
| SegNet         | 1.87 ± 0.83      | 74.67 ± 27.52      | 82.10 ± 24.48     | 74.62 ± 27.58      | 75.55 ± 27.10      |
| DconnNet       | 1.91 ± 0.77      | 71.82 ± 31.42      | 79.36 ± 32.27     | 71.78 ± 31.43      | 72.65 ± 30.99      |
| DeepLab        | 1.98 ± 0.72      | 69.85 ± 27.26      | 82.97 ± 28.94     | 69.79 ± 27.28      | 71.06 ± 26.70      |
| ICNet          | 2.13 ± 0.71      | 64.03 ± 28.00      | 75.49 ± 29.36     | 63.97 ± 28.02      | 65.26 ± 27.56      |
| **HV-Mamba**   | **1.76 ± 0.80**  | **77.56 ± 26.77**  | **86.29 ± 26.15** | **77.52 ± 26.81**  | **78.34 ± 26.36**  |


Table 3: Comparison of Segmentation Performance Across Breast Cancer Medical Imaging Models on the DCE-MRI I-SPY 1 Dataset

| Model            | 95HD↓            | DSC (%)↑           | G-mean (%)↑       | Kappa (%)↑         | MCC (%)↑           |
|------------------|------------------|--------------------|-------------------|--------------------|--------------------|
| U-Net            | 2.64 ± 1.02      | 80.15 ± 22.16      | 88.03 ± 19.84     | 79.95 ± 22.10      | 80.70 ± 21.09      |
| R2U-Net          | 2.99 ± 1.15      | 67.26 ± 27.81      | 87.93 ± 20.96     | 66.86 ± 27.74      | 69.06 ± 25.60      |
| R2AttU-Net       | 2.88 ± 1.26      | 72.74 ± 24.36      | 86.07 ± 20.15     | 72.36 ± 24.39      | 73.85 ± 22.95      |
| PAttUNet         | 3.05 ± 1.18      | 64.43 ± 28.56      | 85.88 ± 22.76     | 63.96 ± 28.48      | 66.51 ± 26.38      |
| DAttUNet         | 3.01 ± 1.17      | 66.33 ± 28.54      | 85.39 ± 24.04     | 65.90 ± 28.47      | 68.04 ± 26.66      |
| CENet            | 2.57 ± 0.94      | 81.27 ± 21.23      | 89.53 ± 18.00     | 81.07 ± 21.18      | 81.75 ± 20.18      |
| DANet            | 2.57 ± 0.89      | 82.05 ± 20.00      | **90.86 ± 15.81** | 81.87 ± 19.97      | 82.57 ± 18.91      |
| OCNet            | 2.56 ± 0.90      | 82.60 ± 18.91      | 82.44 ± 12.79     | 82.42 ± 18.88      | 83.19 ± 17.50      |
| CGNet            | 2.76 ± 1.00      | 78.01 ± 22.34      | 87.06 ± 19.60     | 77.77 ± 22.28      | 78.45 ± 21.49      |
| ENet             | 2.84 ± 1.07      | 75.93 ± 23.34      | 87.61 ± 20.36     | 75.65 ± 23.28      | 76.57 ± 22.29      |
| LEDNet           | 2.93 ± 1.38      | 73.93 ± 25.80      | 85.52 ± 23.05     | 73.66 ± 75.81      | 74.88 ± 24.30      |
| SegNet           | 2.51 ± 0.97      | 82.81 ± 19.98      | 90.42 ± 17.17     | 82.75 ± 19.71      | 83.30 ± 18.91      |
| DconnNet         | 2.54 ± 0.89      | 80.15 ± 25.56      | 89.21 ± 23.66     | 79.97 ± 24.51      | 80.56 ± 23.98      |
| DeepLab          | 2.60 ± 0.90      | 78.45 ± 24.43      | 89.58 ± 23.57     | 78.36 ± 24.20      | 79.03 ± 23.72      |
| ICNet            | 2.78 ± 1.04      | 77.77 ± 21.66      | 86.14 ± 18.61     | 77.53 ± 21.61      | 78.33 ± 20.73      |
| **HV-Mamba**     | **2.44 ± 0.94**  | **83.11 ± 20.99**  | 89.58 ± 18.75     | **83.06 ± 20.73**  | **83.67 ± 19.83**  |

Table 4: Comparison of Segmentation Performance Across Breast Cancer Medical Imaging Models on the DCE-MRI BCMedSet Dataset
| Model          | 95HD↓           | DSC (%)↑           | G-mean (%)↑       | Kappa (%)↑         | MCC (%)↑          |
|----------------|------------------|---------------------|--------------------|---------------------|--------------------|
| U-Net          | 2.31 ± 1.17      | 71.00 ± 27.90      | 83.53 ± 26.67     | 73.10 ± 25.22      | 72.23 ± 27.10      |
| R2U-Net        | 2.34 ± 1.25      | 71.32 ± 30.33      | 78.51 ± 30.10     | 71.53 ± 29.99      | 72.27 ± 29.61      |
| R2AttU-Net     | 2.19 ± 1.18      | 75.09 ± 28.08      | 80.74 ± 27.57     | 76.97 ± 25.58      | 75.64 ± 27.69      |
| PAttUNet       | 2.32 ± 1.29      | 72.18 ± 30.93      | 80.87 ± 32.13     | 73.98 ± 28.95      | 73.03 ± 30.36      |
| DAttUNet       | 2.27 ± 1.10      | 72.97 ± 26.87      | 82.40 ± 26.11     | 73.57 ± 25.94      | 74.12 ± 26.01      |
| CENet          | 2.16 ± 1.07      | 77.44 ± 27.46      | 83.88 ± 26.97     | 79.40 ± 24.65      | 77.87 ± 26.87      |
| DANet          | 2.33 ± 1.06      | 73.76 ± 27.03      | 83.22 ± 27.24     | 74.39 ± 26.08      | 74.20 ± 26.59      |
| OCNet          | 2.39 ± 1.08      | 72.87 ± 26.12      | 81.84 ± 26.34     | 74.27 ± 24.09      | 73.20 ± 25.86      |
| CGNet          | 2.41 ± 1.07      | 73.01 ± 26.64      | 82.79 ± 26.42     | 74.02 ± 25.18      | 73.38 ± 26.32      |
| ENet           | 2.33 ± 1.09      | 72.01 ± 24.27      | 83.54 ± 24.39     | 72.59 ± 23.23      | 72.96 ± 23.72      |
| LEDNet         | 3.08 ± 1.09      | 40.77 ± 38.43      | 47.53 ± 44.35     | 42.40 ± 38.14      | 40.91 ± 38.50      |
| SegNet         | 2.31 ± 1.14      | 72.99 ± 28.46      | 83.47 ± 27.86     | 74.37 ± 26.68      | 73.78 ± 27.74      |
| DconnNet       | 2.16 ± 1.10      | 76.97 ± 28.09      | 83.23 ± 28.28     | 79.34 ± 24.80      | 77.42 ± 27.82      |
| DeepLab        | 2.30 ± 1.06      | 74.04 ± 27.09      | 83.65 ± 28.61     | 76.31 ± 23.91      | 74.57 ± 26.89      |
| ICNet          | 2.39 ± 1.11      | 71.98 ± 27.08      | 81.15 ± 27.54     | 72.97 ± 25.71      | 72.42 ± 26.74      |
| **HV-Mamba**   | **2.01 ± 1.17**  | **78.60 ± 29.43**  | **83.94 ± 28.37** | **78.89 ± 28.97**  | **79.28 ± 28.58**  |



<div align=left>
  <img src="https://github.com/IMOP-lab/HV-Mamba/blob/main/figures/comparison of models.png"width=50% height=50%>
</div>
<p align=left>
  Figure 2: A comparative analysis of segmentation outputs from HV-Mamba and fifteen contemporary breast cancer segmentation architectures reveals the segmentation efficacy of each model on the BreastDM dataset.
</p>

### Ablation study

#### Effect of Removing Module

Table 5: This table presents an ablation analysis of the HV-Mamba model's key modules on the BreastDM dataset. The presence of a module is indicated by a checkmark (✔), while an unmarked cell denotes its absence. Performance metrics include 95HD, DSC, G-Mean, Kappa, MCC, and IOU.

| HSS           | HCF           | seSK          | SpaC          | 95HD↓           | DSC (%)↑       | G-Mean (%)↑      | Kappa (%)↑     | MCC (%)↑       | IOU (%)↑       |
|---------------|---------------|---------------|---------------|-----------------|----------------|------------------|----------------|----------------|----------------|
|               | ✔             | ✔             | ✔             | 1.81            | 75.23          | 80.71            | 75.19          | 76.09          | 67.21          |
| ✔             |               | ✔             | ✔             | 1.80            | 75.98          | 83.48            | 75.93          | 76.72          | 68.40          |
| ✔             | ✔             |               | ✔             | 1.79            | 77.06          | 85.52            | 77.01          | 77.84          | 68.55          |
| ✔             | ✔             | ✔             |               | 1.81            | 74.28          | 85.19            | 74.22          | 75.42          | 64.63          |
|               |               |               | ✔             | 1.81            | 74.44          | 81.91            | 74.40          | 75.28          | 66.55          |
| ✔             | ✔             | ✔             | ✔             | **1.76**        | **77.56**      | **86.29**        | **77.52**      | **78.34**      | **69.18**      |


#### Integration of seSK Unit Across Multiple Architectures

Table 6: This table assesses the integration efficacy of the seSK unit within different network architectures for breast cancer DCE-MRI segmentation. Metrics include 95HD, DSC, G-Mean, Kappa, and MCC, with improvements highlighted in blue to indicate enhanced segmentation precision.

| Architecture           | 95HD↓                  | DSC (%)↑               | G-Mean (%)↑            | Kappa (%)↑            | MCC (%)↑              |
|------------------------|------------------------|------------------------|------------------------|-----------------------|-----------------------|
| U-Net                  | 1.81 ± 0.81           | 74.65 ± 29.14          | 82.70 ± 29.06          | 74.60 ± 29.17         | 75.59 ± 28.62         |
| + seSK unit            | **1.75 ± 0.79** (↓0.06) | **77.93 ± 27.29** (↑3.28) | **86.55 ± 27.15** (↑3.85) | **77.88 ± 27.34** (↑3.28) | **78.68 ± 26.88** (↑3.09) |
| R2U-Net                | 1.81 ± 0.82           | 76.18 ± 27.84          | 82.70 ± 26.17          | 76.13 ± 27.88         | 77.12 ± 27.27         |
| + seSK unit            | **1.75 ± 0.82** (↓0.06) | **78.24 ± 25.23** (↑2.06) | **85.73 ± 23.68** (↑3.03) | **78.19 ± 25.28** (↑2.06) | **79.05 ± 24.70** (↑1.93) |
| PAttUNet               | 1.88 ± 0.84           | 72.15 ± 31.01          | 80.76 ± 30.32          | 72.09 ± 31.05         | 73.01 ± 30.54         |
| + seSK unit            | **1.80 ± 0.88** (↓0.08) | **76.90 ± 27.07** (↑4.75) | **84.76 ± 25.73** (↑4.00) | **76.85 ± 27.11** (↑4.76) | **77.87 ± 26.38** (↑4.86) |
| DAttUNet               | 1.82 ± 0.81           | 74.62 ± 28.11          | **84.77 ± 28.38**      | 74.57 ± 28.13         | 75.78 ± 27.47         |
| + seSK unit            | **1.72 ± 0.69** (↓0.10) | **76.76 ± 24.83** (↑2.14) | 83.99 ± 24.77 (↓0.78) | **76.71 ± 24.88** (↑2.14) | **78.26 ± 23.50** (↑2.48) |
| ENet                   | 2.00 ± 0.70           | 68.77 ± 27.58          | 77.91 ± 27.59          | 68.72 ± 27.59         | 70.05 ± 26.86         |
| + seSK unit            | **1.97 ± 0.66** (↓0.03) | **70.12 ± 25.86** (↑1.35) | **79.37 ± 25.72** (↑1.46) | **70.07 ± 25.88** (↑1.35) | **71.33 ± 25.19** (↑1.28) |
| SegNet                 | 1.87 ± 0.83           | 74.67 ± 27.52          | 82.10 ± 24.48          | 74.62 ± 27.58         | 75.55 ± 27.10         |
| + seSK unit            | **1.82 ± 0.80** (↓0.05) | **77.16 ± 25.48** (↑2.49) | **85.39 ± 23.99** (↑3.27) | **77.11 ± 25.53** (↑2.49) | **77.97 ± 24.97** (↑2.42) |


## Question

If you have any qusetion, please contact 'gaopeng.huang@hdu.edu.cn'

