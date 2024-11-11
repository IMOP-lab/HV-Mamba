# HV-Mamba: Harmonic Vision Mamba

### [Project page](https://github.com/IMOP-lab/HV-Mamba) 

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
### Baselines
