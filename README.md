# 3D neuMatrix

## Overview

This repository contains Python scripts for analyzing calcium signal data from 3D neuMatrix experiments. The analysis includes preprocessing image sequences, extracting fluorescence intensity, calculating ΔF/F0, and analyzing neural firing rates and correlations.

## Experimental Background

The 3D neuMatrix was washed with HBSS and incubated with Fluo-4 AM (Thermo Fisher) for 30 minutes at 37°C. It was then transferred to a 35 mm confocal dish with preheated complete neurobasal medium and observed under a laser scanning confocal microscope (A1R; Nikon) in a stage-top incubator (Tokai Hit). Five-minute time-lapse sequences were captured, with a maximum of 3 levels on the z-axis to reflect 3D neural signals. For drug treatment, image sequences were taken as controls before neurotransmitter receptor antagonists were added to the medium. After 5 minutes of reaction and stabilization, image sequences were captured for the treatment group.

## Data Preprocessing

Image sequences were preprocessed using [FluoroSNNAP](https://github.com/tapan-patel/FluoroSNNAP) for large-scale imaging and [Suite2P](https://github.com/MouseLand/Suite2P) for small-scale imaging. Putative neural clusters were selected by thresholding the average image, and fluorescence intensity was extracted for each cluster. The ΔF/F0 calculation and further analysis were performed using custom Python scripts. F0 was defined as the mean fluorescence intensity of the 10 consecutive frames with the lowest intensity. Frames with ΔF/F0 greater than the mean ΔF/F0 plus 3 standard deviations were defined as frames with neural firing. The firing rate for each neural cluster was calculated as the percentage of frames with neural firing.

## Correlation Analysis

Pearson’s correlation coefficients of ΔF/F0 were computed to analyze the correlation between neural clusters. Clusters with no firing across all frames were excluded to avoid false high correlations.

## Code Features

- **Data Loading and Initialization**: Load data from MATLAB files and initialize parameters.
- **Image and Signal Processing**: Plot masks, centers, and links; compute and plot correlation matrices and firing curves.
- **Signal Analysis**: Calculate ΔF/F0 and correlation coefficients to analyze neural cluster firing rates and correlations.

## Usage

[Example Notebook](example_notebook.ipynb)

## Requirements

- Python 3.x
- Required Python packages: `scipy`, `pandas`, `numpy`, `matplotlib`, `seaborn`, `PIL`

## Installation

Clone the repository and install the required packages:

```bash
git clone https://github.com/yourusername/neuMatrix.git
cd neuMatrix
pip install -r requirements.txt
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code, please cite the following paper:

```
@article{https://doi.org/10.1002/advs.202504455,
author = {Yang, Huiyu and Zhang, Jiangang and Li, Yiran and Zhong, Zihan and Li, Wenhua and Luo, Haojun and Liu, Yanyong and Ouyang, Liujian and Jiang, Zhuoran and Sun, Yuning and Sun, Hang and Liu, Lulu and Yang, Huayu and Wang, Yu and Yang, Nan and Ma, Wenbin and Mao, Yilei},
title = {Multiscale Organization of Neural Networks in a 3D Bioprinted Matrix},
journal = {Advanced Science},
pages = {e04455},
doi = {https://doi.org/10.1002/advs.202504455},
```
