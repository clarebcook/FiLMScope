## FiLM-Scope 

This repository contains the code for our paper, "Fourier Lightfield Multiview Stereoscope for Large Field-of-View 3D Imaging in Microsurgical Settings", published in Advanced Photonics Nexus: https://doi.org/10.1117/1.APN.4.4.046008. The code here demonstrates how to calibrate the FiLM-Scope system and generate 3D height maps and videos from our data. 

## Hardware and System Requirements
3D Reconstruction was performed using Nvidia RTX A5000 and A6000 GPUs, with 24 and 48 GB RAM respectively, and CUDA 12.4. We anticipate the code will run on alternate Nvidia GPUs, with minor modifications, but this has not been tested.

Calibration steps do not require GPU acceleration

## Downloading Data 
All data needed to run the example scripts is published in Duke University's digital repository: https://doi.org/10.7924/r4hd8386m. The sub-folders in this repository contain README's with information on which data needs to be downloaded for each script. 

## Environment and Installation 

We recommend running this code using a conda environment. The repository and environment can be set up using the following steps. 

1. Clone repository

```
git clone git@github.com:clarebcook/FiLMScope.git 
cd FiLMScope
```

2. Set up the environment
```
conda env create -f environment.yml
conda activate filmscope
conda develop .
```

Note that using `conda develop` requires install `conda-build`. This can be done with: 
```
conda install conda-build
```

3. Set variables in the config file 

Navigate to `filmscope/config.py`. Change `path_to_data` to the location where the downloaded data is stored. To log runs with Neptune.ai, `neptune_project` and `neptune_api_token` must be filled in. However, that is not needed to run the example scripts. 

## Usage 
The example scripts are split into three folders, each of which contain READMEs with additional information. 

1. Calibration: Scripts to run the calibration procedure with an example dataset. 

2. Reconstruction: Scripts to run 3D reconstruction on three example datasets, including a video. 

3. Visualization: MATLAB script for visualizing reconstructed data as a 3D mesh. 
