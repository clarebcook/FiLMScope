## FiLM-Scope 

This repository contains data for our new paper, "Fourier Lightfield Multiview Stereoscope for Large Field-of-View 3D Imaging in Microsurgical Settings", currently under review. More details on the project will be added here after acceptance. 

## Hardware and System Requirements
3D Reconstruction was performed using Nvidia RTX A5000 and A6000 GPUs, with 24 and 48 GB RAM respectively, and CUDA 12.4. We anticipate the code will run on alternate Nvidia GPUs, with minor modifications, but this has not been tested.

Calibration steps do not require GPU acceleration

## Downloading Data 
Data needed to run the example scripts in this repository are temoprarily available on Google Drive (https://drive.google.com/drive/folders/1q-tyXVybuK36g5OZaqQBDijxQNmk1Nps?usp=sharing), and a larger body of data will be made available at a permanent DOI soon. 

The full folder can be downloaded using gdown (https://anaconda.org/conda-forge/gdown), by running the following Python snippet from the location where data will be stored. 

```
import gdown

file_id = "1q-tyXVybuK36g5OZaqQBDijxQNmk1Nps"
url = f"http://drive.google.com/drive/u/1/folders/{file_id}"
gdown.download_folder(url, quiet=True)
```

Alternatively, individual example scripts provide specific information on which sub-directories and files are needed. 

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
3. Set variables in the config file 

Navigate to `filmscope/config.py`. Change `path_to_data` to the location where the downloaded data is stored. To log runs with Neptune.ai, `neptune_project` and `neptune_api_token` must be filled in. However, that is not needed to run the example scripts. 

## Usage 
The example scripts are split into three folders, each of which contain READMEs with additional information. 

1. Calibration: Scripts to run the calibration procedure with an example dataset. 

2. Reconstruction: Scripts to run 3D reconstruction on three example datasets, including a video. 

3. Visualization: MATLAB script for visualizing reconstructed data as a 3D mesh. 