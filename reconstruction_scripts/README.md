This folder contains scripts to run 3D reconstruction on FiLM-Scope data. More information on the algorithm used is detailed in our paper: https://doi.org/10.7924/r4hd8386m. Each script in this folder requires a different dataset, all of which are stored in Duke's digital repository: https://doi.org/10.7924/r4hd8386m.


The installation steps in the main README file should be completed prior to using these scripts. 

## Preparing Dataset

Prior to performing reconstruction, the FiLM-Scope must be calibrated and information on the specific dataset must be saved. For datasets used in these scripts, these steps have already been completed. The `calibration_scripts` folder contains scripts for FiLM-Scope calibration, and ``save_new_sample.ipynb`` in this folder demonstrates how to save information on a newly acquired dataset. To run this notebook, first download and unzip `finger.zip` from the repository, and organize it as 

```
path_to_data/finger
```
using the `path_to_data` variable set in the `config.py` file.

## Reconstruction scripts

Each of the remaining scripts demonstrates a different way to perform 3D reconstruction with this repository. ``gpu_number`` is specified in each of the scripts and should be edited before running.  

1. ``frame_reconstruction.py``

This script demonstrates how to perform reconstruction on a single image frame. Settings can be edited in the script, including whether to perform logging with Neptune.ai or locally. This script requires ``skull_with_tool`` and `sample_info.json` to be downloaded, unzipped, and arranged as

```
path_to_data/skull_with_tool
path_to_data/sample_info.json
```
This folder contains an `.nc` file with the images, as well as the needed calibration information for the FiLM-Scope. ``sample_info.json`` contains necessary information about the sample, previously saved using ``save_new_sample.ipynb``. 

2. ``reconstruction_with_guide_map.py``

This script demonstrates how to perform full resolution reconstruction on a patch of an image, by using a previously reconstructed low resolution height map as a guide. This script requires ``finger`` and ``sample_info.json`` to be downloaded and arranged as 

```
path_to_data/finger
path_to_data/sample_info.json
```
The folder contains an `.nc` file with the images of the finger, the calibration information for the FiLM-Scope, and a low resolution reconstruction of the finger data. 


3. ``video_reconstruction.py``

This script demonstrates how to perform reconstruction on multiple frames of a video. This script requires ``knuckle_video`` and ``sample_info.json`` to be downloaded and arranged as 

```
path_to_data/knuckle_video
path_to_data/sample_info.json
```

The folder contains an `.nc` file with the video, an `.nc` file with the illumination correction images, and the calibration information for the FiLM-Scope.
