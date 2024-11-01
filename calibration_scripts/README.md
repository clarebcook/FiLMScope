This folder contains scripts needed to calibrate the FiLM-Scope. More information on the calibration procedure will be available in our paper and described here after acceptance. 

The installation steps in the main README file should be completed prior to using these scripts. 

## Downloading data

Data needed to run these scripts can currently be downloaded from our Google Drive folder (https://drive.google.com/drive/folders/1q-tyXVybuK36g5OZaqQBDijxQNmk1Nps?usp=sharing). The "calibration_data" sub-folder should be downloaded and organized as:

```
path_to_data/calibration_data
```
using the `path_to_data` variable set in the `config.py` file. 

The folder contains a single set of FiLM-Scope calibration images, as well as an example output from the calibration procedure. 

## Performing calibration

Calibration is completed by following the prompts in `run_calibration.ipynb`. Two steps in the process require the user to manually select points, which is done in a GUI by running `remove_vertices.py` and `select_alignment_points.py`. The notebook will indicate at what point this needs to be completed, and allow the user to use example data to skip those steps. 

## Output

The end result of this process is a single file, ``calibration_information``, which by default will be stored in the same ``calibration_data`` folder. This is the information needed to proceed with 3D reconstruction. 