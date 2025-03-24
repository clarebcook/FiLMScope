from filmscope.config import path_to_data 
from filmscope.calibration import SystemCalibrator, CalibrationInfoManager
from filmscope.util import load_dictionary, save_dictionary
from filmscope.recon_util import get_sample_information, add_sample_entry
from filmscope.datasets import FSDataset
from filmscope.reconstruction import generate_config_dict
import torch 
from matplotlib import pyplot as plt 
from tqdm import tqdm 

load_filename = path_to_data + '/stamp_2024_02_08/calibration_information_v2'
load_filename = path_to_data + '/stamp_06_21/calibration_information'

config_dict = generate_config_dict("0", "stamp_06_21") #"stamp_02_08")

# find crop centers from the original dataset 
info = config_dict["sample_info"]
dataset = FSDataset(
    path_to_data + info["image_filename"], 
    path_to_data + info["calibration_filename"], 
    image_numbers=torch.arange(48).tolist(), 
    downsample=8, 
    ref_crop_center=info["ref_crop_center"], 
    crop_size=info["crop_size"],
    height_est=info["height_est"]
    #crop_values=sample_info["crop_values"]
)

cm = CalibrationInfoManager(path_to_data + info["calibration_filename"])
image_shape = cm.image_shape
image_shape = [image_shape[0] / 8, image_shape[1] / 8]
crop_centers = {} 
for num, item in dataset.full_crops.items():
    center = ((item[1] + item[0]) / 2, (item[2] + item[3]) / 2)
    center = (center[0] / image_shape[0], center[1] / image_shape[1])
    crop_centers[num] = center

crop_size = (0.15, 0.2)


for cam_number in tqdm(range(48)):
#cam_number = 20
#if True:
    save_filename = load_filename + f'_cam_{cam_number}'

    original = load_dictionary(load_filename) 
    save_dictionary(original, save_filename)

    reference_plane = CalibrationInfoManager(load_filename).reference_plane


    plane_separation_mm = 1 
    calibrator = SystemCalibrator(
        calibration_filename=save_filename, 
        reference_plane=reference_plane,
        reference_camera=cam_number,
        plane_separation_mm = plane_separation_mm,
        ref_plane_image_folder = None, #image_folder,
        useable_plane_numbers = None #np.arange(5) # if None, this will use all planes
    )

    calibrator.run_inter_camera_calibration(show=False, order=2)
    calibrator.run_slope_calibration(show=False, order=2) 


    sample_name = f"stamp_06_21_camera_{cam_number}"
    folder = '/stamp_06_21'
    image_filename = folder + '/stamp_20240621_134344_972.nc'
    calibration_filename = folder + f'/calibration_information_cam_{cam_number}'
    blank_filename = None 
    crop_center = crop_centers[num] 
    crop_values = [crop_center[0] - crop_size[0] / 2, crop_center[0] + crop_size[0] / 2, 
                crop_center[1] - crop_size[1] / 2, crop_center[1] + crop_size[1] / 2]



    info = {
        "folder": folder, 
        "sample_name": sample_name, 
        "calibration_filename": calibration_filename, 
        "image_filename": image_filename, 
        "crop_values": crop_values, 
        "depth_range": info["depth_range"],
        "blank_filename": None, 
    }

    add_sample_entry(sample_name, info, overwrite=True)