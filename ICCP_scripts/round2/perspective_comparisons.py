from filmscope.recon_util import get_sample_information 
from filmscope.reconstruction import generate_config_dict
from filmscope.util import load_image_set
from filmscope.config import path_to_data
from filmscope.datasets import FSDataset
from matplotlib import pyplot as plt 

import numpy as np

config_dict = generate_config_dict("0", "stamp_02_08",
                                   custom_crop_info={"crop_size": (0.15, 0.2)})
sample_info = config_dict["sample_info"]

dataset = FSDataset(
    path_to_data + sample_info["image_filename"], 
    path_to_data + sample_info["calibration_filename"], 
    sample_info["image_numbers"], 
    sample_info["downsample"], 
    sample_info["crop_values"], 
    frame_number=-1, 
    ref_crop_center=sample_info["ref_crop_center"], 
    crop_size=sample_info["crop_size"], 
    height_est=sample_info["height_est"],
    blank_filename=None, 
    noise=[0, 0]
)


images = dataset.images.squeeze() 
images_c = images[:, 100:300, 100:300]