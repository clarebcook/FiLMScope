import os 
from filmscope.recon_util import (get_sample_information,
                                  get_ss_volume_from_dataset,
                                  get_height_aware_vol_from_dataset) 
from filmscope.datasets import FSDataset
from filmscope.config import path_to_data
from filmscope.util import play_video
import numpy as np 
import torch 
from matplotlib import pyplot as plt 
import torch.nn.functional as F

gpu_number = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
sample_name = "finger"

sample_info = get_sample_information(sample_name)
image_numbers = torch.arange(48).tolist() 
#image_numbers = [1, 5, 10, 20, 40]
image_filename = sample_info["image_filename"]
calibration_filename = sample_info["calibration_filename"]
crop_values = sample_info["crop_values"]
frame_number = -1 
bf = None 


patch = [1200, 2160 + 32 * 10, 400, 1360 + 32 * 10]
crop_values = [patch[0] / 4096, patch[1]/4096, patch[2] / 3072, patch[3]/3072] 

noise = [0, 0]


# new dataset with the crops
downsample = 1
dataset = FSDataset(
    path_to_data + image_filename,
    path_to_data + calibration_filename,
    image_numbers,
    downsample,
    crop_values,
    enforce_divisible=-1,
    frame_number=frame_number,
    blank_filename=bf,
    noise=noise
)


use_patch = False 
if use_patch:
    low_res_downsample = 4
    low_res_depth = torch.load(path_to_data + '/finger/low_res_height_map.pt',
                                        weights_only=False)
    depth_patch = low_res_depth[
                    int(patch[0]/low_res_downsample):int(patch[1]/low_res_downsample),
                    int(patch[2]/low_res_downsample):int(patch[3]/low_res_downsample)]
    depth_patch = F.interpolate(
        depth_patch[None, None], size=(patch[1] - patch[0], patch[3] - patch[2])).squeeze(0) 
    depth_patch = depth_patch.cuda()


    depth_range = (-1, 1)
    low_plane = depth_range[0] 
    high_plane = depth_range[1]
    num = 30
    depths = np.linspace(low_plane, high_plane, num, endpoint=True)
    depth_values = torch.from_numpy(depths).to(torch.float32).cuda()

    batch_size = 10
    volume = get_height_aware_vol_from_dataset(
        dataset, batch_size, depth_values, depth_patch, get_squared=False)
    volume = volume.squeeze().cpu().numpy()


depth_range = (-4, 1)
low_plane = depth_range[0] 
high_plane = depth_range[1]
num = 30
batch_size = 10
depths = np.linspace(low_plane, high_plane, num, endpoint=True)

depths = np.asarray([-1.58 - 0.025, -1.58, -1.58 + 0.025])
depth_values = torch.from_numpy(depths).to(torch.float32).cuda()
volume = get_ss_volume_from_dataset(
    dataset, batch_size, depth_values, get_squared=False)
volume = volume.squeeze().cpu().numpy()


plt.figure()
plt.imshow(volume[14][200:800, :600], cmap='gray')
plt.show()



plt.figure()
plt.imshow(volume[14], cmap='gray')
plt.show()