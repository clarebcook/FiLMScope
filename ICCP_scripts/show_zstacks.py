import os 
from filmscope.recon_util import (get_sample_information,
                                  get_ss_volume_from_dataset,
                                  get_height_aware_vol_from_dataset) 
from filmscope.datasets import FSDataset
from filmscope.config import path_to_data
from filmscope.reconstruction import generate_config_dict
from filmscope.util import play_video
import numpy as np 
import torch 
from matplotlib import pyplot as plt 
import torch.nn.functional as F

import cv2 

# # showing tool portion of skull
# gpu_number = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
# sample_name = "skull_with_tool"

# config_dict = generate_config_dict(gpu_number, sample_name)
# sample_info = config_dict["sample_info"]

# # sample_info = get_sample_information(sample_name)
# image_numbers = [13, 15, 20, 25, 27]
# #image_numbers = [6, 10, 20, 36, 40]
# image_numbers = torch.arange(48).tolist() 
# image_filename = sample_info["image_filename"]
# calibration_filename = sample_info["calibration_filename"]
# # #crop_values = sample_info["crop_values"]
# frame_number = -1 
# bf = None 

# # new dataset with the crops
# downsample = 2
# dataset = FSDataset(
#     path_to_data + image_filename,
#     path_to_data + calibration_filename,
#     image_numbers,
#     downsample,
#     #crop_values,
#     ref_crop_center=(0.45, 0.3), #sample_info["ref_crop_center"], 
#     crop_size=(0.4, 0.6), #sample_info["crop_size"], 
#     enforce_divisible=-1,
#     frame_number=frame_number,
#     blank_filename=bf,
# )

# plt.figure()
# plt.imshow(dataset.images[0].squeeze(), cmap='gray')
# plt.show()


# depth_range = (-10, 5)
# low_plane = depth_range[0] 
# high_plane = depth_range[1]
# num = 30
# depths = np.linspace(low_plane, high_plane, num, endpoint=True)
# depth_values = torch.from_numpy(depths).to(torch.float32).cuda()

# batch_size = 10
# volume = get_ss_volume_from_dataset(
#     dataset, batch_size, depth_values, get_squared=False)
# volume = volume.squeeze().cpu().numpy()


# for n in [15, 20, 25, 29]:
#     plt.figure() 
#     plt.imshow(volume[n][50:-50, 50:-50], cmap='gray') 
#     plt.title(n) 


# play_video(volume)








# # knuckle 
# gpu_number = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
# sample_name = "knuckle_video"

# config_dict = generate_config_dict(gpu_number, sample_name)

# sample_info = config_dict["sample_info"]
# image_numbers = torch.arange(48).tolist() 
# #image_numbers = [13, 15, 20, 25,27]
# #image_numbers = [6, 10, 20, 36, 40]
# image_filename = sample_info["image_filename"]
# calibration_filename = sample_info["calibration_filename"]
# crop_values = sample_info["crop_values"]
# frame_number = 438

# # # new dataset
# downsample = 1
# dataset = FSDataset(
#     path_to_data + image_filename,
#     path_to_data + calibration_filename,
#     image_numbers,
#     downsample,
#     #crop_values,
#     ref_crop_center=sample_info["ref_crop_center"], 
#     crop_size=sample_info["crop_size"], 
#     enforce_divisible=-1,
#     frame_number=frame_number,
#     blank_filename=path_to_data + sample_info["blank_filename"],
#     noise=[0, 0]
# )

# plt.figure()
# plt.imshow(dataset.images[0].squeeze(), cmap='gray')
# plt.show()


# depth_range = sample_info["depth_range"] #(-1, 2)

# depth_range = [-4.5, -2, 1]
# low_plane = depth_range[0] 
# high_plane = depth_range[1]
# num = 2
# depths = np.linspace(low_plane, high_plane, num, endpoint=True)
# depth_values = torch.from_numpy(depths).to(torch.float32).cuda()

# batch_size = 10
# volume = get_ss_volume_from_dataset(
#     dataset, batch_size, depth_values, get_squared=False)
# volume = volume.squeeze().cpu().numpy()


# for n in [0, 1]:
#     plt.figure() 
#     plt.imshow(volume[n][50:-50, 50:-50], cmap='gray') 
#     plt.title(float(depth_values[n]))


# play_video(volume)








# # showing center portion of skull well 
# gpu_number = "3"
# os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
# sample_name = "skull_with_tool"

# config_dict = generate_config_dict(gpu_number, sample_name)
# sample_info = config_dict["sample_info"]

# # sample_info = get_sample_information(sample_name)
# image_numbers = [13, 15, 20, 25, 27]
# #image_numbers = [6, 10, 20, 36, 40]
# image_numbers = torch.arange(48).tolist() 
# image_filename = sample_info["image_filename"]
# calibration_filename = sample_info["calibration_filename"]
# # #crop_values = sample_info["crop_values"]
# frame_number = -1 
# bf = None 

# # new dataset with the crops
# downsample = 1
# dataset = FSDataset(
#     path_to_data + image_filename,
#     path_to_data + calibration_filename,
#     image_numbers,
#     downsample,
#     #crop_values,
#     ref_crop_center=(0.65, 0.5), #sample_info["ref_crop_center"], 
#     crop_size=(0.25, 0.35), #sample_info["crop_size"], 
#     enforce_divisible=-1,
#     frame_number=frame_number,
#     blank_filename=bf,
#     noise=[55, 0]
# )

# plt.figure()
# plt.imshow(dataset.images[0].squeeze(), cmap='gray')
# plt.show()


# depth_range = (-1, 2)
# low_plane = depth_range[0] 
# high_plane = depth_range[1]
# num = 30
# depths = np.linspace(low_plane, high_plane, num, endpoint=True)
# depth_values = torch.from_numpy(depths).to(torch.float32).cuda()

# batch_size = 10
# volume = get_ss_volume_from_dataset(
#     dataset, batch_size, depth_values, get_squared=False)
# volume = volume.squeeze().cpu().numpy()


# for n in [16, 18, 20, 22, 24]:
#     plt.figure() 
#     plt.imshow(volume[n][50:-50, 50:-50], cmap='gray') 
#     plt.title(n) 


# play_video(volume)












# code for z-stacking with the patched finger sample 


gpu_number = "3"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number
sample_name = "finger"

sample_info = get_sample_information(sample_name)
image_numbers = torch.arange(48).tolist() 
#image_numbers = [13, 15, 20, 25,27]
image_numbers = [6, 10, 20, 36, 40]
image_filename = sample_info["image_filename"]
calibration_filename = sample_info["calibration_filename"]
crop_values = sample_info["crop_values"]
frame_number = -1 
bf = None 


patch = [1200, 2160 + 32 * 10, 400, 1360 + 32 * 10]
crop_values = [patch[0] / 4096, patch[1]/4096, patch[2] / 3072, patch[3]/3072] 

noise = [20, 0]


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


show_hairs = False 
if show_hairs:
    depth_range = (-4, 1)
    low_plane = depth_range[0] 
    high_plane = depth_range[1]
    num = 30
    batch_size = 10
    depths = np.linspace(low_plane, high_plane, num, endpoint=True)

    depths = np.asarray([-1.58 - 0.1, -1.58, -1.58 + 0.1])
    depths = np.linspace(-2, -1.3, 10)
    depth_values = torch.from_numpy(depths).to(torch.float32).cuda()
    volume = get_ss_volume_from_dataset(
        dataset, batch_size, depth_values, get_squared=False)
    volume = volume.squeeze().cpu().numpy()


    plt.figure()
    plt.imshow(volume[0][200:800, :600], cmap='gray')
    plt.show()

    nums = [0, 2, 4, 8, 9]
    for n in nums:
        plt.figure()
        plt.imshow(volume[n][200:800, :600], cmap='gray')
        plt.title("{:.2f}".format(depths[n])) 
        plt.show()

    plt.figure()
    plt.imshow(volume[14], cmap='gray')
    plt.show()

else:
    depth_range = (-2, -1.8)
    low_plane = depth_range[0] 
    high_plane = depth_range[1]
    num = 10
    batch_size = 10
    depths = np.linspace(low_plane, high_plane, num, endpoint=True)


    num = 3
    depths = np.asarray([-2, -1.9, -1.8])


    depth_values = torch.from_numpy(depths).to(torch.float32).cuda()
    volume = get_ss_volume_from_dataset(
        dataset, batch_size, depth_values, get_squared=False)
    volume = volume.squeeze().cpu().numpy()


    plt.figure()
    plt.imshow(volume[0][200:400, 700:900], cmap='gray')
    plt.show()

    for n in range(num):
        plt.figure()
        plt.imshow(volume[n][200:400, 700:900], cmap='gray')
        plt.title("{:.2f} \n 200x200".format(depths[n])) 
        ax = plt.gca()
        ax.set_xticks([]) 
        ax.set_yticks([])
        plt.show()

ri = dataset.reference_image.cpu().squeeze()
plt.imshow(ri[200:400, 700:900], cmap='gray') 
ax = plt.gca()
ax.set_xticks([]) 
ax.set_yticks([])