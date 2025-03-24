import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
#from .misc import tocuda
from filmscope.recon_util import generate_base_grid, tocuda, inverse_warping

from filmscope.reconstruction import generate_config_dict
from filmscope.datasets import FSDataset
from filmscope.config import path_to_data, log_folder
import numpy as np
import os 

from utility_functions import remove_global_tilt

from matplotlib import pyplot as plt 




gpu_number = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number


# testing modifications from warp_functions.py


# the goal of this is to warp the "other_image" into the reference camera frame
# using the reference camera's depth map
# view_image: [batch_size, channels, height, width]
# reference_depth: [batch_size, height, width, 1]
# ref_shift_slope: [batch_size, height, width, 2]
# view_inv_inter_camera: [batch_size, height, width, 2]
# view_warped_shift_slope: [batch_size, height, width, 2]
def inverse_warping2(
    image,
    depth_est,
    # ref_shift_slope,
    #view_inv_inter_camera,
    warped_shift_slope,
    base_grid=None,
):
    depth_est = depth_est.unsqueeze(-1)
    # if the mask is not None, mulitiply it with the ref_depth_est

    if base_grid is None:
        base_grid = generate_base_grid(image.shape[2:]).cuda()

    slope_shifts = warped_shift_slope * -1 * depth_est
    full_grid = base_grid + slope_shifts #+ view_inv_inter_camera

    # and warp
    warped_image = F.grid_sample(
        image, full_grid, mode="bilinear", padding_mode="zeros",
        align_corners=False
    )

    # the mask is just where "full_grid" is >1 or <-1
    grid_x = full_grid[:, :, :, [0]]
    grid_y = full_grid[:, :, :, [1]]
    mask = (grid_x >= -1) & (grid_x <= 1) & (grid_y >= -1) & (grid_y <= 1)
    # permute to match images shape 
    mask = mask.permute(0, 3, 1, 2)

    return warped_image, mask


# probably the easiest way to get the maps is just to make a dataset 
# with only the camera of interest 
cam_number = 3
sample_name = f"stamp_camera_{cam_number}"
config_dict = generate_config_dict(
    sample_name=sample_name, gpu_number="0", downsample=1,
    camera_set="all", use_neptune=False,
    frame_number=-1,
    run_args={"iters": 1, "batch_size": 12, "num_depths": 32,
                "display_freq": 20},
    #custom_image_numbers=custom_image_numbers, 
    custom_crop_info={'crop_size': (0.15, 0.2)} #, "ref_crop_center": (0.5, 0.5)}
)


base_num = 0
if cam_number == base_num:
    nums = [base_num]
else:
    nums = [cam_number, base_num]
info = config_dict["sample_info"]
dataset = FSDataset(
    path_to_data + info["image_filename"], 
    path_to_data + info["calibration_filename"], 
    image_numbers=nums, 
    downsample=1, 
    ref_crop_center=info["ref_crop_center"], 
    crop_size=info["crop_size"],
    height_est=info["height_est"]
    #crop_values=sample_info["crop_values"]
)


depth_filename = log_folder + f'/stamp_perspective_comp/camera_{cam_number}_iter_140_depth.npy'
image = dataset.reference_image


height_est = 0 #2.45 # this is the amount of shift already accounted for when the images are aligned 

depth_est = torch.from_numpy(np.load(depth_filename)) - height_est


depth_est = depth_est #* 5

warped_shift_slope = dataset.ref_camera_shift_slopes

warped, mask = inverse_warping2(
    #image[None].cuda(),
    depth_est[None, None].cuda(),
    depth_est[None].cuda(),
    # ref_shift_slope,
    #view_inv_inter_camera,
    warped_shift_slope.cuda(),
    base_grid=None,
)

warped = warped.squeeze().cpu() 
mask = mask.squeeze().cpu()

# plt.figure()
# plt.imshow(warped, cmap='magma') 
# plt.show()

# plt.figure()
# plt.imshow(depth_est, cmap='magma') 
# plt.show()



def get_slopes(height_map):
    c1 = height_map[50:-50, 50:-50]
    dim0 = np.mean(c1, axis=0) 
    dim1 = np.mean(c1, axis=1) 

    slope0, intercept0 = np.polyfit(np.arange(len(dim0)), dim0, 1)
    slope1, intercept1 = np.polyfit(np.arange(len(dim1)), dim1, 1)

    return slope0, slope1 






# thiss should warp an image from the reference perspective
# into a different camera's perspective (but at the 0 plane)
# using the reference camera's depth map 
def forward_warping(ref_image, depth_map, view_inv_inter_camera,
                    ref_shift_slope, 
                    view_shift_slope = None):
    base_grid = generate_base_grid(ref_image.shape[2:]).cuda()
    
    slope_shifts = (ref_shift_slope * -1) * depth_map
    if view_shift_slope is not None:
        slope_shifts = slope_shifts + view_shift_slope * depth_map

    # i'm doing negative inter_camera because these are 
    # loaded from the dataset
    # which I think turned them negative at some point 
    # well... yeah because we have to use the inv shifts
    # since the depth map is still from the perspective of 
    # the reference image 
    full_grid = base_grid + slope_shifts - view_inv_inter_camera

    view_image = F.grid_sample(
        ref_image, full_grid, mode="bilinear", padding_mode="zeros",
        align_corners=False
    )

    return view_image 


index =torch.where(dataset.image_numbers == base_num)[0][0]
camera_map = dataset.inv_inter_camera_maps[index][None]

base_cam_perspective = forward_warping(
    depth_est[None, None].cuda(),
    depth_est[None, :, :, None].cuda(),
    ref_shift_slope=warped_shift_slope.cuda(), 
    view_inv_inter_camera=camera_map.cuda(),
    view_shift_slope=None #dataset.warped_shift_slope_maps[index][None].cuda()
).squeeze().cpu()


print(get_slopes(depth_est.squeeze().numpy()))
print(get_slopes(warped.squeeze().numpy()))
print(get_slopes(base_cam_perspective.numpy()))









f = "/media/Friday/Temporary/Clare/ICCP_result_storage/round_2_results/stamp_perspective_comp"
f = "/media/Friday/Temporary/Clare/ICCP_result_storage/round_2_results/stamp_perspective_comp_06_21"
# okay idk, let's do something else 
for num in range(48):
    image_filename = f + f'/camera_{num}_iter_239_depth.npy'
    if not os.path.exists(image_filename):
        continue 
    image = np.load(image_filename) 
    image = remove_global_tilt(image) 
    image = image - np.min(image)
    plt.figure()
    plt.imshow(image[50:-50, 50:-50], cmap='magma') 
    plt.title(f"camera {num}")
    plt.show()

    plt.figure()
    plt.imshow(image[100:300, 100:300], cmap='magma') 
    plt.title(f"camera {num}")
    plt.show()



    #break


f = "/media/Friday/Temporary/Clare/ICCP_result_storage/round_2_results/stamp_perspective_comp_06_21"
fig, axes = plt.subplots(6, 8)
mn = np.inf 
mx = -np.inf
avgs = []
ranges = []
for i, j in np.ndindex(axes.shape):
    ax = axes[i, j]
    ax.set_xticks([])
    ax.set_yticks([])
    number = (5 - i) + 6 * j

    image_filename = f + f'/camera_{number}_iter_150_depth.npy'
    if not os.path.exists(image_filename):
        image_filename = f + f'/camera_{number}_iter_140_depth.npy'
    if not os.path.exists(image_filename):
        continue

    image = np.load(image_filename)
    image = image[50:-50, 50:-50]
    image = remove_global_tilt(image) 
    #mn = min(mn, np.min(image)) 
    #mx = max(mx, np.max(image))
    #avgs.append(np.mean(image))
    #image = image - np.min(image)
    #ax.imshow(image[100:200, 100:200], cmap='magma') 
    imc = image[100:200, 100:200]
    ax.imshow(imc, cmap='magma', clim=(-1.1, -0.8))

    mn = min(mn, np.min(imc)) 
    mx = max(mx, np.max(imc))

    avgs.append(np.mean(imc))
    ranges.append(np.max(imc) - np.min(imc))

    
    #break





# probably the easiest way to get the maps is just to make a dataset 
# with only the camera of interest 
cam_number = 3
sample_name = f"stamp_06_21_camera_{cam_number}"
config_dict = generate_config_dict(
    sample_name=sample_name, gpu_number="0", downsample=1,
    camera_set="all", use_neptune=False,
    frame_number=-1,
    run_args={"iters": 1, "batch_size": 12, "num_depths": 32,
                "display_freq": 20},
    #custom_image_numbers=custom_image_numbers, 
    custom_crop_info={'crop_size': (0.15, 0.2)} #, "ref_crop_center": (0.5, 0.5)}
)
base_num = 0
if cam_number == base_num:
    nums = [base_num]
else:
    nums = [cam_number, base_num]
info = config_dict["sample_info"]
dataset = FSDataset(
    path_to_data + info["image_filename"], 
    path_to_data + info["calibration_filename"], 
    image_numbers=np.arange(48).tolist(), 
    downsample=1, 
    ref_crop_center=info["ref_crop_center"], 
    crop_size=info["crop_size"],
    height_est=info["height_est"]
    #crop_values=sample_info["crop_values"]
)




fig, axes = plt.subplots(6, 8)
for i, j in np.ndindex(axes.shape):
    ax = axes[i, j]
    ax.set_xticks([])
    ax.set_yticks([])
    number = (5 - i) + 6 * j

    image = dataset.images[number].squeeze()
    #image = remove_global_tilt(image) 
    #image = image - np.min(image)
    ax.imshow(image[100:200, 100:200], cmap='gray') 