import numpy as np 
from skimage.draw import disk
from filmscope.util import load_dictionary, save_dictionary
from filmscope.config import log_folder
from matplotlib import pyplot as plt 
from utility_functions import remove_global_tilt
from filmscope.reconstruction import RunManager, generate_config_dict
import cv2 
import torch 
import scipy
from tqdm import tqdm 
import os 

from filmscope.recon_util import get_sample_information
from filmscope.calibration import CalibrationInfoManager
from filmscope.calibration import Filmscope_System
from filmscope.config import path_to_data

centers = np.load("peak_locations_0327.npy")
background = np.load("trough_locations_0327.npy")

experiment_dict = load_dictionary(log_folder + '/stamp_runs_round5_rectified.json')
experiment_dict = load_dictionary(log_folder + '/stamp_runs_round6_rectified_0327.json')
experiment_log_folder = log_folder + f'/stamp_results_r5_rectified'
experiment_log_folder = log_folder + f'/stamp_results_r6_rectified'
base_key = 9981
base_key = 8382

ref_image = np.load("full_run_ref_rect5.npy")
ref_image = np.load("full_run_ref_0327.npy")
# some good keys for 0325 runs (round 5)
# 5874 (5x5)
# 7456 (4x4)
# 5846 (3x3) 
# 6216 (spread 3x3) 
# 


# let's get angle information for the cameras 
info = get_sample_information(sample_name="stamp_20250327")
system = Filmscope_System(path_to_data + '/' + info["calibration_filename"])

# not sure why this wasn't in there
#system.calib_manager.vertex_spacing_m = 2e-3

pixel_size_mm = system.calib_manager.pixel_size * 1e3


iter = 50
height_filename = experiment_log_folder + f"/run_{9531}_iter_{iter}_depth.npy"
height = np.load(height_filename)  
height_b = cv2.GaussianBlur(height, (25, 25), 0) 
height_surf = height - height_b

binary_mask = np.load("binary_mask.npy")
plt.imshow(height_surf[50:-50, 50:-50], cmap='magma') 
plt.colorbar()
plt.imshow(binary_mask[50:-50, 50:-50], cmap='gray', alpha=0.2)
ax = plt.gca()
ax.set_xlim(400, 500)
ax.set_ylim(500, 400) 


# get angle for each camera ? 

angles = []
for cam_num in range(48):
    mag = system.get_magnification_at_plane(cam_num, 0, 0) 
    pixel_shift_x, pixel_shift_y = system.get_shift_slopes(cam_num, [1500], [1500]) 
    thetax = np.arctan(pixel_shift_x[0] * pixel_size_mm / mag)
    thetay = np.arctan(pixel_shift_y[0] * pixel_size_mm / mag)
    angles.append([thetax, thetay])

angles = np.asarray(angles)
angles_deg = angles * 180 / np.pi

# define a bunch of functions
# from perplexity
def average_circle_value(image, location, radius):
    """
    Calculate the average pixel value within a circle centered at (center_x, center_y) with the given radius.

    Parameters:
    - image: 2D numpy array representing the image.
    - center_x: x-coordinate of the circle's center.
    - center_y: y-coordinate of the circle's center.
    - radius: Radius of the circle.

    Returns:
    - Average pixel value within the circle.
    """
    
    center_y, center_x = location

    # Ensure the center coordinates are floats for precise calculations
    center_x, center_y = float(center_x), float(center_y)
    
    # Generate a mask for the circle
    rr, cc = disk((int(center_y), int(center_x)), radius, shape=image.shape)
    
    # Calculate the average pixel value using the mask
    masked_image = image[rr, cc]
    return masked_image
    average_value = np.mean(masked_image)
    
    return average_value

def process_run(key, show=True, iter=75, rad=1):
    height_filename = experiment_log_folder + f"/run_{key}_iter_{iter}_depth.npy"
    height = np.load(height_filename)  
    height_b = cv2.GaussianBlur(height, (25, 25), 0) 
    height_surf = height - height_b   

    if show:
        plt.figure()
        plt.imshow(height_surf, cmap='magma')
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(height_surf[600:650, 600:650], cmap='magma') 
        plt.colorbar()
        plt.show()

        plt.figure()
        plt.imshow(height_surf, cmap='magma')
        plt.scatter(centers[:, 0], centers[:, 1], s=2)
        plt.scatter(background[:, 0], background[:, 1], s=2)
        ax = plt.gca()
        ax.set_xlim(100, 300)
        ax.set_ylim(700, 500)

        warp_filename = experiment_log_folder + f"/run_{key}_iter_{iter}_warp.npy"
        warp = np.load(warp_filename) 
        plt.figure()
        mn = np.min(warp[500:700, 100:300])
        mx = np.max(warp[500:700, 100:300])
        plt.imshow(warp, cmap='gray', clim=(mn, mx))
        plt.scatter(centers[:, 0], centers[:, 1], s=2, color='red',
                    label="pillars")
        plt.scatter(background[:, 0], background[:, 1], s=2, color='blue',
                    label="background")
        ax = plt.gca()
        plt.legend()
        ax.set_xlim(100, 300)
        ax.set_ylim(700, 500)


    center_vals = [] 
    for p in centers:
        center_vals = np.concatenate((center_vals, average_circle_value(height_surf, [p[1], p[0]], radius=rad)))
    #center_vals = [average_circle_value(height_surf, [p[1], p[0]], radius=rad) for p in centers]
    #background_vals = [average_circle_value(height_surf, [p[1], p[0]], radius=1) for p in background]
    
    
    # use the other approach 
    height_surfc = height_surf[50:-50, 50:-50]
    maskc = binary_mask[50:-50, 50:-50]
    background_vals = height_surfc[maskc == 0].flatten()
    
    
    center_vals = np.asarray(center_vals)
    center_vals = center_vals[~np.isnan(center_vals)] 
    background_vals = np.asarray(background_vals) 
    background_vals = background_vals[~np.isnan(background_vals)] 

    return center_vals, background_vals

def hist_peaks_troughs(peak_values, trough_values, use_median=True):
    p2 = peak_values * 1e3 
    t2 = trough_values * 1e3

    plt.figure()
    plt.hist(p2, color='red', alpha=0.5, density=True, bins=45, label="Pillars")
    plt.hist(t2, color='blue', alpha=0.5, density=True, bins=45*5, label="Background")

    plt.axvline(x=np.mean(p2), color='black', linestyle='--')
    plt.axvline(x=np.mean(t2), color='black', linestyle='--')

    plt.xlabel("Axial Position (um)")
    plt.ylabel("Density") 

    plt.legend()
    mu_t, std_t = scipy.stats.norm.fit(t2)
    if use_median:
        mu_t =  np.median(t2)
    xmin, xmax = (np.min(t2), np.max(t2))
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mu_t, std_t)
    #plt.plot(x, p, 'k', linewidth=2, color='blue')
    plt.axvline(x=mu_t, color='blue', linestyle='--')

    mu_p, std_p = scipy.stats.norm.fit(p2)
    if use_median:
        mu_p = np.median(p2)
    xmin, xmax = (np.min(p2), np.max(p2))
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mu_p, std_p)
    #plt.plot(x, p, 'k', linewidth=2, color='red')
    plt.axvline(x=mu_p, color='red', linestyle='--')

    # remove this 
    ax = plt.gca()
    #ax.set_xlim(-50, 60)

    plt.title("pillar std: {:.0f} um, background std: {:.0f} um\n separation: {:.0f} um".format(std_p, std_t, mu_p - mu_t))

# 2219 (4 cams) - should probably show this
# 2048 (9 cams)
# 7652 (5 cams) also very strong, include this 

center_vals, background_vals = process_run(1934, iter=50, rad=2.5) 
hist_peaks_troughs(center_vals, background_vals, use_median=False)


# center_vals, background_vals = process_run(3421, iter=25) 
# hist_peaks_troughs(center_vals, background_vals)


# let's make an updated result dictionary
# which has all values (all iters) for all runs


name = "rectified_results_0327.json"
if os.path.exists(name):
    values = load_dictionary(name) 
else:
    values = {}


# re-running with larger radius in processing
name = "rectified_results_0327_v2.json"
if os.path.exists(name):
    values = load_dictionary(name) 
else:
    values = {}

# commenting out because we already did this 
for key, item in tqdm(experiment_dict.items()):
    if key in values:
        continue
    run_dict = item.copy()
    
    for iter in [25, 50, 75, 100]:
        center_vals, background_vals = process_run(key, iter=iter, show=False,
                                                   rad=2.5) 
        run_dict[iter] = {
            "center": center_vals, 
            "background": background_vals
        }

    values[key] = run_dict 

save_dictionary(values, name)







# these were for the round 5 (0325 stamp) runs
# eligible_keys = [
#     6216, 6118, 5846, 2732, 
#     3264, 9225, 7891, 4373, 
#     5874, 6744, 3468, 705, 
#     2599, 9893, 537, 6216, 
#     6291, 4851, 2046, 1871, 
#     7599, 797, 659, 3219, 
#     8615, 7456, 3337, 2745, 
#     4735, 6687, 2292, 9981, 
#     545, 2161, 8021, 8060, 
#     3044, 2779, 6655, 1740, 
#     6118, 5846, 3781, 2775, 
#     2603, 
# ]

def get_angle_information(cameras):
    set_angles = angles[cameras]

    max_angles = np.max(set_angles, axis=0) 
    min_angles = np.min(set_angles, axis=0)
    ranges = max_angles - min_angles
    averages = np.mean(set_angles, axis=0)
    return ranges, averages




def get_code(cameras):
    binary_rep = [0] * 48 
    for num in cameras:
        binary_rep[num] = 1 
    binary_string = ''.join(map(str, binary_rep))
    decimal_number = int(binary_string, 2) 
    return decimal_number 

# so we have the values... 
# then what do we do? 
# let's initially pick a few 
# subsets to look at 
#noise = 10 

nums = []
stds = [] 
errors = []
ranges = []
avgs = []
keys = []
codes = []
med_errors = []
noises = []

for key, item in tqdm(experiment_dict.items()):
    #if key not in eligible_keys:
    #    continue 
    #if item["noise"][0] != noise:
    #    continue

    noises.append(item["noise"][0])

    locs = values[key] 

    min_dist_error = np.inf 
    min_std = np.inf 
    best_iter = np.inf
    #iter0 = np.inf 
    #iter1 = np.inf
    for iter in [50]: #[25, 50, 75, 100]:
        run_centers = locs[iter]["center"]
        run_background = locs[iter]["background"]
        mu_c, std_c = scipy.stats.norm.fit(run_centers)
        mu_b, std_b = scipy.stats.norm.fit(run_background)
        
        dist_error = abs(50e-3 - (mu_c - mu_b))
        #if dist_error < min_dist_error:
        #    iter0 = iter
        #min_dist_error = min(min_dist_error, dist_error) 
        if std_c < min_std:
            min_std = std_c
            best_iter = iter
            min_dist_error = dist_error


            median_error = np.median(run_centers) - np.median(run_background)
        #min_std = min(std_c, min_std)

    if min_dist_error > 0.05:
        print(key, len(item["cameras"]), min_dist_error, 
              best_iter)
        

    code = get_code(item["cameras"])
    codes.append(code)

    med_errors.append(median_error)

    keys.append(key)

    nums.append(len(item["cameras"]))
    errors.append(min_dist_error) 
    stds.append(min_std)

    angle_range, angle_avg = get_angle_information(item["cameras"]) 
    ranges.append(angle_range.tolist()) 
    avgs.append(angle_avg.tolist())
    #break 

ranges = np.asarray(ranges) 
avgs = np.asarray(avgs)
keys = np.asarray(keys) 
codes = np.asarray(codes)
noises = np.asarray(noises)
nums = np.asarray(nums)
med_errors = np.asarray(med_errors)

plt.figure()
plt.scatter(nums, np.asarray(errors)*1e3)
#plt.title(f"noise: {noise}")
plt.xlabel("# Cameras") 
plt.ylabel("Height Error (um)")


plt.figure()
plt.scatter(nums, np.asarray(stds) * 1e3)
#plt.title(f"noise: {noise}")
plt.xlabel("# Cameras") 
plt.ylabel("Height Estimation Std Dev (um)")


noise = 50
idx = noises == noise
plt.figure()
plt.scatter(nums[idx], np.asarray(stds)[idx] * 1e3)
#plt.title(f"noise: {noise}")
plt.xlabel("# Cameras") 
plt.ylabel("Height Estimation Std Dev (um)")
ax = plt.gca()
ax.set_ylim(-1, 160)
plt.title(f"noise {noise}")


plt.figure()
plt.scatter(nums[idx], np.abs(np.asarray(errors)[idx]*1e3))
#plt.title(f"noise: {noise}")
plt.xlabel("# Cameras") 
plt.ylabel("Height Error (um)")
ax = plt.gca()
ax.set_ylim(-1, 140)
plt.title(f"noise {noise}")


plt.figure()
plt.scatter(np.max(ranges, axis=1), np.asarray(stds)*1e3,
            c=nums)

plt.figure()
plt.scatter(np.max(ranges, axis=1), np.abs(np.asarray(errors)*1e3),
            c=nums)

# plt.figure()
# plt.scatter(avgs[:, 1], np.asarray(stds)*1e3)

# plt.figure()
# plt.scatter(avgs[:, 1], avgs[:, 0], c=stds)



# angle_diff = np.linalg.norm(avgs - angles[20], axis=1)
# plt.figure()
# plt.scatter(angle_diff, stds)

# plt.figure()
# plt.scatter(angle_diff, errors)








key = 3836
idx = np.where(keys==key)[0][0]
code = codes[idx]
code_keys = np.where(codes==code)



idx = np.where(noises==0)[0]
np.argmax(med_errors[idx])

keys[idx][42]






# split things out a bit 
unique_noises = np.sort(np.unique(noises))
num_noises = len(np.unique(noises)) 
num_cam_batches = 8 
arr = np.zeros((num_cam_batches, num_noises))
counts = np.zeros((num_cam_batches, num_noises))

for num, noise, dev in zip(nums, noises, stds):
    bach_num = int((num - 1) / 5)
    noise_num = np.where(unique_noises == noise)[0][0]
    arr[bach_num, noise_num] += dev
    counts[bach_num, noise_num] += 1
arr = arr / counts

plt.figure()
for p in arr:
    #plt.plot(unique_noises, p, '.')
    plt.plot(p*1e3, '.-')
    #break 
ax = plt.gca()
#ax.set_ylim(-0.0001, 0.002)

plt.figure()
for i, noise in enumerate(unique_noises):
    plt.plot(arr[:, i], '.', label=noise)
plt.legend()

















# # inefficiently getting the original images
sample_name = "stamp_20250325"
custom_image_numbers = np.arange(48).tolist()
gpu_number = "0" 
config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=1,
                                    camera_set="custom", 
                                    frame_number=-1,
                                    run_args={"iters": 2, "batch_size": 12, "num_depths": 32,
                                                "display_freq": 20, "lr": 0.0007},
                                    loss_weights={"smooth": 0.2}, 
                                    custom_image_numbers=custom_image_numbers, 
                                    #custom_crop_info={'crop_size': (0.15, 0.2)} #, "ref_crop_center": (0.5, 0.5)}
                                    )

from filmscope.datasets import FSDataset
info = config_dict["sample_info"]
dataset = FSDataset(
    path_to_data + info["image_filename"],
    path_to_data + info["calibration_filename"],
    info["image_numbers"],
    info["downsample"],
    info["crop_values"],
    frame_number=-1,
    ref_crop_center=info["ref_crop_center"],
    crop_size=info["crop_size"],
    height_est=info["height_est"],
    blank_filename=None #path_to_data + config_dict["sample_info"]["blank_filename"],
    #noise=noise,
)
#run_manager = RunManager(config_dict)
images =dataset.images.squeeze().cpu().detach().numpy() 


measures = np.zeros((48, 3))
for i, num in enumerate(custom_image_numbers):
    img = images[i]
    plt.figure()
    plt.title(num) 
    plt.imshow(img, cmap='gray')
    #plt.imshow(img[500:700, 100:300], cmap='gray')

    rms_contrast = np.std(img) / np.mean(img) 
    ic = img
    ghc = np.max(ic) - np.min(ic)

    imax = np.max(ic) 
    imin = np.min(ic)
    michelson = (imax - imin) / (imax + imin) 

    measures[i, 0] = rms_contrast 
    measures[i, 1] = ghc 
    measures[i, 2] = michelson 

    plt.title("{}, {:.4f} \n {:.4f} {:.4f}".format(num, rms_contrast, ghc, michelson))


plt.figure()
plt.plot((measures[:, 0] - np.min(measures[:, 0]))/ (np.max(measures[:, 0]) - np.min(measures[:, 0]))) 
plt.plot((measures[:, 1] - np.min(measures[:, 1]))/ (np.max(measures[:, 1])- np.min(measures[:, 1]))) 
plt.plot((measures[:, 2] - np.min(measures[:, 2]))/ (np.max(measures[:, 2])- np.min(measures[:, 2])))


import matplotlib
cmap = matplotlib.cm.turbo
for k in range(measures.shape[1]):
    m = measures[:, k] 
    m = (m - np.min(m)) / (np.max(m) - np.min(m)) 
    fig, axes = plt.subplots(6, 8, figsize=(2, 1.4))
    for i, j in np.ndindex(axes.shape):
        ax = axes[i, j] 
        ax.set_xticks([])
        ax.set_yticks([])
        number = (5 - i) + 6 * j

        color = cmap(m[number])
        ax.set_facecolor(color)
        




dataset2 = FSDataset(
    path_to_data + info["blank_filename"],
    path_to_data + info["calibration_filename"],
    info["image_numbers"],
    info["downsample"],
    info["crop_values"],
    frame_number=-1,
    ref_crop_center=info["ref_crop_center"],
    crop_size=info["crop_size"],
    height_est=info["height_est"],
    blank_filename=None #path_to_data + config_dict["sample_info"]["blank_filename"],
    #noise=noise,
)
blank_images =dataset2.images.squeeze().cpu().detach().numpy() 























