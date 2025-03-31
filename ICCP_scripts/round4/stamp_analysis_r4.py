from utility_functions import download_image, remove_global_tilt, get_reference_image
import numpy as np 
from matplotlib import pyplot as plt 
from filmscope.config import log_folder, path_to_data
from filmscope.util import load_dictionary, save_dictionary
import torch 
import torch.nn.functional as F
from tqdm import tqdm 
import os 
import math 
import scipy
import cv2 
from filmscope.reconstruction import generate_config_dict, RunManager


import numpy as np
from skimage.draw import disk
from regions.core import PixCoord
from regions.shapes.circle import CirclePixelRegion


#experiment_dict = load_dictionary(log_folder + '/stamp_runs_round4.json')
#experiment_log_folder = log_folder + f'/stamp_results_r4'
#base_key = 7784

experiment_dict = load_dictionary(log_folder + '/stamp_runs_round5_rectified.json')
experiment_dict = load_dictionary(log_folder + '/stamp_runs_round6_rectified_0327.json')
#experiment_log_folder = log_folder + f'/stamp_results_r5_rectified'
experiment_log_folder = log_folder + f'/stamp_results_r6_rectified'
base_key = 8382



height_maps = None 
show = True 
iter = 50
for i, (key, item) in enumerate(experiment_dict.items()):

    filename = experiment_log_folder + f"/run_{key}_iter_{iter}_depth.npy"
    height = np.load(filename) 
    if height_maps is None: 
        height_maps = np.zeros((len(experiment_dict), height.shape[0], height.shape[1]))
    height_maps[i] = height



    if len(item["cameras"]) > 16:
       continue


    if show and item["noise"][0] == 0:
        print(item["cameras"])
        fig, axes = plt.subplots(6, 8, figsize=(2, 1.4))
        for i, j in np.ndindex(axes.shape):
            ax = axes[i, j] 
            ax.set_xticks([])
            ax.set_yticks([])
            number = (5 - i) + 6 * j
            
            if number in item["cameras"]:
                ax.set_facecolor("red") 
            else:
                ax.set_facecolor("black") 
        axes[3, 3].set_facecolor('blue')
        #plt.title(f"{key}, noise {item['noise'][0]}")
        #plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2) 
        axes[1].imshow(height, cmap='turbo') 
        height = remove_global_tilt(height) 
        height = height - np.min(height) 
        axes[0].imshow(height[500:650, 450:600], cmap='magma')
        plt.title(f"{key}")
        plt.show()

        # filename = experiment_log_folder + f"/run_{key}_iter_{iter}_warp.npy"
        # warp = np.load(filename) 
        # plt.figure()
        # plt.imshow(warp[100:300, 100:300], cmap='gray')
        # plt.show()

        # plt.figure()
        # plt.plot(item["loss"])
        # plt.title(f"{key}, {i}")
        # plt.show()
    #break



warp_filename = experiment_log_folder + f"/run_{base_key}_iter_{75}_warp.npy"
warp = np.load(warp_filename) 
plt.imshow(warp[50:-50, 50:-50], cmap='gray')
plt.imshow(warp[450:600, 450:600], cmap='gray') 






# # inefficiently getting reference image 
# sample_name = "stamp_20250327"
# custom_image_numbers = [20] 
# gpu_number = "0" 
# config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=1,
#                                     camera_set="custom", 
#                                     frame_number=-1,
#                                     run_args={"iters": 2, "batch_size": 12, "num_depths": 32,
#                                                 "display_freq": 20, "lr": 0.0007},
#                                     loss_weights={"smooth": 0.2}, 
#                                     custom_image_numbers=custom_image_numbers, 
#                                     #custom_crop_info={'crop_size': (0.15, 0.2)} #, "ref_crop_center": (0.5, 0.5)}
#                                     )
# config_dict["sample_info"]["depth_range"] = [1.5, 2]
# run_manager = RunManager(config_dict)
# ref_image = run_manager.dataset.reference_image.squeeze().cpu().detach().numpy()



# # save the ref image 
# np.save("full_run_ref_0327.npy", ref_image) 
ref_image = np.load("full_run_ref_0327.npy")





w = warp.astype(np.uint8)
gray = cv2.medianBlur(w, 7) 

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 5, param1=8, param2=10, 
                           minRadius=2, maxRadius=4).squeeze()

mn = np.min(w[150:-150, 150:-150])
mx = np.max(w[150:-150, 150:-150])
plt.imshow(w, clim=(mn, mx), cmap='gray') 
plt.scatter(circles[:, 0], circles[:, 1], s=2)
ax = plt.gca() 
ax.set_xlim(300, 0) 
ax.set_ylim(0,300)


centers = circles[:, :2] 
radius = 5
x = [] 
y = []
for c in centers:
    angles = np.linspace(0, 2*np.pi, 9) 
    x_vals = c[0] + radius * np.cos(angles)
    y_vals = c[1] + radius * np.sin(angles) 
    x.append(x_vals) 
    y.append(y_vals) 

x = np.asarray(x) 
y = np.asarray(y)
background = np.concatenate(([x.flatten()], [y.flatten()]), axis=0).T


mn = np.min(w[150:-150, 150:-150])
mx = np.max(w[150:-150, 150:-150])
plt.imshow(w, clim=(mn, mx), cmap='gray') 
plt.scatter(background[:, 0], background[:, 1], s=2)

plt.scatter(centers[:, 0], centers[:, 1], color='red', s=4)
ax = plt.gca()
ax.set_xlim(300, 400) 
ax.set_ylim(400,300)

np.save("peak_locations_0327.npy", centers) 
np.save("trough_locations_0327.npy", background)








# # or we can set a starting point 
# # and an angle and whatever 
# # and make our own grid ? 
# start_point = (20, 15) 

# plt.imshow(w, clim=(100, 120), cmap='gray') 
# plt.scatter([start_point[0]], [start_point[1]])
# #plt.scatter(x, y, s=2)
# #plt.scatter(centers[:, 0], centers[:, 1], color='red', s=4)


# y_end = 500 
# spacing = 11 
# angle = 11 * np.pi / 180 
# y_vals = start_point[1] + 


# ax = plt.gca()
# ax.set_xlim(0, 100) 
# ax.set_ylim(0,100)

















import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from scipy.optimize import minimize

def interactive_grid_refinement(points,
                                manual_angle=0, 
                                manual_spacing=9):
    """Manual grid refinement with visual feedback"""
    # Initial estimates (adjust these interactively)
    #manual_angle = 0.0    # Start with 0° (horizontal)
    #manual_spacing = 9 # Initial guess for spacing
    stagger_offset = 0.5  # 0.5 = 50% offset between rows
    
    # Set up optimization
    def loss(params):
        angle, spacing = params
        grid = generate_grid(points, angle, spacing, stagger_offset)
        dists = KDTree(grid).query(points)[0]
        return np.median(dists)
    
    # Run optimization with bounds
    result = minimize(loss, [manual_angle, manual_spacing], 
                     bounds=[(0.6, 0.8), (11, 12)])
    best_angle, best_spacing = result.x

    print(best_angle, best_spacing)
    
    # Generate final grid
    final_grid = generate_grid(points, best_angle, best_spacing, stagger_offset)
    
    # Visual comparison
    plt.figure(figsize=(12, 8))
    plt.scatter(points[:,0], points[:,1], c='blue', label='Original')
    plt.scatter(final_grid[:,0], final_grid[:,1], c='red', marker='x', 
               alpha=0.5, label='Grid Template')
    plt.title(f"Best Fit: {best_angle:.1f}° angle, {best_spacing:.2f} spacing")
    plt.legend()
    plt.show()
    
    return final_grid

def generate_grid(points, angle, spacing, offset_ratio):
    """Generate grid with manual parameters"""
    theta = np.radians(-angle)
    rot_mat = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
    
    # Get bounds in rotated space
    derotated = points @ rot_mat.T
    x_min, x_max = derotated[:,0].min(), derotated[:,0].max()
    y_min, y_max = derotated[:,1].min(), derotated[:,1].max()
    
    # Generate grid
    grid = []
    y_spacing = spacing  # Keep rectangular for simplicity
    for row_idx, y in enumerate(np.arange(y_min-2*y_spacing, y_max+2*y_spacing, y_spacing)):
        x_offset = (row_idx % 2) * spacing * offset_ratio
        x_vals = np.arange(x_min-2*spacing + x_offset, 
                          x_max+2*spacing + x_offset, 
                          spacing)
        grid.append(np.column_stack([x_vals, np.full_like(x_vals, y)]))
    
    grid = np.vstack(grid)
    return grid @ rot_mat.T

# Usage:
# 1. Run with your data
# 2. Visually check plot
# 3. Adjust manual_angle/manual_spacing in code
# 4. Re-run until alignment looks good
idx0 = circles[:, 0] < 350 
idx1 = circles[:, 1] < 350
your_points = circles[np.bitwise_and(idx0, idx1)]
your_points = your_points[:, :2]
final_grid = interactive_grid_refinement(your_points, manual_spacing=9.5,
                                         manual_angle=-1)














idx0 = circles[:, 0] < 200 
idx1 = circles[:, 1] < 200
points = circles[np.bitwise_and(idx0, idx1)]

plt.scatter(points[:, 0], points[:, 1]) 

# Refine and visualize
refined, dx, dy, calc_angle = refine_staggered_grid(points, visual=True)
print(f"True angle: {angle}° → Calculated angle: {calc_angle:.1f}°")
print(f"X spacing: {dx:.2f}, Y spacing: {dy:.2f}")





import cv2
import numpy as np

def adaptive_three_level_threshold(image, block_size=51, C1=2, C2=2):
    # Convert to grayscale if the image is color
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    # Apply two adaptive thresholds
    thresh1 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, block_size, C1)
    
    plt.figure()
    plt.imshow(thresh1[600:650, 600:650])

    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, block_size, C2)

    plt.figure()
    plt.imshow(thresh2[600:650, 600:650])
    
    # Combine the thresholds to create three levels
    result = np.zeros_like(gray)
    result[thresh1 == 255] = 128
    result[thresh2 == 255] = 255
    
    return result

# Load an image
image = warp.astype(np.uint8)

# Apply the three-level adaptive threshold
result = adaptive_three_level_threshold(image, C1=2.5, C2=0.5)
plt.figure()
plt.imshow(result[500:650, 500:650])
plt.figure()
plt.imshow(result)


np.save("binary_mask.npy", result)

# Display the results
cv2.imshow('Original', image)
cv2.imshow('Three-Level Threshold', result)
cv2.waitKey(0)
cv2.destroyAllWindows()