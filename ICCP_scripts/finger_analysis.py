from utility_functions import download_image, get_reference_image
import numpy as np 
from matplotlib import pyplot as plt 
from filmscope.config import log_folder, path_to_data
from filmscope.util import load_dictionary, save_dictionary
import torch 
import torch.nn.functional as F
from tqdm import tqdm 
import os 
import matplotlib

from skimage.metrics import structural_similarity as ssim


experiment_dict = load_dictionary(log_folder + '/finger_from_low_res.json')

# # can delete this, just making absolutely sure 
# # nothing important has been deleted 
# from filmscope.config import log_folder, path_to_data, neptune_project, neptune_api_token
# import neptune 
# project = neptune.init_project(project=neptune_project,
#                                api_token=neptune_api_token)
# runs_table_df = project.fetch_runs_table().to_pandas()
# ids = runs_table_df["sys/id"].values
# for key in experiment_dict.keys():
#     assert key in ids 


# and let's load in the low res map 
patch = [1200, 2160 + 32 * 10, 400, 1360 + 32 * 10]
low_res_depth = torch.load(path_to_data + '/finger/low_res_height_map.pt',
                                    weights_only=False)
low_res_downsample = 4
depth_patch = low_res_depth[
                int(patch[0]/low_res_downsample):int(patch[1]/low_res_downsample),
                int(patch[2]/low_res_downsample):int(patch[3]/low_res_downsample)]
depth_patch = F.interpolate(
                depth_patch[None, None], size=(patch[1] - patch[0], patch[3] - patch[2]))
depth_patch = depth_patch.squeeze().numpy()
plt.figure()
plt.imshow(depth_patch, cmap='turbo')
plt.colorbar()



save_folder = "/media/Friday/Temporary/Clare/ICCP_result_storage/finger_from_low_res_results"
#np.save(save_folder + '/low_res_depth.npy', depth_patch)
for key, item in tqdm(experiment_dict.items()):
    save_name = save_folder + f'/{key}_depth.npy'
    if not os.path.exists(save_name):
        image = download_image(key, "depth")
        np.save(save_name, image)

    save_name = save_folder + f'/{key}_summed.npy'
    if not os.path.exists(save_name):
        image = download_image(key, "summed_warp")
        np.save(save_name, image)


noises = []
nums = []
for key, item in experiment_dict.items():
    noises.append(item["noise"][0])
    nums.append(len(item["cameras"])) 
plt.figure()
plt.scatter(nums, noises)
plt.show()


# another way 
summarized = {}
for key, item in experiment_dict.items():
    noise = item["noise"][0] 
    num = len(item["cameras"])
    if num not in summarized: 
        summarized[num] = {noise: 1}
    elif noise not in summarized[num]: 
        summarized[num][noise] = 1 
    else: 
        summarized[num][noise] = summarized[num][noise] + 1


# another way? 
unique_noise = np.sort(np.unique(noises))
unique_nums = np.sort(np.unique(nums)) 
organized = np.zeros((len(unique_nums), len(unique_noise)), dtype=np.int16) 
for i, num in enumerate(unique_nums): 
    for j, noise in enumerate(unique_noise): 
        if noise in summarized[num]: 
            organized[i, j] = summarized[num][noise]
# Create a figure and axis
fig, ax = plt.subplots()
# Hide axes
ax.axis('off')      
table = ax.table(organized, colLabels=unique_noise, rowLabels=unique_nums)
fig.subplots_adjust(bottom=0.0, top=0.9, )



# display things of interest 
c = 0
for key, item in experiment_dict.items():
    if len(item["cameras"]) != 5 or item["noise"][0] != 10:
        continue
    save_name = save_folder + f'/{key}_depth.npy'
    image = np.load(save_name)
    plt.figure()
    plt.imshow(image) 
    c += 1



fig, axes = plt.subplots(4, 5) 
noise_levels = [1, 5, 10, 15]
num_cameras = [48, 30, 20, 10, 5]
for i, nl in enumerate(noise_levels):
    for j, nc in enumerate(num_cameras):
        ax = axes[i,j] 
        ax.set_xticks([]) 
        ax.set_yticks([])
        for key, item in experiment_dict.items():
            if len(item["cameras"]) != nc or item["noise"][0] != nl:
                continue

            save_name = save_folder + f'/{key}_depth.npy'
            image = np.load(save_name)
            #ax = axes[i,j] 
            #ax.imshow(image[200:800, :600], cmap='turbo',
            #          clim=(-3.05, -0.6)) 
            
            ax.imshow(image, cmap='turbo') 
            #if nc == 48 and nl == 1:
            #    base_key = key
            #ax.set_xticks([]) 
            #ax.set_yticks([]) 
            break

        #break 
    #break 

fig.suptitle(f"{noise_levels}, {num_cameras}")
plt.tight_layout()



# just to more easily see what we have 
noises = []
nums = []
for key, item in experiment_dict.items():
    noise = item["noise"][0] 
    num = len(item["cameras"])
    noises.append(noise) 
    nums.append(num) 

plt.figure()
plt.scatter(nums, noises)


# compute a bunch of scores 
save_name = "analysis_results/finger_values.json"
analysis_results = load_dictionary(save_name)
base_key = "IC-13"
base_image = np.load(save_folder + f'/{base_key}_depth.npy')
for key, item in experiment_dict.items():
    if key in analysis_results:
        experiment_dict[key] = analysis_results[key]
        continue
    save_name = save_folder + f'/{key}_depth.npy'
    image = np.load(save_name) 
    ssim_score = ssim(base_image, image) 
    mse = np.mean(np.sqrt((base_image - image)**2)) 
    ssim_score_c = ssim(base_image[200:800, :600], image[200:800, :600]) 
    mse_c = np.mean(np.sqrt((base_image[200:800, :600] - image[200:800, :600])**2))
    item["mse"] = mse 
    item["ssim"] = ssim_score 
    item["mse_c"] = mse_c
    item["ssim_c"] = ssim_score_c

save_dictionary(experiment_dict, save_name)


noises = np.asarray([0, 1, 5, 10, 15, 20])
colors = ["gray", "black", "blue", "green", "red", "lightgray"]
used_colors = []
for key, item in experiment_dict.items():
    if key == base_key:
        continue
    noise = item["noise"][0] 
    idx = np.where(noises==noise)[0][0]
    color = colors[idx] 

    score = item["mse_c"] 
    num = len(item["cameras"])

    # hacky way to get the legend? 
    if color not in used_colors:
        label = noise 
    else: 
        label = None 
    used_colors.append(color) 

    plt.scatter(num, score*1e3, color=color, label=label)
plt.legend(title="Noise std dev")
plt.xlabel("# Cameras") 
plt.ylabel("Mean Error (um)")
plt.title("Errors for single trials")


plt.figure()
for noise, color in zip(noises, colors): 
    print(noise, color)
    first = True 
    for key, item in experiment_dict.items():
        if key == base_key:
            continue
        if item["noise"][0] != noise:
            continue 
        num_cameras = len(item["cameras"])
        count = 0 
        value = 0
        for key2, item2 in experiment_dict.items():
            if len(item2["cameras"]) == num_cameras and item2["noise"][0] == noise: 
                count += 1
                value += item2["mse"]

        value = value / count 
        if first:
            label = noise 
        else: 
            label = None
        first = False 
        plt.scatter([num_cameras], [value*1e3], color=color, label=label)
plt.legend(title="Noise std dev")
plt.xlabel("# Cameras") 
plt.ylabel("Mean Error (um)")
plt.title("errors averaged over trials")