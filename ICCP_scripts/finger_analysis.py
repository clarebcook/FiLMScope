from utility_functions import download_image
import numpy as np 
from matplotlib import pyplot as plt 
from filmscope.config import log_folder, path_to_data
from filmscope.util import load_dictionary
import torch 
import torch.nn.functional as F
from tqdm import tqdm 
import os 
import matplotlib


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


fig, axes = plt.subplots(4, 5) 
noise_levels = [1, 5, 10, 15]
num_cameras = [48, 40, 5, 4, 3]
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
            ax.imshow(image[200:800, :600], cmap='turbo',
                      clim=(-3.05, -0.6)) 
            if nc == 48 and nl == 1:
                base_key = key
            #ax.set_xticks([]) 
            #ax.set_yticks([]) 
            break

        #break 
    #break 

plt.tight_layout()



from skimage.metrics import structural_similarity as ssim
base_image = np.load(save_folder + f'/{base_key}_depth.npy')
scores = np.ones((3, 5)) * np.nan

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
            score = ssim(base_image, image) 
            scores[i, j] = score

            break


def mse(image0, image1):
    diff = image0 - image1 
    return np.mean(diff**2)


# use all in a plot 
cmap = matplotlib.cm.turbo
num_cameras = [48, 40, 30, 20, 10, 5, 4, 3]

plt.figure()
for i, nl in enumerate(noise_levels):
    for j, nc in enumerate(num_cameras):
        color = cmap((nc - 3) / 45)

        count = 0
        avg_score = 0
        for key, item in experiment_dict.items():
            if len(item["cameras"]) != nc or item["noise"][0] != nl:
                continue

            save_name = save_folder + f'/{key}_depth.npy'
            image = np.load(save_name)
            score = ssim(base_image[200:800, :600], image[200:800, :600]) 
            score = mse(base_image[200:800, :600], image[200:800, :600]) 
            avg_score += score 
            count += 1
        avg_score  = avg_score / count 
        plt.scatter([nl], [avg_score], color=color)



noises = []
nums = []
for key, item in experiment_dict.items():
    noise = item["noise"][0] 
    num = len(item["cameras"])
    noises.append(noise) 
    nums.append(num) 

plt.figure()
plt.scatter(nums, noises)