from utility_functions import download_image
import numpy as np 
from matplotlib import pyplot as plt 
from filmscope.config import log_folder, path_to_data
from filmscope.util import load_dictionary
import torch 
import torch.nn.functional as F
from tqdm import tqdm 
import os 

# run_id = "IC-42"
# image_type = "depth" 
# image = download_image(run_id, image_type)

experiment_dict = load_dictionary(log_folder + '/finger_from_low_res.json')


# for key, item in experiment_dict.items():
#     if len(item["cameras"]) != 48 or item["noise"][0] != 10:
#         continue

#     image, run = download_image(key, "depth", return_run=True)
#     plt.figure()
#     plt.imshow(image, cmap='turbo')
#     plt.title(f"{len(item['cameras'])} cameras, noise {item['noise'][0]}")
#     plt.show() 

#     break 


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
    if os.path.exists(save_name):
        continue
    image = download_image(key, "depth")
    np.save(save_name, image)


# diff = image - depth_patch 
# plt.figure()
# plt.imshow(diff, cmap='turbo', clim=(-0.3, 0.3)) 
# plt.colorbar() 


fig, axes = plt.subplots(3, 5) 
noise_levels = [1, 5, 10]
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
            ax.imshow(image, cmap='turbo') 
            #ax.set_xticks([]) 
            #ax.set_yticks([]) 
            break

        #break 
    #break 

plt.tight_layout()



from skimage.metrics import structural_similarity as ssim