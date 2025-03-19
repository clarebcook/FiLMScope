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

experiment_dict = load_dictionary(log_folder + '/knuckle_frame_438.json')

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


save_folder = "/media/Friday/Temporary/Clare/ICCP_result_storage/knuckle_frame_438_results"

for key, item in tqdm(experiment_dict.items()):
    save_name = save_folder + f'/{key}_depth.npy'
    if not os.path.exists(save_name):
        image = download_image(key, "depth")
        np.save(save_name, image)

    save_name = save_folder + f'/{key}_summed.npy'
    if not os.path.exists(save_name):
        image = download_image(key, "summed_warp")
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
            ax.imshow(image[50:-50, 50:-100], cmap='turbo') 
            #ax.set_xticks([]) 
            #ax.set_yticks([]) 

            if nc == 48 and nl == 1:
                base_key = key

            break

        #break 
    #break 

plt.tight_layout()



from skimage.metrics import structural_similarity as ssim

base_image = np.load(save_folder + f'/{base_key}_depth.npy')



noise_levels = [1, 5, 10]
num_cameras = [48, 30, 20, 10, 5]

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

        #break 

    #break 


# for key, item in experiment_dict.items():
#     if len(item["cameras"]) < 10:
#         print("here")
#         image, run = download_image(key, "depth", return_run=True)
#         loss = run["reconstruction/loss/unsup"].fetch_values() 
#         loss = loss["value"].values 
#         plt.figure()
#         plt.plot(loss) 
#         plt.show()

#         #break 


for key, item in experiment_dict.items():
    if len(item["cameras"]) == 10 and item["noise"][0] == 5: 
        save_name = save_folder + f'/{key}_depth.npy'
        image = np.load(save_name) 

        plt.figure()
        plt.imshow(image, cmap='turbo') 
        plt.title(len(item["cameras"]))
        plt.show()