from utility_functions import download_image, remove_global_tilt, get_reference_image
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

experiment_dict = load_dictionary(log_folder + '/stamp_runs_v2.json')


save_folder = "/media/Friday/Temporary/Clare/ICCP_result_storage/stamp_results"
#np.save(save_folder + '/low_res_depth.npy', depth_patch)
for key, item in tqdm(experiment_dict.items()):
    save_name = save_folder + f'/{key}_depth.npy'
    if os.path.exists(save_name):
        continue
    image = download_image(key, "depth")
    np.save(save_name, image)


# testing out some processing approaches 
base_key = "IC-284"
base_image = np.load(save_folder + f'/{base_key}_depth.npy')[100:-100, 100:-100]

plt.figure()
plt.imshow(base_image)

image = remove_global_tilt(base_image)
plt.figure()
plt.imshow(image.T[:200, :200]) 
ax = plt.gca()
ax.set_aspect("equal") 



# testing out some processing approaches 
key = "IC-289"
image = np.load(save_folder + f'/{key}_depth.npy')[100:-100, 100:-100]

plt.figure()
plt.imshow(image)

image = remove_global_tilt(image)
plt.figure()
plt.imshow(image[:200, :200]) 
ax = plt.gca()
ax.set_aspect("equal") 





a = download_image(base_key, "summed_warp")
a = a[100:-100, 100:-100]
plt.imshow(a, cmap='gray')

a = get_reference_image(base_key) 
fig = plt.figure(dpi=1000)
plt.imshow(a, cmap='gray') 