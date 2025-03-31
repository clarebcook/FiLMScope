import torch
from filmscope.util import load_dictionary, save_dictionary
import torch.nn.functional as F
from filmscope.config import path_to_data
import numpy as np
from matplotlib import pyplot as plt 
import os 
from tqdm import tqdm 
from skimage.metrics import structural_similarity as ssim

# get dictionary from both rounds of experiments 
# going to manually specify the log folder 
folder = "/media/Friday/Temporary/Clare/ICCP_result_storage"
result_folder1 = folder + '/finger_from_low_res_results'
result_folder2 = folder + '/round_2_results/finger_results'

dict1 = load_dictionary(folder + '/finger_from_low_res.json')
dict2 = load_dictionary(folder + '/round_2_results/finger_from_low_res.json')


# load in the low res map 
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



# compute scores 
save_name = "finger_metrics.json"
if os.path.exists(save_name):
    analysis_results = load_dictionary(save_name)
else:
    analysis_results = {}
base_key = "IC-13" 
base_image = np.load(result_folder1 + f"/{base_key}_depth.npy")
crop_values = (200, 800, 0, 600)

base_image = base_image[crop_values[0]:crop_values[1],
                        crop_values[2]:crop_values[3]]

def process_image(filename):
    image = np.load(filename) 
    image = image[crop_values[0]:crop_values[1], crop_values[2]:crop_values[3]]
    ssim_score = ssim(base_image, image) 
    mse = np.mean(np.sqrt((base_image - image)**2)) 
    return mse, ssim_score

for key, item in tqdm(dict1.items()):
    if key in analysis_results:
        continue 

    save_name = result_folder1 + f"/{key}_depth.npy" 
    mse, ssim_score = process_image(save_name)
    item["mse"] = mse 
    item["ssim"] = ssim_score 
    analysis_results[key] = item
    #break

for key, item in tqdm(dict2.items()):
    if key in analysis_results:
        continue 

    save_name = result_folder2 + f"/run_{key}_depth.npy" 
    mse, ssim_score = process_image(save_name)
    item["mse"] = mse 
    item["ssim"] = ssim_score 
    analysis_results[key] = item
    #break

save_dictionary(analysis_results, save_name)



# make orgnized array of noise, # cameras, mse, ssim 
noise = []
num_cameras = [] 
all_mse = [] 
all_ssim = [] 
for key, item in analysis_results.items():
    noise.append(item["noise"][0])
    num_cameras.append(len(item["cameras"]))
    all_mse.append(item["mse"])
    all_ssim.append(item["ssim"])
noise = np.asarray(noise)
num_cameras = np.asarray(num_cameras) 
all_mse = np.asarray(all_mse) 
all_ssim = np.asarray(all_ssim) 


indices0 = np.bitwise_or(noise == 1, noise == 0)


plt.figure()
plt.scatter(num_cameras[indices0], all_mse[indices0])

plt.figure()
plt.scatter(all_mse, all_ssim)