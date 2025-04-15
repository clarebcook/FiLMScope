import torch
from filmscope.util import load_dictionary, save_dictionary
import torch.nn.functional as F
from filmscope.config import path_to_data
import numpy as np
from matplotlib import pyplot as plt 
import os 
from tqdm import tqdm 
from utility_functions import download_image, get_reference_image
from skimage.metrics import structural_similarity as ssim

# get dictionary from both rounds of experiments 
# going to manually specify the log folder 
folder = "/media/Friday/Temporary/Clare/ICCP_result_storage"
result_folder1 = folder + '/skull_700_results_v2'
result_folder2 = folder + '/round_2_results/skull_frame_700_results'

dict1 = load_dictionary(folder + '/skull_frame_700_v2.json')
dict2 = load_dictionary(folder + '/round_2_results/skull_frame_700_round2.json')

base_key = "IC-594"
base_image = np.load(result_folder1 + f'/{base_key}_depth.npy')

ref_image = get_reference_image(base_key)
mask = ref_image >= 25

# compute scores 
save_name = "skull_metrics.json"
if os.path.exists(save_name):
    analysis_results = load_dictionary(save_name)
else:
    analysis_results = {}

base_image = base_image * mask

def process_image(filename):
    image = np.load(filename) 
    image = image * mask
    ssim_score = ssim(base_image, image) 
    mse = np.mean(np.sqrt((base_image - image)**2)) 
    return mse, ssim_score

for key, item in tqdm(dict1.items()):
    if key in analysis_results:
        continue 

    try:
        save_name = result_folder1 + f"/{key}_depth.npy" 
        mse, ssim_score = process_image(save_name)
        item["mse"] = mse 
        item["ssim"] = ssim_score 
        analysis_results[key] = item
    except Exception as e:
       print(f"failed for {key}, {e}")
    #break

for key, item in tqdm(dict2.items()):
    if key in analysis_results:
        continue 

    try:
        save_name = result_folder2 + f"/run_{key}_depth.npy" 
        mse, ssim_score = process_image(save_name)
        item["mse"] = mse 
        item["ssim"] = ssim_score 
        analysis_results[key] = item
    except Exception as e:
        print(f"failed for {key}, {e}")

save_dictionary(analysis_results, save_name)




def get_code(cameras):
    binary_rep = [0] * 48 
    for num in cameras:
        binary_rep[num] = 1 
    binary_string = ''.join(map(str, binary_rep))
    decimal_number = int(binary_string, 2) 
    return decimal_number 

def decode(code):
    """
    Convert a unique code back to the original list of numbers between 1-48.
    
    Args:
        code (int): The decimal representation of a 48-bit binary number
        
    Returns:
        list: Original subset of numbers between 1-48
    """
    # Validate input range (0 to 2^48 - 1)
    if code < 0 or code >= (1 << 48):
        raise ValueError("Code must be between 0 and 2^48-1")
    
    # Convert to 48-bit binary string with leading zeros
    binary_str = bin(code)[2:].zfill(48)
    
    # Generate numbers from active bits
    numbers = [i for i, bit in enumerate(binary_str) if bit == '1']
    
    return numbers


# make orgnized array of noise, # cameras, mse, ssim 
noise = []
num_cameras = [] 
all_mse = [] 
all_ssim = [] 
keys = []
codes = []

for key, item in analysis_results.items():
    noise.append(item["noise"][0])
    num_cameras.append(len(item["cameras"]))
    all_mse.append(item["mse"])
    all_ssim.append(item["ssim"])
    keys.append(key)

    code = get_code(item["cameras"])
    codes.append(code)

noise = np.asarray(noise)
num_cameras = np.asarray(num_cameras) 
all_mse = np.asarray(all_mse) 
all_ssim = np.asarray(all_ssim) 
keys = np.asarray(keys)
codes = np.asarray(codes)





plt.figure()
plt.scatter(num_cameras, all_mse)

indices = noise == 0

plt.figure()
plt.scatter(num_cameras[indices], all_mse[indices])






code = get_code([20, 21, 26, 27])
code = get_code([6, 10, 20, 30, 34])

idx = np.where(num_cameras==48)[0] 
np.unique(codes[idx])
code = codes[idx][0]

idx = np.where(codes==code)[0]

k2 = keys[idx]
mn = np.min(base_image) 
mx = np.max(base_image)
for k in k2:
    if "IC" in k:
        depth = np.load(result_folder1 + f"/{k}_depth.npy") 
    else:
        depth = np.load(result_folder2 + f"/run_{k}_depth.npy")
    if k not in analysis_results:
        k = int(k)
    noise = analysis_results[k]["noise"][0] 
    plt.figure()
    depth[~mask] = np.nan
    plt.imshow(depth[50:-50, 50:-50], clim=(mn, mx), cmap='turbo')
    #plt.colorbar()
    plt.title(noise)
    ax = plt.gca()
    ax.set_xticks([]) 
    ax.set_yticks([])