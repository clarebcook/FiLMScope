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
ar_save_name = "finger_metrics.json"
ar_save_name = "finger_metrics_base_7542.json"
if os.path.exists(ar_save_name):
    analysis_results = load_dictionary(ar_save_name)
else:
    analysis_results = {}
base_key = "IC-13" 
base_image = np.load(result_folder1 + f"/{base_key}_depth.npy")


base_key = "7542" 
base_image = np.load(result_folder2 + f"/run_{base_key}_depth.npy")

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
    
    try:
        save_name = result_folder2 + f"/run_{key}_depth.npy" 
        mse, ssim_score = process_image(save_name)
        item["mse"] = mse 
        item["ssim"] = ssim_score 
        analysis_results[key] = item
    except Exception as e: 
        print(f"{key}, {e}")
    #break

save_dictionary(analysis_results, ar_save_name)




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
noises = []
num_cameras = [] 
all_mse = [] 
all_ssim = [] 
keys = [] 
codes = []

for key, item in analysis_results.items():
    noises.append(item["noise"][0])
    num_cameras.append(len(item["cameras"]))
    all_mse.append(item["mse"])
    all_ssim.append(item["ssim"])
    keys.append(key)

    code = get_code(item["cameras"])
    codes.append(code)

codes = np.asarray(codes) 
keys = np.asarray(keys) 
noises = np.asarray(noises)
num_cameras = np.asarray(num_cameras) 
all_mse = np.asarray(all_mse) 
all_ssim = np.asarray(all_ssim) 


indices0 = np.bitwise_or(noises == 1, noises == 0)

indices0 = noises == 0


plt.figure()
plt.scatter(num_cameras[indices0], all_ssim[indices0])





idx = np.bitwise_and(num_cameras <= 16, noises == 0)
for k in keys[idx]: 
    try:
        if "IC" in k:
            filename = result_folder1 + f"/{k}_depth.npy"
        else:
            filename = result_folder2 + f"/run_{k}_depth.npy"

        img = np.load(filename)
    except Exception as e:
        print(f"{k}, {e}")

    if k not in analysis_results:
        k = int(k)
    noise = analysis_results[k]["noise"][0]
    
    plt.figure()
    # plt.imshow(img[crop_values[0]:crop_values[1],
    #                     crop_values[2]:crop_values[3]],
    #                     clim=(-3.5, 0.5),
    #                     cmap='turbo') 

    plt.imshow(img[50:-50, 50:-50], cmap='turbo', clim=(-5, 1)) 
    plt.colorbar()
    plt.title(f"{k}, {noise}")
    plt.show()

    fig, axes = plt.subplots(6, 8, figsize=(2, 1.4))
    for i, j in np.ndindex(axes.shape):
        ax = axes[i, j] 
        ax.set_xticks([])
        ax.set_yticks([])
        number = (5 - i) + 6 * j
        
        if number in analysis_results[k]["cameras"]:
            ax.set_facecolor("red") 
        else:
            ax.set_facecolor("black") 
    axes[3, 3].set_facecolor('blue')
    fig.suptitle(f"{k}, {analysis_results[k]['mse']}")
    plt.show()






# get the images and look at the originals 
from filmscope.datasets import FSDataset
bk = 6792
config_dict = load_dictionary(result_folder2 + f'/run_{bk}_config.json')
info = config_dict["sample_info"]


# need to update a few things based 

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
    blank_filename=None, 
    noise=[0, 0]
)

ri = dataset.reference_image.cpu().squeeze()
plt.figure()
plt.imshow(ri[crop_values[0]:crop_values[1],
              crop_values[2]:crop_values[3]], cmap='gray') 
plt.show()


images = dataset.images.cpu().squeeze() 
for i, img in enumerate(images):
    plt.figure()
    plt.imshow(img[crop_values[0]:crop_values[1],
              crop_values[2]:crop_values[3]], cmap='gray') 
    plt.title(i) 
    plt.show()




bk = 7542
bk = 6174
cd2 = load_dictionary(result_folder2 + f'/run_{bk}_config.json')
info = config_dict["sample_info"]