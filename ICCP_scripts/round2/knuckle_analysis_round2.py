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
result_folder1 = folder + '/knuckle_frame_438_results'
result_folder2 = folder + '/round_2_results/knuckle_frame_438_results'

dict1 = load_dictionary(folder + '/knuckle_frame_438.json')
dict2 = load_dictionary(folder + '/round_2_results/knuckle_frame_438.json')

base_key = "IC-61"
base_image = np.load(result_folder1 + f'/{base_key}_depth.npy')



# compute scores 
save_name = "knuckle_metrics.json"
if os.path.exists(save_name):
    analysis_results = load_dictionary(save_name)
else:
    analysis_results = {}

crop_values = (100, -50, 100, -50)

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
    #break

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



indices = np.bitwise_and(all_mse < 0.5, num_cameras == 5)


for key in keys[indices]:
    if key not in analysis_results:
        key = int(key)
    cameras = analysis_results[key]["cameras"]
    fig, axes = plt.subplots(6, 8, figsize=(2, 1.4))
    for i, j in np.ndindex(axes.shape):
        ax = axes[i, j] 
        ax.set_xticks([])
        ax.set_yticks([])
        number = (5 - i) + 6 * j
        
        if number in cameras:
            ax.set_facecolor("red") 
        else:
            ax.set_facecolor("black") 
    axes[3, 3].set_facecolor('blue')
    fig.suptitle(f"{key}, {analysis_results[key]['mse']}")
    plt.show()



key1 = "4859"
key1_index = np.where(keys == key1)[0][0] 
subset_code = codes[key1_index]

matching = codes == subset_code
print(num_cameras[matching])
print(keys[matching])
print(noise[matching])


mn = np.min(base_image) 
mx = np.max(base_image)
for key in keys[matching]:
    if key not in analysis_results:
        key = int(key)
    depth = np.load(result_folder2 + f"/run_{key}_depth.npy") 
    plt.figure()
    plt.imshow(depth[100:-50, 100:-50], clim=(mn, mx), cmap='turbo') 
    plt.title(analysis_results[key]["noise"][0])
    plt.colorbar()
    ax = plt.gca()
    ax.set_xticks([]) 
    ax.set_yticks([])
    plt.show() 



numbers = [21, 27, 20, 26]


def display_cam_locs(numbers):
    fig, axes = plt.subplots(6, 8, figsize=(2, 1.4))
    for i, j in np.ndindex(axes.shape):
        ax = axes[i, j] 
        ax.set_xticks([])
        ax.set_yticks([])
        number = (5 - i) + 6 * j
        if number in numbers:
            color = 'red' 
        else:
            color = 'black'
        if number == 20:
            color = 'blue'
        #color = cmap(m[number])
        ax.set_facecolor(color)




code = 73995745428272
cameras = decode(code) 
display_cam_locs(cameras)







# a lot of temporary nonsense below here 
idx = np.where(num_cameras == 20)[0]
for code in np.unique(codes[idx]):
    numbers = decode(code) 
    display_cam_locs(numbers)
    plt.title(code)


code = codes[idx][0] 
idx2 = np.where(codes==code)[0] 
k2 = keys[idx2]

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
    plt.imshow(depth[100:-50, 100:-50], clim=(mn, mx), cmap='turbo')
    plt.title(noise)
    ax = plt.gca()
    ax.set_xticks([]) 
    ax.set_yticks([])
