from filmscope.reconstruction import RunManager, generate_config_dict 
from filmscope.util import get_timestamp
from filmscope.config import log_folder
from tqdm import tqdm
import os
import torch
from matplotlib import pyplot as plt

# select the name of a sample previously saved using "save_new_sample.ipynb",
# the gpu number, and whether or not to log with neptune.ai
#sample_name = "rat_skull_12_11"
#sample_name = "posterior1"
sample_name = "fluoro_bright"#_calib2"
gpu_number = "0"
use_neptune = False

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# find a better way to use all the available numbers...
image_numbers = [13, 14, 15, 16, 
                 19, 20, 21, 22,
                 25, 26, 27, 28,
                 31, 32, 33, 34]

image_numbers = [13, 15, 20, 22, 25, 27, 32, 34]
image_numbers = [14, 16, 19, 21, 26, 28, 31, 33]

config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=1,
                                   camera_set="custom", use_neptune=use_neptune,
                                   custom_image_numbers=image_numbers,
                                   crop_name="fluoro_patch",
                                   run_args={"iters": 1000, "batch_size": 16, "num_depths": 64,
                                             "display_freq": 50, "unet_layers": 4,
                                             "unet_layer_channels": [8, 16, 16, 16, 16],
                                             "lr": 1e-3},
                                   loss_weights={"smooth": 0.35})
run_manager = RunManager(config_dict)

# perform reconstruction
iters = config_dict["run_args"]["iters"]
losses = []
for i in tqdm(range(iters)):
    log = (i % config_dict["run_args"]["display_freq"] == 0) or (i == iters - 1)
    mask_images, warp_images, numbers, outputs, loss_values = run_manager.run_epoch(
        i, log=(log and config_dict["use_neptune"]))
    losses.append(float(loss_values["total"]))

    # this section can be edited to change what is recorded
    if log and not config_dict["use_neptune"]:
        depth = outputs["depth"].detach().cpu().squeeze()
        plt.figure()
        plt.imshow(depth)
        plt.title(f"iter {i}")
        plt.colorbar()
        plt.show()

        img = outputs["warped_imgs"].detach().cpu().squeeze()
        img = torch.mean(img, axis=0)
        plt.figure()
        plt.imshow(img, cmap='gray') 
        plt.show()

        plt.figure()
        plt.plot(losses)
        plt.show()

    #if i == 900:
    #    save_name = "/media/Friday/Temporary/Clare/eye_cropped_posterior_depth.npy"
run_manager.end()



# import numpy as np
# import cv2
# from filmscope.config import path_to_data
# from filmscope.util import load_image_set

# filename = config_dict["sample_info"]["image_filename"]

# #filename = "anterior_with_tool_20241208_143816_532.nc"
# save_name = path_to_data + '/' + filename[:-3] + "_hm.npy"
# depth = outputs["depth"].cpu().detach().squeeze().numpy()
# depth = cv2.resize(depth, (4096, 3120))
# np.save(save_name, depth)

# f = path_to_data + '/' + filename
# image_numbers = config_dict["sample_info"]["image_numbers"]
# folder_name = f[:-3] + "_images"
# if not os.path.exists(folder_name):
#     os.mkdir(folder_name)
# images = load_image_set(f, image_numbers=image_numbers, debayer=False)

# downsample = 4
# for key, image in images.items():
#     image = image[::downsample, ::downsample]
#     from PIL import Image 
#     image = Image.fromarray(image) 
#     image.save(folder_name + f"/image_{key}.tiff")
