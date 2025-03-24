from utility_functions import download_image, remove_global_tilt, get_reference_image
import numpy as np 
from matplotlib import pyplot as plt 
from filmscope.config import log_folder, path_to_data
from filmscope.util import load_dictionary, save_dictionary
import torch 
import torch.nn.functional as F
from tqdm import tqdm 
import os 
import math 
import scipy


import numpy as np
from skimage.draw import disk
from regions.core import PixCoord
from regions.shapes.circle import CirclePixelRegion


experiment_dict = load_dictionary(log_folder + '/stamp_runs_round2.json')
experiment_log_folder = log_folder + f'/stamp_results'

height_maps = None 
show = True 
iter = 140
for i, (key, item) in enumerate(experiment_dict.items()):
    #if len(item["cameras"]) != 25:
    #    continue
    #if i < len(experiment_dict) - 8:
    #    continue
    filename = experiment_log_folder + f"/run_{key}_iter_{iter}_depth.npy"
    height = np.load(filename) 
    if height_maps is None: 
        height_maps = np.zeros((len(experiment_dict), height.shape[0], height.shape[1]))
    height_maps[i] = height

    if show and item["noise"][0] == 0:

        fig, axes = plt.subplots(6, 8)
        for i, j in np.ndindex(axes.shape):
            ax = axes[i, j] 
            ax.set_xticks([])
            ax.set_yticks([])
            number = (5 - i) + 6 * j
            
            if number in item["cameras"]:
                ax.set_facecolor("red") 
            else:
                ax.set_facecolor("black") 
        axes[3, 3].set_facecolor('blue')
        plt.title(f"{key}, {i}, noise {item['noise'][0]}")
        plt.show()

        fig, axes = plt.subplots(1, 2) 
        axes[1].imshow(height, cmap='turbo') 
        height = remove_global_tilt(height) 
        height = height - np.min(height) 
        axes[0].imshow(height[100:300, 100:300], cmap='magma')
        plt.title(f"{key}, {i}")
        plt.show()

        filename = experiment_log_folder + f"/run_{key}_iter_{iter}_warp.npy"
        warp = np.load(filename) 
        plt.figure()
        plt.imshow(warp[100:300, 100:300], cmap='gray')
        plt.show()

        # plt.figure()
        # plt.plot(item["loss"])
        # plt.title(f"{key}, {i}")
        # plt.show()
    # break