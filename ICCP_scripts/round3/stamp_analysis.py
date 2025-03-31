from filmscope.util import load_dictionary 
from filmscope.config import log_folder 
from matplotlib import pyplot as plt 
import numpy as np 
from utility_functions import remove_global_tilt

from stamp_utility import peak_indices, trough_indices, average_circle_value, display_height_map, hist_peaks_troughs


experiment_dict = load_dictionary(log_folder + '/stamp_runs_round3.json')
experiment_log_folder = log_folder + f'/stamp_results_r3'





height_maps = None 
show = True 
iter = 100
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

    if show and item["noise"][0] == 30:

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
        axes[1, 3].set_facecolor('blue')
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
    break



h2 = height #height[50:-50, 50:-50]
h2 = remove_global_tilt(h2) 
h2 = h2 - np.min(h2) 
plt.figure()
plt.imshow(h2, cmap='magma')


display_height_map(height, peak_indices, trough_indices)


circle_radius_p = 1
circle_radius_t = 1
peak_values = np.asarray([average_circle_value(height, loc, circle_radius_p) for loc in peak_indices])
trough_values = np.asarray([average_circle_value(height, loc, circle_radius_t) for loc in trough_indices])
hist_peaks_troughs(trough_values, peak_values)




# run through many? 
height_maps = None 
show = True 
iter = 100
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
        axes[1, 3].set_facecolor('blue')
        plt.title(f"{key}, {i}, noise {item['noise'][0]}")
        plt.show()


        height = remove_global_tilt(height) 
        height = height - np.min(height) 
        peak_values = np.asarray([average_circle_value(height, loc, circle_radius_p) for loc in peak_indices])
        trough_values = np.asarray([average_circle_value(height, loc, circle_radius_t) for loc in trough_indices])
        hist_peaks_troughs(trough_values, peak_values)

        display_height_map(height, peak_indices, trough_indices)


    #break



