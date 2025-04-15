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



# define a bunch of functions
# from perplexity
def average_circle_value(image, location, radius):
    """
    Calculate the average pixel value within a circle centered at (center_x, center_y) with the given radius.

    Parameters:
    - image: 2D numpy array representing the image.
    - center_x: x-coordinate of the circle's center.
    - center_y: y-coordinate of the circle's center.
    - radius: Radius of the circle.

    Returns:
    - Average pixel value within the circle.
    """
    
    center_y, center_x = location

    # Ensure the center coordinates are floats for precise calculations
    center_x, center_y = float(center_x), float(center_y)
    
    # Generate a mask for the circle
    rr, cc = disk((int(center_y), int(center_x)), radius, shape=image.shape)
    
    # Calculate the average pixel value using the mask
    masked_image = image[rr, cc]
    average_value = np.mean(masked_image)
    
    return average_value

def get_sub_pix_value(image, location):
    low_pix = [int(i) for i in location]
    diff = [location[0] - low_pix[0], location[1] - low_pix[1]]
    value = image[low_pix[0], low_pix[1]] * (1 - diff[0]) * (1 - diff[1]) 
    value = value + image[low_pix[0] + 1, low_pix[1]] * diff[0] * (1 - diff[1]) 
    value = value + image[low_pix[0], low_pix[1] + 1] * (1 - diff[0]) * diff[1] 
    value = value + image[low_pix[0] + 1, low_pix[1] + 1] * diff[0] * diff[1] 
    return value 

def get_line(start_pixel, end_pixel, image, half_thickness=0,
             return_indices=False):
    main_dir = (end_pixel - start_pixel) / np.linalg.norm(end_pixel - start_pixel)
    perp_dir = np.asarray([-main_dir[1], main_dir[0]] )
    num_samples = int(np.linalg.norm(end_pixel - start_pixel))
    
    line_image = np.zeros((num_samples, 2 * half_thickness + 1))
    line_indices = np.zeros((num_samples, 2))
    for i in range(num_samples):
        center_pixel = start_pixel + main_dir * i
        steps = np.tile(np.arange(-half_thickness, half_thickness + 1)[None], [2, 1]) * perp_dir[:, None]
        pixels = steps + center_pixel[:, None]
        
        for j, pixel in enumerate(pixels.T):
            line_image[i, j] = get_sub_pix_value(image, [pixel[0], pixel[1]])
        
        line_indices[i] = center_pixel
          
    line = np.mean(line_image, axis=1)

    # this is x and y
    # we're keeping it in pixels for now
    # line[:, 0] = line[:, 0] - line[0, 0]
    # line[:, 1] = line[:, 1] - line[0, 1] 
    # lat_vals = np.sqrt(line[:, 0]**2 + line[:, 1]**2)
    lat_vals = np.arange(line.shape[0])
            
    if return_indices:
        return line_image, line, lat_vals, line_indices 
    return line_image, line, lat_vals 

def get_peak_trough_diffs(lat_vals, peaks, troughs, heights):
    peak_diffs = lat_vals[peaks][1:] - lat_vals[peaks][:-1]
    trough_diffs = lat_vals[troughs][1:] - lat_vals[troughs][:-1]
    
    # find differences between peaks and adjacent troughs 
    peak_lat_vals = lat_vals[peaks]
    trough_lat_vals = lat_vals[troughs]

    height_diffs = [] 
    for i, peak_loc in enumerate(peak_lat_vals):
        # find the closest values 
        close_indices = np.argsort(np.abs(trough_lat_vals - peak_loc))
        idx0 = close_indices[0] 
        idx1 = close_indices[1]

        height_diffs.append([heights[peaks][i] - heights[troughs][idx0], i, idx0])

        # this would be negative if the two closest troughs are on either side of the peak
        if (peak_lat_vals[i] - trough_lat_vals[idx0]) * (peak_lat_vals[i] - trough_lat_vals[idx1]) > 0:
            continue
        height_diffs.append([heights[peaks][i] - heights[troughs][idx1], i, idx1])

    height_diffs = np.asarray(height_diffs)
    return height_diffs, peak_diffs, trough_diffs

def rotate_line(heights, lat_vals):
    #return heights, lat_vals
    
    m, b = np.polyfit(lat_vals, heights, 1)

    theta = -math.atan(m)

    M = np.asarray([
        [math.cos(theta), -math.sin(theta)],
        [math.sin(theta), math.cos(theta)]
    ])

    points = np.stack((lat_vals, heights))
    points = np.matmul(M, points)
    lat_vals = points[0]
    heights = points[1]
    
    return heights, lat_vals

def clean_line(peaks, troughs, peak_diffs, trough_diffs, 
               line_heights, lat_vals, estimated_separation=11):
    k = 0
    while True:
        peak_diffs_m = np.stack((peak_diffs, np.ones_like(peak_diffs)), axis=1)
        trough_diffs_m = np.stack((trough_diffs, np.zeros_like(trough_diffs)), axis=1)
        all_diffs = np.concatenate((peak_diffs_m, trough_diffs_m), axis=0)

        indices = np.argsort(all_diffs[:, 0])
        all_diffs_sorted = all_diffs[indices]

        diff = all_diffs_sorted[k]

        if diff[0] > estimated_separation / 2: 
            break

        peak = diff[1] == 0 # this means there was an extranneous peak, with two troughs around
        if peak:
            bools0 = peaks
            bools1 = troughs
        else:
            bools1 = peaks
            bools0 = troughs
        indices0 = np.where(bools0)[0] 
        indices1 = np.where(bools1)[0]
        
        # get the peak/trough, and the two around it 
        # first the two surrounding 
        i0 = indices[k]
        if peak:
            i0 = i0 - peak_diffs_m.shape[0] 
        idx0 = indices1[i0] 
        idx1 = indices1[i0 + 1]
        
        # then the center
        loc = np.where(np.logical_and(indices0 > idx0, indices0 < idx1))[0][0]
        center_idx = indices0[loc]
        
        # if we can't do this test... we'll just throw it out ? 
        if loc > 0 and loc < len(indices0) - 1:
            pre_peak = indices0[loc - 1]
            post_peak = indices0[loc + 1]
            if abs(lat_vals[post_peak] - lat_vals[pre_peak]) > 1.5 * estimated_separation:
                k = k + 1
                continue
        
        # get rid of this peak/trough 
        h0 = line_heights[idx0] 
        h1 = line_heights[idx1]
        
        if h0 < h1 and peak:
            remove_idx = idx1
        elif h0 > h1 and not peak:
            remove_idx = idx1
        else:
            remove_idx = idx0 
        
        troughs[remove_idx] = False
        peaks[remove_idx] = False
        troughs[center_idx] = False
        peaks[center_idx] = False
        
        height_diffs, peak_diffs, trough_diffs = get_peak_trough_diffs(lat_vals, peaks, troughs, line_heights)
        
    return peaks, troughs 

def process_line(heights, lat_vals, show=True, clean=True):
    # I'm fitting a line but probably not necessary... or just for display
    #m, b = np.polyfit(lat_vals, heights, 1)
    #heights = heights - lat_vals * m + b
    #heights, lat_vals = rotate_line(heights, lat_vals)
    heights = heights - np.min(heights)
    
    # find peaks and troughs
    diff = np.zeros(len(heights))
    diff[:-1] = heights[1:] - heights[:-1]
    
    peak = np.zeros(len(heights), dtype=bool)
    troughs = peak.copy()
    peak[1:] = np.logical_and(diff[1:] <= 0, diff[:-1] > 0)
    troughs[1:] = np.logical_and(diff[1:] > 0, diff[:-1] <= 0)

    # better way to enforce this, but first and last points can't be peaks or troughs
    peak[0] = False
    peak[-1] = False
    troughs[0] = False 
    troughs[-1] = False
    
    if show:
        plt.figure()
        plt.plot(lat_vals, heights, '.-')
        plt.plot(lat_vals[peak], heights[peak], '.')
        plt.plot(lat_vals[troughs], heights[troughs], '.')
        
    peak_diffs = lat_vals[peak][1:] - lat_vals[peak][:-1]
    trough_diffs = lat_vals[troughs][1:] - lat_vals[troughs][:-1]
    
    # find differences between peaks and adjacent troughs 
    peaks = peak
    height_diffs, peak_diffs, trough_diffs = get_peak_trough_diffs(lat_vals, peak, troughs, heights)
    if clean:
        peaks, troughs = clean_line(peak, troughs, peak_diffs, trough_diffs,
                                    heights, lat_vals)
    
    return peaks, troughs

def get_peaks_troughs(key, line_starts, line_ends, show=True, show_lines=0, clean=True):
    height = np.load(save_folder + f'/{key}_depth.npy')
    height = remove_global_tilt(height)
    height = height - np.min(height)
    # process all the lines 
    peak_heights = [] 
    trough_heights = [] 
    peak_indices = None
    trough_indices = None

    half_thickness = 1
    for line_number, (start_pixel, end_pixel) in enumerate(tqdm(zip(line_starts, line_ends))):
        line_image, line, lat_vals = get_line(start_pixel, end_pixel, ref_image, half_thickness=half_thickness)
        height_image, line_heights, lat_vals, line_indices = get_line(
            start_pixel, end_pixel, height, half_thickness=half_thickness, return_indices=True)
        peaks, troughs = process_line(line_heights, lat_vals, show=False, clean=clean)
        
        if line_number < show_lines:
            fig, (ax1, ax2) = plt.subplots(2, 1)
            fig.suptitle(f"Line {line_number}")
            ax1.imshow(height_image.T, cmap='magma')
            ax1.set_xticks([])
            ax1.set_yticks([])
            ax1.set_aspect(25 / (2 * half_thickness + 1))
            ax2.plot(lat_vals, line_heights)
            ax2.plot(lat_vals[peaks], line_heights[peaks], '.', color='red')
            ax2.plot(lat_vals[troughs], line_heights[troughs], '.', color='blue')
            ax2.set_xlim(np.min(lat_vals), np.max(lat_vals))
            plt.tight_layout()
            plt.show()
        peak_heights = np.concatenate((peak_heights, line_heights[peaks]))
        trough_heights = np.concatenate((trough_heights, line_heights[troughs]))
        if peak_indices is None:
            peak_indices = line_indices[peaks] 
            trough_indices = line_indices[troughs]
        else:
            peak_indices = np.concatenate((peak_indices, line_indices[peaks]))
            trough_indices = np.concatenate((trough_indices, line_indices[troughs]))

        #if line_number > 7:
        #    break 

    peak_heights *= 1e3 
    trough_heights *= 1e3 

    if show:
        plt.figure()
        plt.hist(peak_heights, color='red', alpha=0.5, density=True, bins=45, label="Peaks")
        plt.hist(trough_heights, color='blue', alpha=0.5, density=True, bins=45, label="Troughs")

        plt.axvline(x=np.mean(peak_heights), color='black', linestyle='--')
        plt.axvline(x=np.mean(trough_heights), color='black', linestyle='--')

        plt.xlabel("Axial Position (um)")
        plt.ylabel("Density") 

        plt.legend()

        mu_t, std_t = scipy.stats.norm.fit(trough_heights)
        xmin, xmax = (np.min(trough_heights), np.max(trough_heights))
        x = np.linspace(xmin, xmax, 100)
        p = scipy.stats.norm.pdf(x, mu_t, std_t)
        plt.plot(x, p, 'k', linewidth=2, color='blue')
        plt.axvline(x=mu_t, color='blue', linestyle='--')

        mu_p, std_p = scipy.stats.norm.fit(peak_heights)
        xmin, xmax = (np.min(peak_heights), np.max(peak_heights))
        x = np.linspace(xmin, xmax, 100)
        p = scipy.stats.norm.pdf(x, mu_p, std_p)
        plt.plot(x, p, 'k', linewidth=2, color='red')
        plt.axvline(x=mu_p, color='red', linestyle='--')

        plt.title("peak std: {:.0f} um, trough std: {:.0f} um\n separation: {:.0f} um".format(std_p, std_t, mu_p - mu_t))

    return peaks, troughs, peak_indices, trough_indices

def hist_peaks_troughs(peak_values, trough_values):
    p2 = peak_values * 1e3 
    t2 = trough_values * 1e3

    plt.figure()
    plt.hist(p2, color='red', alpha=0.5, density=True, bins=45, label="Peaks")
    plt.hist(t2, color='blue', alpha=0.5, density=True, bins=45, label="Troughs")

    plt.axvline(x=np.mean(p2), color='black', linestyle='--')
    plt.axvline(x=np.mean(t2), color='black', linestyle='--')

    plt.xlabel("Axial Position (um)")
    plt.ylabel("Density") 

    plt.legend()
    mu_t, std_t = scipy.stats.norm.fit(t2)
    xmin, xmax = (np.min(t2), np.max(t2))
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mu_t, std_t)
    #plt.plot(x, p, 'k', linewidth=2, color='blue')
    plt.axvline(x=mu_t, color='blue', linestyle='--')

    mu_p, std_p = scipy.stats.norm.fit(p2)
    xmin, xmax = (np.min(p2), np.max(p2))
    x = np.linspace(xmin, xmax, 100)
    p = scipy.stats.norm.pdf(x, mu_p, std_p)
    #plt.plot(x, p, 'k', linewidth=2, color='red')
    plt.axvline(x=mu_p, color='red', linestyle='--')

    plt.title("peak std: {:.0f} um, trough std: {:.0f} um\n separation: {:.0f} um".format(std_p, std_t, mu_p - mu_t))

def load_height_map(key):
    height = np.load(save_folder + f'/{key}_depth.npy')
    height = remove_global_tilt(height)
    height = height - np.min(height)
    return height 

def display_height_map(height, peak_indices=None, 
                       trough_indices=None, crop=[100, 300, 100, 300],
                       cmap='magma', clim=None, s=2): 
    if clim is None:
        minc = np.min(height[crop[0]:crop[1], crop[2]:crop[3]])
        maxc = np.max(height[crop[0]:crop[1], crop[2]:crop[3]])
        clim = (minc, maxc) 
    plt.figure()
    plt.imshow(height, cmap=cmap, clim=clim)
    plt.colorbar() 
    if peak_indices is not None: 
        plt.scatter(peak_indices[:, 1], peak_indices[:, 0], color='black', s=s)
    if trough_indices is not None: 
        plt.scatter(trough_indices[:, 1], trough_indices[:, 0], color='blue', s=s)
    ax = plt.gca() 
    ax.set_xlim(crop[2], crop[3]) 
    ax.set_ylim(crop[1], crop[0])




def compute_indices():
    # this is borrowed from other paper supplement
    ## pick lines (these were chosen manually from the reference image )
    line0_start = [596, 34]#[600, 98] 
    line0_end = [80, 68]#[138, 129]
    last_line_start = [607, 568]#[598, 384]
    num_lines = 50
    line_dir_main = 1
    line_dir_sec = 0

    spacing = (last_line_start[line_dir_main] - line0_start[line_dir_main]) / (num_lines - 1)

    line_starts = np.zeros((num_lines, 2)) 
    line_starts[:] = line0_start
    line_starts[:, line_dir_main] = np.arange(num_lines) * spacing + line0_start[line_dir_main]

    line_ends = np.zeros((num_lines, 2))
    line_ends[:] = line0_end
    line_ends[:, line_dir_main] = np.arange(num_lines) * spacing + line0_end[line_dir_main]

    plt.figure()
    plt.imshow(ref_image, cmap='gray')
    for s, e in zip(line_starts, line_ends):
        plt.plot([s[1], e[1]], [s[0], e[0]], color='red')

    peaks, troughs, peak_indices, trough_indices = get_peaks_troughs(base_key, line_starts, line_ends)

    # for each peak, find the six closest peaks 
    # and the midpoints will be troughs 
    trough_indices = None
    for i, value in enumerate(peak_indices):
        dist = np.linalg.norm(peak_indices - value, axis=1) 
        indices = np.argsort(dist)
        t_vals = (value + peak_indices[indices[1:7]]) / 2
        if trough_indices is None:
            trough_indices = t_vals 
        else:
            trough_indices = np.concatenate((trough_indices, t_vals))

    return peak_indices, trough_indices




# for now using same indices for all
# so grab those from the original image ? 
trough_save_name = "../analysis_results/stamp_trough_indices.npy"
peak_save_name = "../analysis_results/stamp_peak_indices.npy"
peak_indices = np.load(peak_save_name) 
trough_indices = np.load(trough_save_name)







