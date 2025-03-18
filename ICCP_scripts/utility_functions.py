import numpy as np
import neptune 
from filmscope.config import neptune_project as project 
from filmscope.config import neptune_api_token as api_token
from filmscope.config import path_to_data
from PIL import Image
from filmscope.util import get_timestamp
from filmscope.calibration import CalibrationInfoManager
import os 
import shutil

def download_image(run_id, image_type, 
                   # download_filename="temp_download_folder/image.png",
                   return_run=False):
    run = neptune.init_run(
        with_id=run_id, mode="read-only", project=project, api_token=api_token
    )

    download_folder=f"temp_download_folder_{get_timestamp()}"

    # image_type = "depth"
    run[f"reconstruction/values/{image_type}"].download_last(download_folder)
    image_filename = download_folder + '/' + os.listdir(download_folder)[0]
    image = np.asarray(Image.open(image_filename))

    minv = run[f"reconstruction/values/{image_type}_min"].fetch_last()
    maxv = run[f"reconstruction/values/{image_type}_max"].fetch_last()

    image = image / 255 * (maxv - minv) + minv 

    shutil.rmtree(download_folder)

    if not return_run:
        return image

    else:
        return image, run


def remove_global_tilt(height_map):#, show=False):
    height_map = height_map.copy()
    dim0 = np.mean(height_map, axis=0)
    # plt.plot(dim0)

    slope0, intercept0 = np.polyfit(np.arange(len(dim0)), dim0, 1)

    # plt.plot(np.arange(len(dim0)) * slope0 + intercept0)

    for i, row in enumerate(height_map):
        height_map[i] = row - np.arange(len(row)) * slope0 

    dim1 = np.mean(height_map, axis=1) 
    slope1, intercept1 = np.polyfit(np.arange(len(dim1)), dim1, 1)

    # plt.figure()
    # plt.plot(dim1) 
    # plt.plot(np.arange(len(dim1)) * slope1 + intercept1) 

    for j in range(height_map.shape[1]):
        col = height_map[:, j] 
        height_map[:, j] = col - np.arange(len(col)) * slope1 

    return height_map

def count_needed_runs(experiment_dict, repeats, all_noise_stds, num_cameras):
    partials = []
    partial_cameras = []
    # and ideally if there's an incomplete set 
    # we'd pick up there 
    # num_cameras = 4 
    tracked_ids = []
    completed = 0
    for id, item in experiment_dict.items():
        #print(id, item) 
        if id in tracked_ids:
            continue 

        tracked_ids.append(id) 
        if len(item["cameras"]) != num_cameras:
            #print(len(item["cameras"]))
            continue 

        cameras = np.asarray(item["cameras"])
        completed_noise = [item["noise"][0]]
        for id2, item2 in experiment_dict.items():
            if id2 in tracked_ids:
                continue 

            #print(id2, item2) 

            cameras2 = item2["cameras"] 
            if len(cameras2) != num_cameras:
                continue 

            intersect = np.intersect1d(cameras, cameras2) 
            if len(intersect) != num_cameras:
                continue 

            # now we know these have the same camera set 
            completed_noise.append(item2["noise"][0])
            tracked_ids.append(id2)

        needed_noise = [i for i in all_noise_stds if i not in completed_noise]
        if len(needed_noise) == 0:
            completed +=1 
        else:
            partials.append(needed_noise)
            partial_cameras.append(cameras.tolist())

    needed_repeats = repeats - len(partials) - completed
    for r in range(needed_repeats):
        partials.append(all_noise_stds)
        partial_cameras.append(None)
    return partials, partial_cameras



def get_reference_image(run):
    # if it's not a string, I'm assuming it's a real run
    if type(run) == str:
        run = neptune.init_run(
            with_id=run, mode="read-only", project=project, api_token=api_token
        )
    
    calibration_filename = path_to_data + run["sample/calibration_filename"].fetch()
    cm = CalibrationInfoManager(calibration_filename) 
    reference_camera = cm.reference_camera

    download_filename = f"temp_download_file_{get_timestamp()}.png"
    run[f"sample/images/image_{reference_camera}"].download(download_filename) 

    image = np.asarray(Image.open(download_filename)) 

    os.remove(download_filename)
    return image 

    

        
