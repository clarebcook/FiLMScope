from filmscope.config import path_to_data, log_folder
from filmscope.reconstruction import generate_config_dict, RunManager
from filmscope.calibration import CalibrationInfoManager
from filmscope.util import load_dictionary, save_dictionary
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt 



def count_needed_runs(experiment_dict, repeats, all_noise_stds, num_cameras):
    partials = []
    partial_cameras = []
    # and ideally if there's an incomplete set 
    # we'd pick up there 
    #num_cameras = 4 
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


experiment_dict_filename = log_folder + '/finger_from_low_res.json'
if os.path.exists(experiment_dict_filename):
    experiment_dict = load_dictionary(experiment_dict_filename) 
else:
    experiment_dict = {}

use_neptune = True 

# all_num_cameras = [48, 40, 30, 20, 15, 10, 5, 3]
# all_repeats = [1, 1, 2, 3, 4, 4, 5, 10, 10]
# all_noise_stds = [5, 10, 20, 40, 70, 100]

all_num_cameras = [4, 3, 48, 40, 30, 20, 10, 5]
all_repeats = [8, 8, 1, 1, 2, 2, 4, 8]
all_noise_stds = [1, 5, 10, 15]

# select gpu number to run on
gpu_number = "2"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# name of a sample previously saved with "save_new_sample.ipynb"
sample = "finger"

for num_cameras, repeats in zip(all_num_cameras, all_repeats):
    iterations = min(int(500 * 48 / num_cameras), 2000)
    # check how many have already been done
    # ig for a repeat to count, 
    # it needs to have happened at all the noise levels
    # that's not ideal but it's fine for now 
    partials, partial_cameras = count_needed_runs(
        experiment_dict, repeats, all_noise_stds, num_cameras
    )

    print(num_cameras, repeats, partials, partial_cameras)

    for cur_noise_stds, custom_image_numbers in zip(partials, partial_cameras):#repeat in range(repeats): 
        if custom_image_numbers is None:
            custom_image_numbers = torch.randperm(48)[:num_cameras]
            if 20 not in custom_image_numbers:
                custom_image_numbers[0] = 20
            custom_image_numbers = custom_image_numbers.tolist()
        for noise_std in cur_noise_stds: 
            noise = [noise_std, 0]

            config_dict = generate_config_dict(
                gpu_number=gpu_number,
                sample_name=sample, 
                use_neptune=use_neptune,
                camera_set="custom", 
                run_args={"num_depths": 32,
                        "batch_size": 12,
                        "iters": iterations,
                        "display_freq": 500},
                downsample=1,
                custom_image_numbers=custom_image_numbers
            )

            # determine what depth range to consider above and below the guide map 
            config_dict["sample_info"]["depth_range"] = (-1, 1) # in mm

            # this is a previously saved low resolution depth map 
            # with 4x4 downsampling
            low_res_downsample = 4 
            low_res_depth = torch.load(path_to_data + '/finger/low_res_height_map.pt',
                                    weights_only=False)

            # crops given in [startx, endx, starty, endx] 
            # with pixels corresponding to the full, uncropped and unbinned image
            # right now the height and width need to be multiples of 32
            patch = [1200, 2160 + 32 * 10, 400, 1360 + 32 * 10]
            #patch = [1200, 2160, 400, 1360]

            # get appropriate patch from low res depth map, and reszie  
            depth_patch = low_res_depth[
                int(patch[0]/low_res_downsample):int(patch[1]/low_res_downsample),
                int(patch[2]/low_res_downsample):int(patch[3]/low_res_downsample)]
            depth_patch = F.interpolate(
                depth_patch[None, None], size=(patch[1] - patch[0], patch[3] - patch[2]))

            # convert to normalized crop values 
            full_image_shape = CalibrationInfoManager(
                path_to_data + config_dict["sample_info"]["calibration_filename"]).image_shape
            crop_center = (((patch[1] + patch[0]) / 2) / full_image_shape[0],
                        ((patch[3] + patch[2]) / 2) / full_image_shape[1]) 
            crop_size = ((patch[1] - patch[0]) / full_image_shape[0],
                        (patch[3] - patch[2]) / full_image_shape[1])
            # determine approximate average height for image alignment 
            height_est = torch.mean(depth_patch)

            # load the information into the config dict
            config_dict["sample_info"]["crop_size"] = crop_size
            config_dict["sample_info"]["ref_crop_center"] = crop_center 
            config_dict["sample_info"]["height_est"] = height_est 

            # set up the run manager 
            # depth patch is given as [1, H, W]
            run_manager = RunManager(config_dict=config_dict,
                                    guide_map=depth_patch.squeeze(0),
                                    noise=noise)

            reference_image = run_manager.dataset.reference_image.cpu().squeeze()


            # perform reconstruction
            iters = config_dict["run_args"]["iters"]
            display_freq = 150
            losses = []
            for i in tqdm(range(iters)):
                log_freq = config_dict["run_args"]["display_freq"]
                log = (i % log_freq == 0) or i == iterations - 1
                _, _, _, outputs, loss_values = run_manager.run_epoch(i, log=log)
                losses.append(float(loss_values["total"]))

                # logging can also be performed with neptune or saving elsewhere
                # but this displays outputs with matplotlib
                if i % display_freq == 0:
                    fig, (ax0, ax1) = plt.subplots(1, 2)
                    ax1.imshow(outputs["depth"].detach().cpu().squeeze(), cmap='turbo')
                    ax0.imshow(reference_image, cmap='gray')
                    fig.suptitle(f"epoch {i}, {num_cameras} cameras, noise {noise}")
                    plt.tight_layout()
                    plt.show()

                    plt.figure()
                    plt.plot(losses)
                    plt.xlabel("iteration")
                    plt.ylabel("loss")
                    plt.title("loss")
                    plt.show()

                    plt.close()


            id = run_manager.logger.neptune_run["sys/id"].fetch()
            dict_entry = {
                "noise": noise, 
                "cameras": custom_image_numbers, 
            }
            experiment_dict[id] = dict_entry
            save_dictionary(experiment_dict, experiment_dict_filename)

            run_manager.end() 