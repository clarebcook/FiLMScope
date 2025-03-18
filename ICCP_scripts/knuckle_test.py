from filmscope.config import path_to_data, log_folder
from filmscope.reconstruction import generate_config_dict, RunManager
from filmscope.calibration import CalibrationInfoManager
from filmscope.util import load_dictionary, save_dictionary
from filmscope.recon_util import get_sample_information
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
import numpy as np
import xarray as xr 
from matplotlib import pyplot as plt 
from utility_functions import count_needed_runs 

# select sample name and gpu number
sample_name = "knuckle_video"
gpu_number = "3"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

image_filename = get_sample_information(sample_name)["image_filename"]
dset = xr.open_dataset(path_to_data + '/' + image_filename)
frame_numbers = dset.frame_number.data
dset = None 

log_description = "tests with knuckle video"

frame = 438 
experiment_dict_filename = log_folder + f'/knuckle_frame_{frame}.json'
if os.path.exists(experiment_dict_filename):
    experiment_dict = load_dictionary(experiment_dict_filename) 
else:
    experiment_dict = {}

use_neptune = True 

all_num_cameras = [4, 3, 48, 40, 30, 20, 10, 5]
all_repeats = [3, 3, 1, 1, 1, 2, 3, 5]
all_noise_stds = [1, 5, 10, 15]

all_num_cameras = [48, 40, 30, 20, 8, 15, 10, 5, 4, 3]
all_repeats = [1, 1, 1, 3, 3, 3, 3, 3, 4, 3]
all_noise_stds = [1, 5, 10]

# all_num_cameras = [10, 8, 5, 4, 3]
# all_repeats = [2, 2, 2, 2, 2]
# all_noise_stds = [1, 5, 10]





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
                sample_name=sample_name, 
                use_neptune=use_neptune,
                camera_set="custom", 
                run_args={"num_depths": 32,
                        "batch_size": 12,
                        "iters": iterations,
                        "display_freq": 500},
                downsample=1,
                frame_number=frame, 
                custom_image_numbers=custom_image_numbers
            )

            config_dict["sample_info"]["depth_range"] = (-7, 6)


            # set up the run manager 
            # depth patch is given as [1, H, W]
            run_manager = RunManager(config_dict=config_dict,
                                    noise=noise)

            reference_image = run_manager.dataset.reference_image.cpu().squeeze()


            # perform reconstruction
            iters = config_dict["run_args"]["iters"]
            display_freq = 25
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

                    a = outputs["warped_imgs"].detach().cpu().squeeze() 
                    a = torch.mean(a, axis=0) 

                    b = outputs["depth"].detach().cpu().squeeze()
                    ax1.imshow(b, cmap='turbo')
                    ax0.imshow(a, cmap='gray')
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


            if os.path.exists(experiment_dict_filename):
                experiment_dict = load_dictionary(experiment_dict_filename)

            experiment_dict[id] = dict_entry
            save_dictionary(experiment_dict, experiment_dict_filename)

            run_manager.end() 

