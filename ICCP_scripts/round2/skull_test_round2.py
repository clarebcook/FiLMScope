from filmscope.reconstruction import RunManager, generate_config_dict 
from filmscope.util import get_timestamp, load_dictionary, save_dictionary
from filmscope.config import log_folder
from tqdm import tqdm
import os
import torch
from matplotlib import pyplot as plt 
from utility_functions import count_needed_runs
from select_subsets import subsets 
import numpy as np

# select the name of a sample previously saved using "save_new_sample.ipynb",
# the gpu number, and whether or not to log with neptune.ai
sample_name = "skull_tool_4x4_08_12"
frame_number = 700
gpu_number = "4"
use_neptune = False

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

experiment_dict_filename = log_folder + f'/skull_frame_{frame_number}_round2.json'
if os.path.exists(experiment_dict_filename):
    experiment_dict = load_dictionary(experiment_dict_filename) 
else:
    experiment_dict = {}

experiment_log_folder = log_folder + f'/skull_frame_{frame_number}_results'
if not os.path.exists(experiment_log_folder):
    os.mkdir(experiment_log_folder)

noise_stds = [0, 1, 5, 10, 15]


def check_if_complete(experiment_dict, image_numbers, noise):
    complete = False
    for item in experiment_dict.values():
        cams = item["cameras"] 
        if len(cams) != len(image_numbers):
            continue 
        if len(np.intersect1d(cams, image_numbers)) != len(cams):
            continue 
        run_noise = item["noise"][0] 
        if run_noise == noise:
            complete = True 
            break
    return complete 


for custom_image_numbers in subsets: 
    print(custom_image_numbers) 
    num_cameras = len(custom_image_numbers) 
    iterations = min(int(300 * 48 / num_cameras), 1000)

    for noise_std in noise_stds:

        if check_if_complete(experiment_dict, custom_image_numbers, noise_std):
            print("continuing!!!!", noise_std, len(custom_image_numbers))
            continue 

        # this isn't foolproof but I'm going to generate 
        # a random run id 
        run_id = np.random.randint(10000) 
        while run_id in experiment_dict.keys():
            run_id = np.random.randint(10000) 

        noise = [noise_std, 0]
        config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=1,
                                            camera_set="custom", use_neptune=use_neptune,
                                            frame_number=frame_number,
                                            run_args={"iters": iterations, "batch_size": 12, "num_depths": 64,
                                                      "display_freq": 100},
                                            custom_image_numbers=custom_image_numbers, 
                                            #custom_crop_info={'crop_size': (1, 1), "ref_crop_center": (0.5, 0.5)}
                                            )
        run_manager = RunManager(config_dict, noise=noise)

        losses = [] 
        display_freq = config_dict["run_args"]["display_freq"]

        for i in tqdm(range(iterations)): 
            mask_images, warp_images, numbers, outputs, loss_values = run_manager.run_epoch(i, log=False)
            losses.append(float(loss_values["total"]))

            # this section can be edited to change what is recorded
            if i % display_freq == 0:
                fig, (ax0, ax1) = plt.subplots(1, 2)

                a = outputs["warped_imgs"].detach().cpu().squeeze() 
                a = torch.mean(a, axis=0) 

                b = outputs["depth"].detach().cpu().squeeze()
                ax1.imshow(b, cmap='turbo')
                ax0.imshow(a, cmap='gray')
                #fig.suptitle(f"epoch {i}, {num_cameras} cameras, noise {noise}")
                plt.tight_layout()
                plt.show()

                plt.figure()
                plt.plot(losses)
                plt.xlabel("iteration")
                plt.ylabel("loss")
                plt.title("loss")
                plt.show()

                plt.close()


        # I'm only saving if it completes   
        warp = outputs["warped_imgs"].detach().cpu().squeeze().numpy()
        warp = np.mean(warp, axis=0) 
        depth = outputs["depth"].detach().cpu().squeeze().numpy()
        depth_savename = experiment_log_folder + f"/run_{run_id}_depth.npy"
        np.save(depth_savename, depth) 
        warp_savename = experiment_log_folder + f"/run_{run_id}_warp.npy"
        np.save(warp_savename, warp)

        settings_savename = experiment_log_folder + f"/run_{run_id}_config.json"
        save_dictionary(config_dict, settings_savename)

        dict_entry = {
            "noise": noise, 
            "cameras": custom_image_numbers, 
            "loss": losses, 
        }

        if os.path.exists(experiment_dict_filename):
            experiment_dict = load_dictionary(experiment_dict_filename)

        experiment_dict[run_id] = dict_entry
        save_dictionary(experiment_dict, experiment_dict_filename)





