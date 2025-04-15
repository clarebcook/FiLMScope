from filmscope.reconstruction import RunManager, generate_config_dict 
from filmscope.util import get_timestamp, load_dictionary, save_dictionary
from filmscope.config import log_folder
from tqdm import tqdm
import os
import torch
from matplotlib import pyplot as plt 
from utility_functions import check_if_complete
from select_subsets import subsets 
import numpy as np


# select the name of a sample previously saved using "save_new_sample.ipynb",
# the gpu number, and whether or not to log with neptune.ai
sample_name = "porcine"
gpu_number = "1"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

log_folder = "/media/Friday/Temporary/Clare/ICCP_result_storage"
experiment_dict_filename = log_folder + f'/porcine_v2.json'
if os.path.exists(experiment_dict_filename):
    experiment_dict = load_dictionary(experiment_dict_filename) 
else:
    experiment_dict = {}

experiment_log_folder = log_folder + f'/porcine_results'
if not os.path.exists(experiment_log_folder):
    os.mkdir(experiment_log_folder)

noise_stds = [0, 10, 20, 30]

# limited run to get a couple things for figures 
subsets_focus = [
    np.arange(48).tolist(), 
    [13, 14, 15, 16, 19, 20, 21, 22, 25, 26, 27, 28, 31, 32, 33, 34], 
    [6, 10, 20, 30, 34],  
    [14, 20, 26, 16, 28],
    [20, 21, 26, 27]
]
noise_stds = [5, 15] 

subsets = subsets_focus + subsets

save_iters = [100, 225, 350]
iterations = 352

for noise_std in noise_stds:
    for custom_image_numbers in subsets:
        print(custom_image_numbers) 
        num_cameras = len(custom_image_numbers)

        # if check_if_complete(experiment_dict, custom_image_numbers, noise_std):
        #     print("continuing - ", noise_std, len(custom_image_numbers))
        #     continue

        run_id = np.random.randint(10000) 
        while run_id in experiment_dict.keys():
            run_id = np.random.randint(10000) 

        noise = [noise_std, 0]

        config_dict = generate_config_dict(
            sample_name=sample_name, gpu_number=gpu_number, downsample=1,
            camera_set="custom", use_neptune=False,
            frame_number=-1,
            run_args={"iters": iterations, "batch_size": 12, "num_depths": 32,
                        "display_freq": 25},
            custom_image_numbers=custom_image_numbers, 
        )

        run_manager = RunManager(config_dict, noise=noise)

        break 
    break 


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

            if i in save_iters:  
                warp = outputs["warped_imgs"].detach().cpu().squeeze().numpy()
                warp = np.mean(warp, axis=0) 
                depth = outputs["depth"].detach().cpu().squeeze().numpy()
                depth_savename = experiment_log_folder + f"/run_{run_id}_iter_{i}_depth.npy"
                np.save(depth_savename, depth) 
                warp_savename = experiment_log_folder + f"/run_{run_id}_iter_{i}_warp.npy"
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
