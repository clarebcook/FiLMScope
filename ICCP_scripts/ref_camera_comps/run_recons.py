from filmscope.reconstruction import RunManager, generate_config_dict 
from filmscope.util import get_timestamp, load_dictionary, save_dictionary
from filmscope.config import log_folder
from tqdm import tqdm
import os
import torch
from matplotlib import pyplot as plt 

import numpy as np


cam_number = 3 
experiment_log_folder = log_folder + f'/stamp_perspective_comp'
gpu_number = "4"
use_neptune = False
log_locs = [20, 60, 100, 140, 200]
iterations = 240

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

for cam_number in range(20, 21):
    # test_name = experiment_log_folder + f"/camera_{cam_number}_iter_{iterations - 1}_depth.npy"
    # if os.path.exists(test_name):
    #     continue

    sample_name = f"stamp_camera_{cam_number}_v2"

    config_dict = generate_config_dict(
        sample_name=sample_name, gpu_number=gpu_number, downsample=1,
        camera_set="all", use_neptune=use_neptune,
        frame_number=-1,
        run_args={"iters": iterations, "batch_size": 12, "num_depths": 64,
                    "display_freq": 20},
        #custom_image_numbers=custom_image_numbers, 
        custom_crop_info={'crop_size': (0.15, 0.2)} #, "ref_crop_center": (0.5, 0.5)}
    )

    run_manager = RunManager(config_dict, noise=[0, 0])

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

            plt.figure()
            plt.imshow(b[100:300, 100:300], cmap='magma') 
            plt.title(f"camera {cam_number}")
            plt.show()

            plt.close()
        
        if i in log_locs:
            # I'm only saving if it completes   
            warp = outputs["warped_imgs"].detach().cpu().squeeze().numpy()
            warp = np.mean(warp, axis=0) 
            depth = outputs["depth"].detach().cpu().squeeze().numpy()
            depth_savename = experiment_log_folder + f"/camera_{cam_number}_iter_{i}_depth.npy"
            #np.save(depth_savename, depth) 
            warp_savename = experiment_log_folder + f"/camera_{cam_number}_iter_{i}_warp.npy"
            #np.save(warp_savename, warp)


    # I'm only saving if it completes   
    warp = outputs["warped_imgs"].detach().cpu().squeeze().numpy()
    warp = np.mean(warp, axis=0) 
    depth = outputs["depth"].detach().cpu().squeeze().numpy()
    depth_savename = experiment_log_folder + f"/camera_{cam_number}_iter_{i}_depth.npy"
    #np.save(depth_savename, depth) 
    warp_savename = experiment_log_folder + f"/camera_{cam_number}_iter_{i}_warp.npy"
    #np.save(warp_savename, warp)

    settings_savename = experiment_log_folder + f"/camera_{cam_number}_config.json"
    #save_dictionary(config_dict, settings_savename)





