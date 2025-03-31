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
from utility_functions import count_needed_runs




log_folder = "/media/Friday/Temporary/Clare/ICCP_result_storage/round_2_results"



# select the name of a sample previously saved using "save_new_sample.ipynb",
# the gpu number, and whether or not to log with neptune.ai
sample_name = "finger"
gpu_number = "0"
use_neptune = False

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

experiment_dict_filename = log_folder + '/finger_from_low_res.json'
if os.path.exists(experiment_dict_filename):
    experiment_dict = load_dictionary(experiment_dict_filename) 
else:
    experiment_dict = {}

experiment_log_folder = log_folder + f'/finger_results'
if not os.path.exists(experiment_log_folder):
    os.mkdir(experiment_log_folder)

noise_stds = [0, 5, 10, 15]



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
    for noise_std in noise_stds:
        print(custom_image_numbers) 
        num_cameras = len(custom_image_numbers) 
        iterations = min(int(500 * 48 / num_cameras), 1200)

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
                                            frame_number=-1,
                                            run_args={"iters": iterations, "batch_size": 12, "num_depths": 32,
                                                      "display_freq": 100},
                                            custom_image_numbers=custom_image_numbers, 
                                            #custom_crop_info={'crop_size': (1, 1), "ref_crop_center": (0.5, 0.5)}
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


        run_manager = RunManager(config_dict, guide_map=depth_patch.squeeze(), noise=noise)

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
        c2 = config_dict.copy()
        c2["sample_info"]["height_est"] = float(c2["sample_info"]["height_est"])
        save_dictionary(c2, settings_savename)

        dict_entry = {
            "noise": noise, 
            "cameras": custom_image_numbers, 
            "loss": losses, 
        }

        if os.path.exists(experiment_dict_filename):
            experiment_dict = load_dictionary(experiment_dict_filename)

        experiment_dict[run_id] = dict_entry
        save_dictionary(experiment_dict, experiment_dict_filename)



