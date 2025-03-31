# just slightly modifying to test with 
# camera 22 as ref again 
# with a different set of subsets


# these should be able to be compared with the stamp tests from round 1
# which were also using 22 as the reference camera 

# those are stored in 
#"ICCP_result_storage/stamp_results_v3"
# with information in "ICCP_result_storage/stamp_runs_v3.json"

from filmscope.reconstruction import RunManager, generate_config_dict 
from filmscope.util import get_timestamp, load_dictionary, save_dictionary
from filmscope.config import log_folder, path_to_data
from filmscope.recon_util import get_sample_information
from tqdm import tqdm
import os
import numpy as np 
import torch
from matplotlib import pyplot as plt 
from utility_functions import count_needed_runs
from select_subsets_r3 import subsets 

#info = get_sample_information("stamp_02_08")

sample_name = "stamp_02_08"
gpu_number = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

experiment_dict_filename = log_folder + f'/stamp_runs_round3.json'
if os.path.exists(experiment_dict_filename):
    experiment_dict = load_dictionary(experiment_dict_filename) 
else:
    experiment_dict = {}


experiment_log_folder = log_folder + f'/stamp_results_r3'
if not os.path.exists(experiment_log_folder):
    os.mkdir(experiment_log_folder)

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

log_locs = [60, 100, 140, 240]
noise_stds = [0, 10, 20, 30]
noise_stds = [0]

for noise_std in noise_stds:
    for custom_image_numbers in subsets: 
        print(custom_image_numbers) 
        num_cameras = len(custom_image_numbers) 
        iterations = 250



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
                                            camera_set="custom", use_neptune=False,
                                            frame_number=-1,
                                            run_args={"iters": iterations, "batch_size": 12, "num_depths": 64,
                                                      "display_freq": 20},
                                            custom_image_numbers=custom_image_numbers, 
                                            #custom_crop_info={'crop_size': (0.15, 0.2)} #, "ref_crop_center": (0.5, 0.5)}
                                            )
        
        # things have gotten a little weird, 
        # so I'm manually copying over some information from previous tests
        # so that I can more eaisly continue to compare 
        # with old runs 
        config_dict = generate_config_dict(gpu_number, sample_name=sample_name)
        info = config_dict["sample_info"] 
        info["height_est"] = 2.55 
        info["ref_crop_center"] = (0.537109375, 0.46875)
        info["crop_size"] = (0.15, 0.2)
        info["calibration_filename"] = "/stamp_2024_02_08/calibration_information_old"

        assert os.path.exists(path_to_data + info["calibration_filename"])

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

                plt.figure()
                plt.imshow(b[100:300, 100:300], cmap='magma') 
                plt.show()

                plt.close()
            
            if i in log_locs:
                # I'm only saving if it completes   
                warp = outputs["warped_imgs"].detach().cpu().squeeze().numpy()
                warp = np.mean(warp, axis=0) 
                depth = outputs["depth"].detach().cpu().squeeze().numpy()
                depth_savename = experiment_log_folder + f"/run_{run_id}_iter_{i}_depth.npy"
                np.save(depth_savename, depth) 
                warp_savename = experiment_log_folder + f"/run_{run_id}_iter_{i}_warp.npy"
                np.save(warp_savename, warp)


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
