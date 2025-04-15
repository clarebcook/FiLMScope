# this is not a very important script
# but I'm loading in and re-running some previous runs
# to see what may have gone well or poorly during the reconstruction 
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

# crops given in [startx, endx, starty, endx] 
# with pixels corresponding to the full, uncropped and unbinned image
# right now the height and width need to be multiples of 32
patch = [1200, 2160 + 32 * 10, 400, 1360 + 32 * 10]
#patch = [1200, 2160, 400, 1360]

low_res_downsample = 4 
low_res_depth = torch.load(path_to_data + '/finger/low_res_height_map.pt',
                        weights_only=False)

# get appropriate patch from low res depth map, and reszie  
depth_patch = low_res_depth[
    int(patch[0]/low_res_downsample):int(patch[1]/low_res_downsample),
    int(patch[2]/low_res_downsample):int(patch[3]/low_res_downsample)]
depth_patch = F.interpolate(
    depth_patch[None, None], size=(patch[1] - patch[0], patch[3] - patch[2]))

log_folder = "/media/Friday/Temporary/Clare/ICCP_result_storage/round_2_results"
noise = [10, 0]

run_id = 6174
run_id = 6792
run_id = 3657
config_dict = load_dictionary(log_folder + f'/finger_results/run_{run_id}_config.json')


run_manager = RunManager(config_dict, guide_map=depth_patch.squeeze(), noise=noise)

display_freq = 25
losses = []
for i in tqdm(range(300)): 
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