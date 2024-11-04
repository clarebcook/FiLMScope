from filmscope.config import path_to_data
from filmscope.reconstruction import generate_config_dict, RunManager
from filmscope.calibration import CalibrationInfoManager
import torch.nn.functional as F
import torch
import os
from tqdm import tqdm
from matplotlib import pyplot as plt 

# select gpu number to run on
gpu_number = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# name of a sample previously saved with "save_new_sample.ipynb"
sample = "finger"

config_dict = generate_config_dict(
    gpu_number=gpu_number,
    sample_name=sample, 
    use_neptune=False,
    camera_set="all", 
    run_args={"num_depths": 32,
              "batch_size": 4,
              "iters": 300,
              "display_freq": 75}
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
patch = [1200, 2160, 400, 1360]

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
                         guide_map=depth_patch.squeeze(0))

reference_image = run_manager.dataset.reference_image.cpu().squeeze()

# perform reconstruction
iters = config_dict["run_args"]["iters"]
losses = []
for i in tqdm(range(iters)):
    _, _, _, outputs, loss_values = run_manager.run_epoch(i)
    losses.append(float(loss_values["total"]))

    # logging can also be performed with neptune or saving elsewhere
    # but this displays outputs with matplotlib
    if i % config_dict["run_args"]["display_freq"] == 0:
        fig, (ax0, ax1) = plt.subplots(1, 2)
        ax1.imshow(outputs["depth"].detach().cpu().squeeze(), cmap='turbo')
        ax0.imshow(reference_image, cmap='gray')
        fig.suptitle(f"epoch {i}")
        plt.tight_layout()
        plt.show()

        plt.figure()
        plt.plot(losses)
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.title("loss")
        plt.show()
 