from filmscope.reconstruction import RunManager, generate_config_dict 
from filmscope.util import get_timestamp
from filmscope.config import log_folder
from tqdm import tqdm
import os
import torch

# select the name of a sample previously saved using "save_new_sample.ipynb",
# the gpu number, and whether or not to log with neptune.ai
sample_name = "skull_with_tool"
gpu_number = "0"
use_neptune = False

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=4,
                                   camera_set="all", use_neptune=use_neptune,
                                   run_args={"iters": 250, "batch_size": 12, "num_depths": 64,
                                             "display_freq": 50},)
run_manager = RunManager(config_dict)

# if logging is not done with neptune
# set up a folder for logging models and depth maps during reconstruction
if not config_dict["use_neptune"]:
    run_log_folder = log_folder + f"/run_{sample_name}_{get_timestamp()}"
    if not os.path.exists(run_log_folder):
        os.mkdir(run_log_folder)

# perform reconstruction
iters = config_dict["run_args"]["iters"]
losses = []
for i in tqdm(range(iters)):
    log = (i % config_dict["run_args"]["display_freq"] == 0) or (i == iters - 1)
    mask_images, warp_images, numbers, outputs, loss_values = run_manager.run_epoch(
        i, log=(log and config_dict["use_neptune"]))
    losses.append(float(loss_values["total"]))

    # this section can be edited to change what is recorded
    if log and not config_dict["use_neptune"]:
        # save model
        model_filename = run_log_folder + f'/model_epoch_{i}.pth'
        torch.save(run_manager.model.state_dict(), model_filename)
        # save height map
        depth_filename = run_log_folder + f'/depth_epoch_{i}.pt'
        torch.save(outputs["depth"], depth_filename)

run_manager.end()