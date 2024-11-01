from filmscope.reconstruction import generate_config_dict, RunManager
from filmscope.recon_util import get_sample_information
from filmscope.config import path_to_data

import xarray as xr
import os
from tqdm import tqdm
import sys 
import select
from matplotlib import pyplot as plt

sample_name = "knuckle_video"
gpu_number = "0"

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

# used for neptune logging
log_description = "shortened kncukle video recon"

# determine frame numbers 
image_filename = get_sample_information(sample_name)["image_filename"]
dset = xr.open_dataset(path_to_data + '/' + image_filename)
frame_numbers = dset.frame_number.data
dset = None

use_neptune = False
config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=1,
                                   camera_set="all", use_neptune=use_neptune,
                                   log_description=log_description, frame_number=frame_numbers[0],
                                   run_args={"iters": 150, "batch_size": 12, "num_depths": 32,
                                             "display_freq": 1000})
run_manager = RunManager(config_dict)

run_args = config_dict["run_args"]
for frame_number in frame_numbers:
    print("")
    print(f"starting for frame {frame_number}")
    print("enter 'iters: {number}' to adjust # iterations for the NEXT frame")
    print("OR enter 'continue' to move on to the next frame")
    print("")

    # useful to update description for each frame
    run_manager.config_dict["log_description"] = f"frame {frame_number} " + log_description
    run_manager.swap_frames(frame_number)
    
    losses = []
    for i in tqdm(range(run_args["iters"])):
        log = (i == run_args["iters"] - 1) or (i % run_args["display_freq"] == 0)
        _, _, _, outputs, loss_values = run_manager.run_epoch(i, log and use_neptune)
        losses.append(float(loss_values["total"]))

        # check here for terminal inputs to move on to next frame if desired
        # or adjust the number of iterations being used
        if select.select([sys.stdin], [], [], 0.1)[0]:
            user_input = sys.stdin.readline().strip()

            command_parts = user_input.split(": ", 1)
            if len(command_parts) == 2:
                command, value = command_parts
                if command.lower() == "iters":
                    print(f"switching to {value} iters on next frame")
                    run_args["iters"] = int(value)
            elif user_input.lower() == "continue":
                print("moving on to next frame...")
                break
    
        if log and not use_neptune:
            fig, (ax0, ax1) = plt.subplots(1, 2) 
            ax1.imshow(outputs["depth"].detach().cpu().squeeze(), cmap='turbo')
            ax0.imshow(run_manager.reference_image.cpu().squeeze(), cmap='gray')
            ax1.set_title(f"height map, iteration {i}")
            ax0.set_title(f"reference image, frame {frame_number}")
            plt.show()

            plt.figure()
            plt.plot(losses)
            plt.title(f"losses, frame {frame_number}")
            plt.xlabel("iteration")
            plt.ylabel("loss")
            plt.show()