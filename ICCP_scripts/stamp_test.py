from filmscope.reconstruction import RunManager, generate_config_dict 
from filmscope.util import get_timestamp, load_dictionary, save_dictionary
from filmscope.config import log_folder
from tqdm import tqdm
import os
import torch
from matplotlib import pyplot as plt 
from utility_functions import count_needed_runs

# select the name of a sample previously saved using "save_new_sample.ipynb",
# the gpu number, and whether or not to log with neptune.ai
sample_name = "stamp_02_08"
gpu_number = "0"
use_neptune = True

os.environ["CUDA_VISIBLE_DEVICES"] = gpu_number

experiment_dict_filename = log_folder + f'/stamp_runs_v2.json'
if os.path.exists(experiment_dict_filename):
    experiment_dict = load_dictionary(experiment_dict_filename) 
else:
    experiment_dict = {}


print(experiment_dict_filename) 

all_num_cameras = [48, 40, 30, 20, 10, 5, 4, 3]
all_repeats = [1, 1, 1, 2, 3, 5, 5, 5]
all_noise_stds = [0, 5, 10]



for num_cameras, repeats in zip(all_num_cameras, all_repeats):
    iterations = 240 # min(int(500 * 48 / num_cameras), 1000)

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



            config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=1,
                                            camera_set="all", use_neptune=use_neptune,
                                            load_crop_entry=False, 
                                            # frame_number=frame_number,
                                            run_args={"iters": iterations, "batch_size": 12, "num_depths": 64,
                                                        "display_freq": 20},
                                            loss_weights = {"smooth": 0.35},
                                            custom_crop_info={'crop_size': (0.15, 0.2)} #, "ref_crop_center": (0.42, 0.65)}
                                            )
            run_manager = RunManager(config_dict, noise=noise)

            # perform reconstruction
            iters = config_dict["run_args"]["iters"]
            losses = []
            display_freq = 100
            for i in tqdm(range(iters)):
                log = (i % config_dict["run_args"]["display_freq"] == 0) or (i == iters - 1)
                mask_images, warp_images, numbers, outputs, loss_values = run_manager.run_epoch(
                    i, log=(log and config_dict["use_neptune"]))
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
                    plt.imshow(b)
                    plt.title("full") 
                    plt.show()

                    plt.figure()
                    plt.imshow(b[150:300, 0:300])
                    plt.title("partial") 
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
