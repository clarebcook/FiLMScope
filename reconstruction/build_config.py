

### import everything
import numpy as np
from FiLMScope.recon_util import (get_individual_crop, get_sample_information,
                                  add_individual_crop)

default_loss_weights = {
        "ssim": 6,
        "smooth": 0.35, 
        "smooth_lambda": 1, # this is the weight of the reference image when computing depth smoothness
    }

# some of these are only relevant for specific run types
default_run_args = {
        "lr": 1e-3,
        "weight_decay": 0,
        "batch_size": 12,
        "iters": 200,  # not used for "video"
        # "num_depths" currently needs to be divisible by 2^unet_layers
        "num_depths": 32,
        "optim": "Adam",
        "refine": False,
        # if false, the height map will align with the reference image
        # if true, it will be corrected to the reference plane
        "rectify_perspective": False,
        "loader_shuffle": True,
        "loader_num_workers": 4,
        "all_images_in_volume": True,
        "display_freq": 15,
        "seg_clusters": 4,
        "drop_last": False,
        "unet_layers": 4,

        # for now the below two settings should be lists 
        # or None to use default values
        # though we can probably adjust that later 
        "unet_layer_channels": [8, 16, 16, 16, 16], # length of this list should be unet_layers + 1
        # can optionally be specified to deviate from default, otherwise None
        "unet_layer_strides":None, 
        "reuse_model": True,  # only for "video"
        "start_frame": 0,  # only for "video"
        "end_frame": None,  # only for "video", set to None to do all frames
    }

cam_num_sets = {
        "all": np.arange(48),
        "6x6": np.arange(6, 42),
        "5x5": np.concatenate(
            (
                np.arange(6, 11),
                np.arange(12, 17),
                np.arange(18, 23),
                np.arange(24, 29),
                np.arange(30, 35),
            )
        ),
        "4x4": np.concatenate(
            (np.arange(13, 17), np.arange(19, 23), np.arange(25, 29), np.arange(31, 35))
        ),
        "3x3": np.concatenate((np.arange(14, 17), np.arange(20, 23), np.arange(26, 29))),
        "2x2": np.asarray([21, 22, 27, 28]),
        "spread_2x2": np.asarray([6, 10, 22, 30, 34]),
        "spread_3x3": np.asarray([6, 8, 10, 18, 20, 22, 30, 32, 34])
    }


def generate_config_dict(gpu_number, sample_name, use_neptune=False,
                         downsample=1, camera_set="all", run_type="frame",
                         use_individual_crops=True, load_crop_entry=False, log_description="",
                         loss_weights={}, run_args={}, custom_image_numbers=None):
    if custom_image_numbers is not None:
        camera_set = "custom"
        cam_num_sets["custom"] = custom_image_numbers

    # put the settings into the sample info/wherever they should be 
    sample_info = get_sample_information(sample_name) 
    sample_info["downsample"] = downsample
    sample_info["camera_set"] = camera_set

    if use_individual_crops:
        if load_crop_entry:
            # option 1: specify entry number or name for previous entry
            crop_number = None
            crop_name = "full"
            crop_info, entry_number = get_individual_crop(
                sample_name, crop_name, crop_number
            )

        else:
            # option 2: specify values
            # any of these can be set to None, and values will be taken
            # from the sample info
            depth_range = None
            height_est = None
            ref_crop_center = None #
            crop_size = None 
            crop_name = None

            # can choose to save this for later
            save = False
            crop_name = None
            crop_info, entry_number = add_individual_crop(
                sample_name,
                crop_name,
                save=save,
                depth_range=depth_range,
                height_est=height_est,
                ref_crop_center=ref_crop_center,
                crop_size=crop_size,
            )

        crop_info["crop_entry_number"] = entry_number
        for key, value in crop_info.items():
            sample_info[key] = value

    else:
        sample_info["height_est"] = 0
        sample_info["ref_crop_center"] = None
        sample_info["crop_size"] = None

    for key in default_run_args:
        if key not in run_args:
            run_args[key] = default_run_args[key]
    run_args["run_type"] = run_type

    for key in default_loss_weights:
        if key not in loss_weights:
            loss_weights[key] = default_loss_weights[key]


    ### code to manage the supplied settings
    if "ind_crops" in sample_info:
        sample_info.pop("ind_crops")

    # remove values from "run_args" based on run type
    if run_type == "frame":
        for setting in ["start_frame", "end_frame", "reuse_model"]:
            run_args.pop(setting)

    
    sample_info["image_numbers"] = cam_num_sets[sample_info["camera_set"]]
    sample_info["num_cameras"] = len(sample_info["image_numbers"])


    # then arrange everything needed into one dictionary
    config_dictionary = {
        "run_args": run_args,
        "log_description": log_description,
        "use_neptune": use_neptune,
        "run_type": run_type,
        "sample_info": sample_info,
        "loss_weights": loss_weights,
        "gpu_number": gpu_number,
    }

    return config_dictionary
