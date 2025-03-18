

### import everything
import numpy as np
from filmscope.recon_util import (get_individual_crop, get_sample_information,
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
        "reuse_model": True,  # only for videos
    }


default_crop_info = {
    "depth_range": None, # (min_height, max_height)
    "height_est": None, # float, within depth_range 
    "ref_crop_center": None, #(centerx, centery) between 0 and 1
    "crop_size": None, #(sizex, sizey) between 0 and 1 
    "crop_name": None, # only needed if saving 
    "save": None, # bool 
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
                         downsample=1, camera_set="all",
                         frame_number=-1,
                         use_individual_crops=True, load_crop_entry=False, log_description="",
                         loss_weights={}, run_args={}, custom_image_numbers=None,
                         custom_crop_info={}):
    if custom_image_numbers is not None:
        camera_set = "custom"
        cam_num_sets["custom"] = custom_image_numbers

    # put the settings into the sample info/wherever they should be 
    sample_info = get_sample_information(sample_name) 
    sample_info["downsample"] = downsample
    sample_info["camera_set"] = camera_set

    if use_individual_crops:
        if crop_name is not None or crop_number is not None:
            # option 1: specify entry number or name for previous entry
            crop_info, entry_number = get_individual_crop(
                sample_name, crop_name, crop_number
            )

        else:
            # any values that are set to None 
            # will be pulled from the information saved with the sample
            for key in default_crop_info:
                if key not in custom_crop_info:
                    custom_crop_info[key] = default_crop_info[key]

            crop_info, entry_number = add_individual_crop(
                sample_name,
                custom_crop_info["crop_name"],
                save=custom_crop_info["save"],
                depth_range=custom_crop_info["depth_range"],
                height_est=custom_crop_info["height_est"],
                ref_crop_center=custom_crop_info["ref_crop_center"],
                crop_size=custom_crop_info["crop_size"],
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

    for key in default_loss_weights:
        if key not in loss_weights:
            loss_weights[key] = default_loss_weights[key]

    # remove stored sample information not used in this run
    if "ind_crops" in sample_info:
        sample_info.pop("ind_crops")

    # add the image numbers used 
    sample_info["image_numbers"] = cam_num_sets[sample_info["camera_set"]]
    sample_info["num_cameras"] = len(sample_info["image_numbers"])

    # and the frame number 
    run_args["frame_number"] = frame_number

    # then arrange everything needed into one dictionary
    config_dictionary = {
        "run_args": run_args,
        "log_description": log_description,
        "use_neptune": use_neptune,
        "sample_info": sample_info,
        "loss_weights": loss_weights,
        "gpu_number": gpu_number,
    }

    return config_dictionary
