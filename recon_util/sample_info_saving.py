# this file contains useful functions for saving information 
# like crop range and depth extent for different datasets

from FiLMScope.util import load_dictionary, save_dictionary
import os
import numpy as np

samples_filename = "/media/Friday/Temporary/Clare/FiLMScope_paper_data/sample_info.json"

# if any of the first 4 args are None
# sample_info cannot be None
def _prep_individual_crop_info(
    depth_range=None,
    height_est=None,
    ref_crop_center=None,
    crop_size=None,
    sample_info=None,
):
    if depth_range is None:
        depth_range = sample_info["depth_range"]
    if height_est is None:
        height_est = np.mean(depth_range)
    if ref_crop_center is None:
        crop_values = sample_info["crop_values"]
        c0 = (crop_values[0] + crop_values[1]) / 2
        c1 = (crop_values[2] + crop_values[3]) / 2
        ref_crop_center = (c0, c1)
    if crop_size is None:
        s0 = crop_values[1] - crop_values[0]
        s1 = crop_values[3] - crop_values[2]
        crop_size = (s0, s1)

    info = {
        "depth_range": depth_range,
        "height_est": height_est,
        "ref_crop_center": ref_crop_center,
        "crop_size": crop_size,
    }
    return info

def add_individual_crop(
    sample_name,
    crop_name=None,
    # this should generally not be specified
    filename=None,
    save=True,
    *args,
    **kwargs,
):
    sample_info = get_sample_information(sample_name, filename)
    if "ind_crops" not in sample_info:
        crop_info = {}
        entry_num = 0
    else:
        crop_info = sample_info["ind_crops"]
        entry_num = np.max([i for i in crop_info.keys()]) + 1

    crop_info[entry_num] = _prep_individual_crop_info(
        sample_info=sample_info, *args, **kwargs
    )

    if not save:
        return crop_info[entry_num], -1

    crop_info[entry_num]["crop_name"] = crop_name

    sample_info["ind_crops"] = crop_info

    if filename is None:
        filename = samples_filename

    sample_dict = load_dictionary(filename)
    sample_dict[sample_name] = sample_info
    save_dictionary(sample_dict, filename)
    return crop_info[entry_num], entry_num


def get_individual_crop(sample_name, crop_name=None, crop_number=None, filename=None):
    if crop_name is None and crop_number is None:
        raise Exception(
            "At least one of 'crop_name' or 'crop_number' must be specified"
        )

    sample_info = get_sample_information(sample_name, filename)

    if crop_number is not None:
        return sample_info["ind_crops"][crop_number], crop_number

    for entry_num, values in sample_info["ind_crops"].items():
        crop_names = []
        if values["crop_name"] == crop_name:
            return values, entry_num
        else:
            crop_names.append(values["crop_name"])

    raise Exception(f"{crop_name} not in dict, options are {crop_names}")

def add_sample_entry(
    sample_name,
    sample_entry,
    overwrite=False,
    filename=None,
):
    if filename is None:
        filename = samples_filename

    if os.path.exists(filename):
        sample_dict = load_dictionary(filename)
    else:
        sample_dict = {}
    if sample_name in sample_dict and not overwrite:
        print("entry added, not overwriting")
        return sample_dict[sample_name]

    if not overwrite:
        for name, entry in sample_dict.items():
            if entry["image_filename"] == sample_entry["image_filename"]:
                print(f"This dataset is previously saved under name {name}, returning")
                return entry

    sample_dict[sample_name] = sample_entry
    save_dictionary(sample_dict, filename)



def get_sample_information(sample_name, filename=None):
    if filename is None:
        filename = samples_filename

    if not os.path.exists(filename):
        return -1

    samples = load_dictionary(filename)
    if sample_name not in samples:
        raise Exception(
            f"{sample_name} has not been saved, call 'get_all_sample_names' for options"
        )
    else:
        # retroactively add things if necessary
        if "blank_filename" not in samples[sample_name]:
            samples[sample_name]["blank_filename"] = None
        return samples[sample_name]

def get_all_sample_names(filename=None):
    if filename is None:
        filename = samples_filename

    if not os.path.exists(filename):
        return -1

    samples = load_dictionary(filename).keys()
    return [i for i in samples]