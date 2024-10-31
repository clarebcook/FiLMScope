#### functions for loading FiLMScope images
#### Note that these were originally designed to import images from a range of systems,
#### which could have images of different sizes.
#### this is why dictionaries are initially used to store the images

import numpy as np 
import json 
import os
from FiLMScope.util import load_dictionary
from tqdm import tqdm 
from PIL import Image
import xarray as xr 

# image_x_y_locs should be an n x 2 array, with x and y camera values for each image
# x locatioins should be in the first column
# this is hardcoded for the 6x8 camera array
def convert_to_single_image_numbers(
    image_x_y_locs, total_images=48, exif_orientation=8
):
    if exif_orientation != 8 or total_images != 48:
        raise ValueError(
            "This function is not set up to deal with non-conventional orientations or # of images"
        )

    image_numbers = np.zeros(len(image_x_y_locs), dtype=np.uint8)
    for i, numbers in enumerate(image_x_y_locs):
        x_num = numbers[0]
        y_num = numbers[1]
        number = 6 * y_num + x_num
        image_numbers[i] = number
    return image_numbers

def _convert_to_array_image_number(image_number):
    y_cam = int(image_number / 6)
    x_cam = image_number % 6
    return x_cam, y_cam


def convert_to_array_image_numbers(image_numbers, total_images=48, exif_orientation=8):
    if exif_orientation != 8 or total_images != 48:
        raise ValueError(
            "This function is not set up to deal with non-conventional orientations or # of images"
        )
    image_x_y_locs = np.zeros((len(image_numbers), 2), dtype=np.uint8)
    for i, number in enumerate(image_numbers):
        image_x_y_locs[i] = _convert_to_array_image_number(number)
    return image_x_y_locs

def load_image_set(filename, image_numbers=None,
                   downsample=1, frame_number=-1):
    dataset = xr.open_dataset(filename)

    # this is a little convoluted,
    # but just trying to find the image numbers
    # for all cameras in the array
    if image_numbers is None:
        x_cams, y_cams = np.meshgrid(dataset.image_x.data,
                             dataset.image_y.data)
        x_cams = x_cams.flatten()
        y_cams = y_cams.flatten()
        cam_nums = np.stack((x_cams[:, None], y_cams[:, None]), axis=1).squeeze()
        image_numbers = convert_to_single_image_numbers(cam_nums)

    # get needed x cameras and y cameras
    image_x_y_locs = convert_to_array_image_numbers(image_numbers)

    if frame_number != -1:
        dataset = dataset.sel(frame_number=frame_number)
        # 2024/08/15 this is a temporary hack
        # to deal with a specific dataset that had duplicate frames
        if "frame_number" in dataset.dims:
            dataset = dataset.isel({"frame_number": 0})

    images = {}
    for number in tqdm(image_numbers):
        image_x_y_locs = convert_to_array_image_numbers([number])
        x_cam = image_x_y_locs[0, 0]
        y_cam = image_x_y_locs[0, 1]
        single_image = dataset.sel(image_y=y_cam, image_x=x_cam)

        single_image = single_image.images.data
        single_image = single_image[::downsample, ::downsample]
        images[number] = Image.fromarray(single_image)
        # rotate the necessary amount
        images[number] = images[number].transpose(Image.ROTATE_90)

        images[number] = np.asarray(images[number])

    return images


# Currently, there needs to have been saved some calibration information
# containing information about the filenames of the plane images
# This can be generated with the help of the Jupyter notebook script
# Ideally we'll find a way to not hard code image numbers eventually
def load_graph_images(
        folder,
        image_numbers=None,
        plane_numbers=None,
        calibration_filename=None,
        *args,
        **kwargs,
    ):
    if calibration_filename is None:
        calibration_filename = folder + "/calibration_information"
    name_dict = load_dictionary(calibration_filename)["plane_names"]
    if plane_numbers is None:
        plane_numbers = np.arange(len(name_dict))
    all_images = np.zeros(len(plane_numbers), dtype=object)
    all_contents = os.listdir(folder)

    for i, plane in enumerate(plane_numbers):
        for name in all_contents:
            if name_dict[plane] in name:
                image_folder = name
                break
        images_dict = load_image_set(
            filename=folder + "/" + image_folder,
            image_numbers=image_numbers,
            *args,
            **kwargs,
        )

        all_images[i] = images_dict

    return all_images