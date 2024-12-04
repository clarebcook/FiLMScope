# the purpose of this file is to make it easy to display all the vertices
# found with the calibration sequence.
# and manually select those that should be removed.

import numpy as np
from matplotlib import pyplot as plt
import cv2
import os

from filmscope.util import load_graph_images
from filmscope.config import path_to_data
from filmscope.calibration import CalibrationInfoManager

# path location of calibration dataset
image_folder = path_to_data + "/calibration_data"

# set "current_plane" and "current_camera_index" to values other than 0
# to start with plane/image other than the first one
# this can be done if this step is partially completed
global current_plane, current_camera_index, current_image, current_camera
current_plane = 0
current_camera_index = 0
current_image_set = None
current_image = None

# if example_only is True, 
# this will use the example calibration filename and not save deleted vertices
example_only = True
if example_only:
    calibration_filename = image_folder + '/calibration_information_example'
else:
    calibration_filename = image_folder + '/calibration_information'

assert os.path.exists(calibration_filename)

# finish setup  
display_downsample = 8

calibration_manager = CalibrationInfoManager(calibration_filename) 
image_numbers = calibration_manager.image_numbers
vertices_dict = calibration_manager.all_vertices
if not vertices_dict:
    raise ValueError("You have not yet identified vertices for this image folder.")

plane_numbers = np.sort([x for x in vertices_dict.keys()])

def get_title():
    return f"Double click to remove point from camera {current_camera}, plane {current_plane} \n or double click out of canvas to proceed to next image"


def add_circles():
    image = current_image.copy()
    points = vertices_dict[current_plane][current_camera]

    radius = 40
    color = (255, 0, 0)
    thickness = 4
    for i, point in enumerate(points):
        cv2.circle(image, (int(point[0] / display_downsample), int(point[1] / display_downsample)), int(radius / display_downsample), color, thickness)

    return image


def remove_point(event):
    if not event.dblclick:
        return
    global current_plane
    global current_camera_index
    global current_image
    global current_image_set 
    global current_camera

    ix, iy = event.xdata, event.ydata

    if ix is None or iy is None:
        # go on to the next image
        # if we're on the last camera for this plane
        if current_camera_index == len(image_numbers) - 1:
            current_camera_index = 0
            current_plane = current_plane + 1

            if current_plane == plane_numbers[-1] + 1:
                if not example_only:
                    calibration_manager.save_all_info()
                plt.close()
                return
            # TODO: put up some sort of message that loading is happening?
            # idk if there's a good way to display what's happening
            current_image_set = load_graph_images(
                   folder=image_folder,
                   image_numbers=image_numbers,
                   plane_numbers=[current_plane],
                   downsample=display_downsample,
                   calibration_filename=calibration_filename
               )[0]
            if not example_only:
                calibration_manager.save_all_info()
        else:
            current_camera_index = current_camera_index + 1
            current_camera = image_numbers[current_camera_index]
        current_image = current_image_set[current_camera]

        if not example_only:
            calibration_manager.save_all_info()

    else:
        # remove the selected point
        vertices = vertices_dict[current_plane][current_camera]
        vertices = np.asarray(vertices)
        closest_index = np.argmin(
            np.sqrt((vertices[:, 0] - ix * display_downsample) ** 2 + (vertices[:, 1] - iy * display_downsample) ** 2)
        )
        vertices = np.delete(vertices, closest_index, axis=0)
        vertices_dict[current_plane][current_camera] = vertices.tolist()

    im.set_data(add_circles())
    fig.suptitle(get_title())
    fig.canvas.draw()
    return


fig, ax = plt.subplots(1, 1)
current_image_set = load_graph_images(
                   folder=image_folder,
                   image_numbers=image_numbers,
                   plane_numbers=[current_plane],
                   downsample=display_downsample,
                   calibration_filename=calibration_filename
               )[0]
current_camera = image_numbers[current_camera_index]
current_image = current_image_set[current_camera]

image = add_circles()
im = ax.imshow(image)
fig.suptitle(get_title())
cid = fig.canvas.mpl_connect("button_press_event", remove_point)

plt.show()

if not example_only:
    calibration_manager.save_all_info()