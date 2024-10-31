# this is a file that is slightly different from "remove_identified_vertices.py"
# while I can probably find a way to better combine them,
# right now the loading in the other file is too slow when using an nc file directly
# so this file handles loading slightly differently 

# the purpose of this file is to make it easy to display all the vertices
# found with the calibration sequence.
# and visually select those that should be removed.

# We could think about eventually adding in an option for selecting new vertices
# but that might not be ideal.
import numpy as np
from matplotlib import pyplot as plt
import cv2

from FiLMScope.util import load_graph_images
from FiLMScope.calibration import CalibrationInfoManager

image_folder = "/media/Friday/Temporary/Clare/FiLMScope_paper_data/calibration_data"

display_downsample = 8

global current_plane, current_camera_index, current_image, current_camera
current_plane = 2
current_camera_index = 0
current_image_set = None
current_image = None

calibration_filename = None
if calibration_filename is None:
    calibration_filename = image_folder + '/calibration_information'

calibration_manager = CalibrationInfoManager(calibration_filename) 
vertices_dict = calibration_manager.all_vertices
if not vertices_dict:
    raise ValueError("You have not yet identified vertices for this image folder.")

plane_numbers = np.sort([x for x in vertices_dict.keys()])

def get_title():
    return f"Double click to remove point from camera {current_camera}, plane {current_plane} \n or double click out of canvas to proceed to next image"


def add_circles():
    image = current_image.copy() #all_images[current_plane][current_camera].copy()
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
                calibration_manager.save_all_info()
                plt.close()
                return
            # TODO: put up some sort of message that loading is happening?
            # idk if there's a good way to display what's happening
            current_image_set = load_graph_images(
                   folder=image_folder,
                   image_numbers=image_numbers,
                   plane_numbers=[current_plane],
                   downsample=display_downsample
               )[0]
            calibration_manager.save_all_info()
        else:
            current_camera_index = current_camera_index + 1
            current_camera = image_numbers[current_camera_index]
        current_image = current_image_set[current_camera]

        # we'll take this moment to save progress
        # calibration_manager.save_all_info()

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
                   downsample=display_downsample
               )[0]
current_camera = image_numbers[current_camera_index]
current_image = current_image_set[current_camera]

image = add_circles()
im = ax.imshow(image)  # fix this, maybe not [0][0]
fig.suptitle(get_title())
cid = fig.canvas.mpl_connect("button_press_event", remove_point)

plt.show()
calibration_manager.save_all_info()