import numpy as np
from matplotlib import pyplot as plt

from filmscope.util import load_image_set
from filmscope.calibration import CalibrationInfoManager
from filmscope.config import path_to_data

# image to be used to select the approximate alignment points
image_filename = path_to_data + '/calibration_data/graph_03_20240812_114900_196.nc'

# if example_only is True, 
# this will use the example calibration filename and not save deleted vertices
example_only = True 
if example_only:
    calibration_filename = path_to_data + '/calibration_data/calibration_information_example'
else:
    calibration_filename = path_to_data + '/calibration_data/calibration_information'
calibration_manager = CalibrationInfoManager(calibration_filename)

# be careful that these line up with the loaded vertices
image_numbers = calibration_manager.image_numbers
display_downsample = 4

image_set = load_image_set(filename=image_filename,
                           image_numbers=image_numbers,
                           downsample=display_downsample)
points_dict = {}

global camera_number
global camera_number_index 
camera_number_index = 0
camera_number = image_numbers[camera_number_index] 

def get_title():
    return f"Double click to select alignment point for camera {camera_number}"

def select_point(event):
    if not event.dblclick:
        return 

    global camera_number
    global camera_number_index 

    ix, iy = event.xdata, event.ydata 
    points_dict[camera_number] = (int(iy * display_downsample), int(ix * display_downsample))

    camera_number_index = camera_number_index + 1
    if camera_number_index >= len(image_numbers):
        plt.close()
        return
    camera_number = image_numbers[camera_number_index]

    image = image_set[camera_number]
    im.set_data(image)
    fig.suptitle(get_title()) 
    fig.canvas.draw()

fig, ax = plt.subplots(1, 1)
image = image_set[camera_number]
im = ax.imshow(image) 
fig.suptitle(get_title()) 
cid = fig.canvas.mpl_connect("button_press_event", select_point) 

plt.show()

if not example_only:
    calibration_manager.approx_alignment_points = points_dict
    calibration_manager.save_all_info()
