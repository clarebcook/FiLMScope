# run this script to select approximate alignment points from a calibration dataset
# set "image_filename" to the file location of any image acquired with this FiLM-Scope
# with identifiable features
# set "example_only" to True to test the GUI without saving information

from matplotlib import pyplot as plt
import os

from filmscope.util import load_image_set, load_graph_images
from filmscope.calibration import CalibrationInfoManager
from filmscope.config import path_to_data

# image to be used to select the approximate alignment points
path_to_data = "D:/20241226_fluoro_chicken"
image_folder = path_to_data
# image_filename = path_to_data + '/graph2_20241208_140546_003.nc'#/calibration_data/graph_03_20240812_114900_196.nc'

# if example_only is True,
# this will use the example calibration filename and not save deleted vertices
example_only = False
if example_only:
    calibration_filename = (
        path_to_data + "/calibration_data/calibration_information_example"
    )
else:
    calibration_filename = path_to_data + "/calibration_information"

assert os.path.exists(calibration_filename)

calibration_manager = CalibrationInfoManager(calibration_filename)
image_numbers = calibration_manager.image_numbers

# set "camera_number" to start with an image other than the first
# this is useful if some alignment points have already been selected
global camera_number
global camera_number_index
camera_number_index = 0
camera_number = image_numbers[camera_number_index]

# finish setup
display_downsample = 4
# image_set = load_image_set(filename=image_filename,
#                            image_numbers=image_numbers,
#                            downsample=display_downsample)

# this could be replaced with a different image set
image_set = load_graph_images(
    folder=image_folder,
    image_numbers=image_numbers,
    plane_numbers=[2],
    calibration_filename=calibration_filename,
    downsample=display_downsample,
)[0]

points_dict = {}


def get_title():
    return f"Double click to select alignment point for camera {camera_number}"


def select_point(event):
    if not event.dblclick:
        return

    global camera_number
    global camera_number_index

    ix, iy = event.xdata, event.ydata
    points_dict[camera_number] = (
        int(iy * display_downsample),
        int(ix * display_downsample),
    )

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
