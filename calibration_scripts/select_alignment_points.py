import numpy as np
from matplotlib import pyplot as plt

from fourier_lightfield.util import load_image_set, save_dictionary
from fourier_lightfield.calibration import CalibrationInfoManager

calibration_folder = "C:/Users/clare/Downloads/temp"#"Z:/2024_05_13_skull_small_lens"
#image_filename = calibration_folder + '/graph_03_20240108_151306_975.nc'
# image_filename = calibration_folder + '/rat_skull_20240108_152431_993.nc'
image_filename = calibration_folder + '/graph_3_20240514_105313_279.nc'
set_type = "MCAM"
# be careful that these line up with the loaded vertices
image_numbers = np.arange(48)
display_downsample = 4

image_set = load_image_set(filename=image_filename,
                           set_type=set_type, 
                           image_numbers=image_numbers,
                           downsample=display_downsample)
points_dict = {}

global camera_number
global camera_number_index 
camera_number_index = 0
camera_number = image_numbers[camera_number_index] 

calibration_filename = None
if calibration_filename is None:
    calibration_filename = calibration_folder + '/calibration_information'
calibration_manager = CalibrationInfoManager(calibration_filename)

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

save_for_alignment = True 
if save_for_alignment:
    calibration_manager.approx_alignment_points = points_dict
    calibration_manager.save_all_info()
# just wanting to do something else with the points for now ...
else: 
    save_filename = image_filename[:-3] + '_crop_points.json'
    image_shape = image_set[0].shape
    for key, value in points_dict.items(): 
        value = (value[0] / (image_shape[0] * display_downsample), value[1] / (image_shape[1] * display_downsample))
        points_dict[key] = value 
    
    save_dictionary(points_dict, save_filename)