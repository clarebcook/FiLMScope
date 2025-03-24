
from filmscope.util import load_dictionary
import numpy as np 
from matplotlib import pyplot as plt 


path1 = "/media/Friday/Temporary/Clare/ICCP_result_storage"
path2 = path1 + '/round_2_results'

dict1 = load_dictionary(path1 + '/stamp_runs_v3.json')
dict2 = load_dictionary(path2 + '/stamp_runs_round2.json')


f1 = path1 + '/stamp_results_v3/IC-674_depth.npy'
f2 = path2 + '/stamp_results/run_8736_depth.npy'

map1 = np.load(f1) 
map2 = np.load(f2)


height_map = map1.copy()
dim0 = np.mean(height_map, axis=0)
# plt.plot(dim0)

slope0_1, intercept0 = np.polyfit(np.arange(len(dim0)), dim0, 1)

# plt.plot(np.arange(len(dim0)) * slope0 + intercept0)

for i, row in enumerate(height_map):
    height_map[i] = row - np.arange(len(row)) * slope0_1

dim1 = np.mean(height_map, axis=1) 
slope1_1, intercept1 = np.polyfit(np.arange(len(dim1)), dim1, 1)


print(slope0_1, slope1_1) 


height_map = map2.copy()
dim0 = np.mean(height_map, axis=0)
# plt.plot(dim0)

slope0_2, intercept0 = np.polyfit(np.arange(len(dim0)), dim0, 1)

# plt.plot(np.arange(len(dim0)) * slope0 + intercept0)

for i, row in enumerate(height_map):
    height_map[i] = row - np.arange(len(row)) * slope0_2

dim1 = np.mean(height_map, axis=1) 
slope1_2, intercept2 = np.polyfit(np.arange(len(dim1)), dim1, 1)
print(slope0_2, slope1_2) 


# in mm per pixel 
# so how many pixel per mm 
pixel_per_mm = 0.12 / 1.1e-3 
slope1_m= slope0_2 * pixel_per_mm
slope2_m = slope1_2 * pixel_per_mm


print(np.arctan(slope1_m) * 180 / np.pi)
print(np.arctan(slope2_m) * 180 / np.pi)