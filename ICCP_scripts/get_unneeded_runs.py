from filmscope.config import log_folder, path_to_data, neptune_project, neptune_api_token
from filmscope.util import load_dictionary
import neptune 

max_num = 623 

dictionary_filenames = [
    "/finger_from_low_res.json",
    "/skull_frame_700_v2.json", 
    "/knuckle_frame_438.json", 
    "/stamp_runs_v3.json"
]

old_filenames = [
    "/skull_frame_700.json", 
    "/stamp_runs.json", 
    "/stamp_runs_v2.json"
]


dictionaries = [load_dictionary(log_folder + i) for i in dictionary_filenames]
old_dictionaries = [load_dictionary(log_folder + i) for i in old_filenames]

needed = [] 
not_needed = []
un_accounted = []
for i in range(623):
    key = f"IC-{i}"

    found = False 
    for item in dictionaries:
        if key in item:
            needed.append(key) 
            found = True 
            continue 
    if found:
        continue 

    for item in old_dictionaries:
        if key in item:
            not_needed.append(key) 
            found = True 
            continue 
    if found:
        continue 

    un_accounted.append(key) 




project = neptune.init_project(project=neptune_project,
                               api_token=neptune_api_token)

runs_table_df = project.fetch_runs_table().to_pandas()

ids = runs_table_df["sys/id"].values

for key in needed:
    assert key in ids 

need_to_trash = []
for key in not_needed:
    if key in ids: 
        need_to_trash.append(key)

for key in un_accounted:
    if key in ids: 
        need_to_trash.append(key)

     