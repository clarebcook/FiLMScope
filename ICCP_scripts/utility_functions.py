import numpy as np

def count_needed_runs(experiment_dict, repeats, all_noise_stds, num_cameras):
    partials = []
    partial_cameras = []
    # and ideally if there's an incomplete set 
    # we'd pick up there 

    tracked_ids = []
    completed = 0
    for id, item in experiment_dict.items():
        #print(id, item) 
        if id in tracked_ids:
            continue 

        tracked_ids.append(id) 
        if len(item["cameras"]) != num_cameras:
            #print(len(item["cameras"]))
            continue 

        cameras = np.asarray(item["cameras"])
        completed_noise = [item["noise"][0]]
        for id2, item2 in experiment_dict.items():
            if id2 in tracked_ids:
                continue 

            #print(id2, item2) 

            cameras2 = item2["cameras"] 
            if len(cameras2) != num_cameras:
                continue 

            intersect = np.intersect1d(cameras, cameras2) 
            if len(intersect) != num_cameras:
                continue 

            # now we know these have the same camera set 
            completed_noise.append(item2["noise"][0])
            tracked_ids.append(id2)

        needed_noise = [i for i in all_noise_stds if i not in completed_noise]
        if len(needed_noise) == 0:
            completed +=1 
        else:
            partials.append(needed_noise)
            partial_cameras.append(cameras.tolist())

    needed_repeats = repeats - len(partials) - completed
    for r in range(needed_repeats):
        partials.append(all_noise_stds)
        partial_cameras.append(None)
    return partials, partial_cameras