from FiLMScope.reconstruction import RunManager, generate_config_dict 
from tqdm import tqdm

sample_name = "skull_with_tool"
gpu_number = "0"

config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=4,
                                   camera_set="all", use_neptune=True,
                                   run_args={"iters": 10, "batch_size": 12, "num_depths": 64,
                                             "display_freq": 5},)
run_manager = RunManager(config_dict)

iters = config_dict["run_args"]["iters"]
losses = []
for i in tqdm(range(iters)):
    log = (i % config_dict["run_args"]["display_freq"] == 0) or (i == iters - 1)
    mask_images, warp_images, numbers, outputs, loss_values = run_manager.run_epoch(i, log=log)
    losses.append(float(loss_values["total"]))

run_manager.end()