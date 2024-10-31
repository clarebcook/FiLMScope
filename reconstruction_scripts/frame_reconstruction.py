from FiLMScope.reconstruction import RunManager, generate_config_dict 
from tqdm import tqdm

sample_name = "skull_with_tool"
gpu_number = "0"

config_dict = generate_config_dict(sample_name=sample_name, gpu_number=gpu_number, downsample=3,
                                   camera_set="all", run_args={"iters": 150})
run_manager = RunManager(config_dict)

iters = config_dict["run_args"]["iters"]
for i in tqdm(range(iters)):
    res = run_manager.run_epoch(i)