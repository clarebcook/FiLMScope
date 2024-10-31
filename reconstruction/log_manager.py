# just defining log/display functions that I don't want taking up space in the main run file
from matplotlib import pyplot as plt
import numpy as np
import os
import neptune
import warnings
from neptune.types import File
from FiLMScope.util import get_timestamp
from PIL import Image
import shutil
import torch

from FiLMScope.config import neptune_api_token as api_token 
from FiLMScope.config import neptune_project as project
from FiLMScope.config import log_folder


class NeptuneLogManager:
    def __init__(self, dataset, model, config_dictionary, run_name=None):
        # start by logging start information about the run
        self.loss_weights = config_dictionary["loss_weights"]
        sample_info = config_dictionary["sample_info"]
        if run_name is None:
            run_name = f"{sample_info['sample_name']}_{get_timestamp()}"

        neptune_run = neptune.init_run(project=project, api_token=api_token)
        self.neptune_run = neptune_run

        # log a few stand alone things
        neptune_run["run_name"] = run_name
        neptune_run["description"] = config_dictionary["log_description"]

        # then log all the sample info
        for key, value in sample_info.items():
            if isinstance(value, (list, tuple, np.ndarray)):
                for val in value:
                    neptune_run[f"sample/{key}"].append(val)
            elif value is None:
                neptune_run[f"sample/{key}"] = "None"
            else:
                neptune_run[f"sample/{key}"] = value

        # other dictionaries
        neptune_run["reconstruction/loss/weights"] = config_dictionary["loss_weights"]

        # Neptune doesn't handle None well, so we log that as a string
        run_args = config_dictionary["run_args"].copy()
        for key, item in run_args.items():
            if item is None:
                run_args[key] = "None"
            elif key == "unet_layer_channels" or key == "unet_layer_strides": 
                assert isinstance(item, list)
                run_args[key] = str(item)
        neptune_run["model/parameters/run_args"] = run_args

        self.log_folder = log_folder + f"/temp_log_folder_{get_timestamp()}"

        if not os.path.exists(self.log_folder):
            os.mkdir(self.log_folder)

        self.image_folder = self.log_folder + "/images"
        if not os.path.exists(self.image_folder):
            os.mkdir(self.image_folder)


        # log the images
        # right now, to save space, we only include the reference image
        image = dataset.reference_image.squeeze() 
        mask = dataset.masks.squeeze()[dataset.reference_index]

        im = Image.fromarray(image.cpu().numpy() * mask.cpu().numpy()) 
        im = im.convert("L")
        number = dataset.reference_camera
        filename = self.log_folder + f"/image_{number}.png" 
        im.save(filename)
        neptune_run[f"sample/images/image_{number}"].upload(filename)

        self.model_folder = self.log_folder + "/model_info"
        if not os.path.exists(self.model_folder):
            os.mkdir(self.model_folder)

        m_filename = self.model_folder + f"model_start_architecture.pth"
        torch.save(model, m_filename)
        neptune_run[f"model/architecture"].upload(m_filename)

    # tested for [H, W] tensors 
    # it is challenging to directly log tensors with Neptune
    # so images are converted to int8, and the maximum/minimum values are saved
    # for re-computing later 
    def log_values(self, values, name):
        self.neptune_run[f"reconstruction/values/{name}_min"].append(
            float(torch.min(values))
        )
        self.neptune_run[f"reconstruction/values/{name}_max"].append(
            float(torch.max(values))
        )

        # scale values [0, 1]
        values = values - torch.min(values)
        values = values / torch.max(values)

        self.neptune_run[f"reconstruction/values/{name}"].append(
            File.as_image(values)
        )
        return 
    
    def log_loss(self, loss_dict):
        for name, value in loss_dict.items():
            self.neptune_run[f"reconstruction/loss/{name}"].append(float(value))

    # this can make it easier to visualize on the neptune website
    def log_value_plot(self, values, name, epoch_number, *args, **kwargs):
        plt.figure()
        plt.imshow(values, *args, **kwargs)
        plt.colorbar() 
        plt.title(name + f" at epoch {epoch_number}")

        filename = self.image_folder + f"/{name}.png"
        plt.savefig(filename) 
        self.neptune_run[f"reconstruction/images/{name}"].append(File(filename))
        plt.close()
        return 
    
    def log_model(self, model, epoch):
        m_state_filename = (
            self.log_folder + f"/model_info/epoch_{epoch}.pth"
        )
        torch.save(model.state_dict(), m_state_filename)
        self.neptune_run[f"model/checkpoints/epoch{epoch}"].upload(
            m_state_filename
        )

    def end(self, delete_local=True):
        self.neptune_run.stop()

        if delete_local and os.path.exists(self.log_folder):
            shutil.rmtree(self.log_folder)

    def __del__(self):
        self.end()