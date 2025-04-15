from filmscope.datasets import FSDataset
from filmscope.models import VolumeConvNet
from filmscope.losses import UnSupLoss
from filmscope.config import path_to_data
from filmscope.recon_util import (tocuda, get_ss_volume_from_dataset,
                                  get_height_aware_vol_from_dataset)
from .log_manager import NeptuneLogManager
import torch
from torch.utils.data import DataLoader
import torch.optim as optim

class RunManager:
    def __init__(self, config_dict, guide_map=None,
                 prev_model=None, global_mask=None, run_name=None,
                 noise=[0,0]):
        self.run_name = run_name
        self.config_dict = config_dict
        self.run_args = config_dict["run_args"]
        self.rectify = self.run_args["rectify_perspective"]
        self.info = config_dict["sample_info"]
        self.loss_w = config_dict["loss_weights"]

        self.guide_map = guide_map
        if self.guide_map is not None: 
            self.guide_map = self.guide_map.cuda()
        self.global_mask = global_mask
        if self.global_mask is not None:
            self.global_mask = self.global_mask.cuda()

        frame_number = self.run_args["frame_number"]

        if self.info["blank_filename"] is not None:
            blank_filename = path_to_data + self.info["blank_filename"]
        else:
            blank_filename = None
        self.dataset = FSDataset(
            path_to_data + self.info["image_filename"],
            path_to_data + self.info["calibration_filename"],
            self.info["image_numbers"],
            self.info["downsample"],
            self.info["crop_values"],
            frame_number=frame_number,
            ref_crop_center=self.info["ref_crop_center"],
            crop_size=self.info["crop_size"],
            height_est=self.info["height_est"],
            blank_filename=blank_filename,
            noise=noise,
        )

        self.image_loader = DataLoader(
            self.dataset,
            self.run_args["batch_size"],
            shuffle=self.run_args["loader_shuffle"],
            num_workers=self.run_args["loader_num_workers"],
            drop_last=self.run_args["drop_last"],
            pin_memory=True
        )

        if prev_model is not None:
            self.model = prev_model
        else:
            self.model = VolumeConvNet(
                num_channels=1,
                num_layers=self.run_args["unet_layers"],
                layer_channels=self.run_args["unet_layer_channels"],
                layer_strides=self.run_args["unet_layer_strides"]
            ).cuda()

        assert self.run_args["optim"] == "Adam"
        optim_params = self.model.parameters() 
        self.optimizer = optim.Adam(
            optim_params,
            lr=self.run_args["lr"],
            betas=(0.9, 0.999),
            weight_decay=self.run_args["weight_decay"]
        )

        # make the criterion
        self.criterion = UnSupLoss(
            smooth_weight=self.loss_w["smooth"],
            ssim_weight=self.loss_w["ssim"],
            smooth_lambda=self.loss_w["smooth_lambda"]
        ).cuda()

        # prepare other things needed throughout reconstruction
        self.reference_image = self.dataset.reference_image.cuda()
        self.reference_shift_slopes = self.dataset.ref_camera_shift_slopes.cuda()
        self.depth_values = torch.linspace(
                self.info["depth_range"][0],
                self.info["depth_range"][1],
                self.run_args["num_depths"], dtype=torch.float32)
        self.depth_values = tocuda(self.depth_values)

        self.prepare_volume()

        self.dataset.to_device("cuda")

        self.logger = None
        if config_dict["use_neptune"]:
            self.setup_logger()

    def prepare_volume(self):
        with torch.no_grad():
            if self.guide_map is None:
                volume, volume_sq = get_ss_volume_from_dataset(
                    self.dataset,
                    self.run_args["batch_size"],
                    self.depth_values,
                    get_squared=True,
                    rectify_perspective=self.rectify
                )
            else:
                volume, volume_sq = get_height_aware_vol_from_dataset(
                    self.dataset,
                    self.run_args["batch_size"],
                    self.depth_values,
                    self.guide_map,
                    get_squared=True,
                    rectify_perspective=self.rectify
                )
            num_views = len(self.dataset)
            volume_variance = volume_sq.div_(num_views).sub_(
                volume.div_(num_views).pow_(2)
            )

            if not self.run_args["use_variance"]:
                self.volume = volume
            else:
                self.volume = volume_variance

    def setup_logger(self): 
        self.logger = NeptuneLogManager(
            dataset=self.dataset,
            model=self.model,
            config_dictionary=self.config_dict, 
            run_name=self.run_name
        )

    def swap_frames(self, frame_number):
        self.run_args["frame_number"] = frame_number

        # not sure why this is necessary
        # the the DataLoader used to make the volume 
        # fails if the data is not moved back to the cpu
        self.dataset.to_device("cpu")
        self.dataset.swap_frames(frame_number)
        self.reference_image = self.dataset.reference_image.cuda()
        self.prepare_volume()
        self.dataset.to_device("cuda")

        if self.config_dict["use_neptune"]:
            self.setup_logger()

    def run_forward_model(self):
        outputs = {}
        outputs["depth"]= self.model(
            self.volume, 
            self.depth_values,
        )

        if self.guide_map is not None:
            # add the guide map to the depth 
            outputs["depth"] = outputs["depth"] + self.guide_map.squeeze(-1)
        return outputs
    
    def train_sample(self, sample):
        self.model.train()
        self.optimizer.zero_grad()

        outputs = self.run_forward_model()

        # compute the loss
        loss_values = {}
        loss_outputs = {}

        main_loss, losses, loss_outputs = self.criterion(
            sample["imgs"],
            outputs["depth"],
            sample["warped_shift_slope_maps"],
            sample["inv_inter_camera_maps"],
            self.reference_shift_slopes,
            self.reference_image,
            sample["masks"],
            global_mask=self.global_mask,
            rectify_perspective=self.rectify, 
        )

        for key, item in losses.items():
            loss_values[key] = item
        for key, item in loss_outputs.items():
            loss_outputs[key] = item

        loss_values["total"] = main_loss
        main_loss.backward()
        self.optimizer.step()

        for key in ["warped_imgs", "masks"]:
            outputs[key] = loss_outputs[key]
        return outputs, loss_values

    def run_epoch(self, i, log=False):
        sample = self.dataset.get_full_sample()
        numbers = sample['image_numbers'].tolist()
        outputs, loss_values = self.train_sample(sample)
        warp_images = outputs["warped_imgs"]
        mask_images = outputs["masks"]

        if self.logger is not None:
            self.logger.log_loss(loss_values)
        
        if log:
            if self.logger is None:
                print("only set up for logging with Neptune")
            else:
                self.log_results(mask_images, warp_images, outputs, i) 

        return mask_images, warp_images, numbers, outputs, loss_values 

    def log_results(self, mask_images, warp_images, outputs, epoch):
        # log the model 
        self.logger.log_model(self.model, epoch)

        # log the summed masks and warped images
        sum_mask = torch.mean(mask_images.to(torch.float32), axis=0).squeeze().cpu().detach()
        self.logger.log_values(sum_mask, "summed_mask")
        sum_warp = torch.mean(warp_images, axis=0).squeeze().cpu().detach()
        self.logger.log_values(sum_warp, "summed_warp")

        # log depth as a plot and raw data
        depth = outputs["depth"].squeeze().cpu().detach()
        self.logger.log_values(depth, "depth")
        self.logger.log_value_plot(depth, "depth", epoch, cmap='turbo')
        return

    def end(self):
        self.__del__()

    def __del__(self):
        if self.logger is not None:
            self.logger.end()