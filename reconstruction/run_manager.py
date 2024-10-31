from FiLMScope.datasets import FSDataset
from FiLMScope.models import VolumeConvNet
from FiLMScope.losses import UnSupLoss
from FiLMScope.recon_util import (tocuda, get_ss_volume_from_dataset,
                                  get_height_aware_vol_from_dataset)

import torch
from torch.utils.data import DataLoader
import torch.optim as optim

class RunManager:
    def __init__(self, config_dict, guide_map=None,
                 prev_model=None, global_mask=None):
        self.config_dict = config_dict
        self.run_args = config_dict["run_args"]
        self.info = config_dict["sample_info"]
        self.loss_w = config_dict["loss_weights"]

        self.guide_map = guide_map
        if self.guide_map is not None: 
            self.guide_map = self.guide_map.cuda()
        self.global_mask = global_mask
        if self.global_mask is not None:
            self.global_mask = self.global_mask.cuda()

        self.run_type = config_dict["run_type"]

        if self.run_type == "video":
            frame_number = self.run_args["start_frame"]
        else:
            frame_number = -1

        self.dataset = FSDataset(
            self.info["image_filename"],
            self.info["calibration_filename"],
            self.info["image_numbers"],
            self.info["downsample"],
            self.info["crop_values"],
            frame_number=frame_number,
            ref_crop_center=self.info["ref_crop_center"],
            crop_size=self.info["crop_size"],
            height_est=self.info["height_est"],
            blank_filename=self.info["blank_filename"]
        )

        self.image_loader = DataLoader(
            self.dataset,
            self.run_args["batch_size"],
            shuffle=self.run_args["loader_shuffle"],
            num_workers=self.run_args["loader_num_workers"],
            drop_last=self.run_args["drop_last"],
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
        self.reference_image = torch.from_numpy(self.dataset.reference_image).cuda()
        self.reference_shift_slopes = torch.from_numpy(
            self.dataset.ref_camera_shift_slopes
        ).cuda()
        self.depth_values = torch.linspace(
                self.info["depth_range"][0],
                self.info["depth_range"][1],
                self.run_args["num_depths"], dtype=torch.float32)
        self.depth_values = tocuda(self.depth_values)

        with torch.no_grad():
            if guide_map is None:
                volume, volume_sq = get_ss_volume_from_dataset(
                    self.dataset,
                    self.run_args["batch_size"],
                    self.depth_values,
                    get_squared=True,
                )
            else:
                volume, volume_sq = get_height_aware_vol_from_dataset(
                    self.dataset,
                    self.run_args["batch_size"],
                    self.depth_values,
                    self.guide_map,
                    get_squared=True
                )
            num_views = len(self.dataset)
            self.volume_variance = volume_sq.div_(num_views).sub_(
                volume.div_(num_views).pow_(2)
            )

        self.logger = None
        self.setup_logger()

    def setup_logger(self):
        print("WARNING: Logging is not yet set up")

    def run_forward_model(self):
        outputs = {}
        outputs["depth"]= self.model(
            self.volume_variance, 
            self.depth_values,
        )

        if self.guide_map is not None:
            # add the guide map to the depth 
            outputs["depth"] = outputs["depth"] + self.guide_map.unsqueeze(0)
        return outputs
    
    def train_sample(self, sample):
        from time import time 
        t0 = time()
        self.model.train()
        self.optimizer.zero_grad()
        t1 = time() 
        print("setup time", t1 - t0)

        t2 = time()
        outputs = self.run_forward_model()
        t3 = time() 
        print("forward time:", t3 - t2)

        # compute the loss
        loss_values = {}
        loss_outputs = {}
        t4 = time()
        sample_cuda = tocuda(sample)
        t5 = time() 
        print("moving sample to GPU:", t5- t4)

        t6 = time()
        main_loss, losses, loss_outputs = self.criterion(
            sample_cuda["imgs"],
            outputs["depth"],
            sample_cuda["warped_shift_slope_maps"],
            sample_cuda["inv_inter_camera_maps"],
            self.reference_shift_slopes,
            self.reference_image,
            sample_cuda["masks"],
            global_mask=self.global_mask,
        )
        t7 = time() 
        print("getting loss", t7 - t6)

        t8 = time()
        for key, item in losses.items():
            loss_values[key] = item
        for key, item in loss_outputs.items():
            loss_outputs[key] = item

        loss_values["total"] = main_loss
        main_loss.backward()
        self.optimizer.step()

        for key in ["warped_imgs", "masks"]:
            outputs[key] = loss_outputs[key]
        t9 = time() 
        print("other", t9 - t8)
        print("")
        return outputs, loss_values

    def run_epoch(self, i, log=False):
        epoch_warp_images = []
        epoch_mask_images = []
        numbers = []
        for sample in self.image_loader:
            numbers = numbers + sample['image_numbers'].tolist()
            outputs, loss_values = self.train_sample(sample)
            epoch_warp_images = epoch_warp_images + outputs["warped_imgs"]
            epoch_mask_images = epoch_mask_images + outputs["masks"]

            # self.logger.log_loss(loss_values)
        
        if log:
            print("Not yet set up for logging")
        #     self.recent_loss = []
        #     self.save_dict = self._get_save_dict(outputs, epoch_warp_images, epoch_mask_images, numbers)
        #     self.plot_dict = self._get_plot_dict(outputs, epoch_warp_images)
        #     self.logger.log_outputs(i, self.plot_dict, self.save_dict) 

        return epoch_mask_images, epoch_warp_images, numbers, outputs 

    def end(self):
        self.__del__()

    def __del__(self):
        if self.logger is not None:
            self.logger.end()