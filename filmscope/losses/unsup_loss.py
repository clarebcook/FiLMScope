# modified from MVSNet

import torch
import torch.nn as nn
import torch.nn.functional as F

# from config import args, device
from .modules import SSIM, depth_smoothness
from filmscope.recon_util import inverse_warping

class UnSupLoss(nn.Module):
    def __init__(self, ssim_weight=6, smooth_weight=0.18, smooth_lambda=1):
        super(UnSupLoss, self).__init__()
        self.ssim = SSIM()

        self.ssim_weight = ssim_weight
        self.smooth_weight = smooth_weight
        self.smooth_lambda = smooth_lambda

    # images is [N, C, H, W]
    def forward(
        self,
        images,
        depth_est,
        warped_shift_slopes,
        inv_inter_camera_maps,
        ref_camera_shift_slopes,
        ref_img,
        individual_masks,
        global_mask=None, 
        rectify_perspective=True,
    ):
        # [B, H, W, 1] -> [B, 1, H, W] 
        individual_masks = individual_masks.permute(0, 3, 1, 2) 

        # if the depth map is rectified, the ref_camera_shift_slopes 
        # should be set to zero 
        if rectify_perspective:
            rcss = torch.zeros_like(ref_camera_shift_slopes)
        else:
            rcss = ref_camera_shift_slopes

        warped_images, masks  = inverse_warping(
            images, depth_est, rcss,
            inv_inter_camera_maps, warped_shift_slopes,
            rectify_perspective=rectify_perspective)
        
        masks = masks * individual_masks
        if global_mask is not None:
            masks = masks * global_mask


        # If the depth map is rectified, we need to warp the reference image
        # to the reference plane using the depth_est 
        if rectify_perspective:
            ref_img, _ = inverse_warping(ref_img[None], depth_est, torch.zeros_like(ref_camera_shift_slopes), 
                                    torch.zeros_like(ref_camera_shift_slopes), ref_camera_shift_slopes, 
                                        rectify_perspective=rectify_perspective)
            ref_img = ref_img.squeeze(0)
        
        ref_images = ref_img[None].repeat(images.shape[0], 1, 1, 1)
        batch_ssim = self.ssim(ref_images, warped_images, masks)
        self.ssim_loss = torch.mean(batch_ssim)

        # permute from shape [C, H, W] -> [B, H, W, C]
        self.smooth_loss = depth_smoothness(
            depth_est[None].permute(0, 2, 3, 1),
            ref_img[None].permute(0, 2, 3, 1), self.smooth_lambda
        )

        self.unsup_loss = (
            + self.ssim_weight * self.ssim_loss
            + self.smooth_weight * self.smooth_loss
        )

        loss_dict = {
            "unsup": self.unsup_loss,
            "ssim": self.ssim_loss,
            "smooth": self.smooth_loss,
        }

        outputs_dict = {
            "masks": masks,
            "warped_imgs": warped_images,
        }
        return self.unsup_loss, loss_dict, outputs_dict