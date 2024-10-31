# modified from MVSNet

import torch
import torch.nn as nn
import torch.nn.functional as F

# from config import args, device
from .modules import SSIM, depth_smoothness
from FiLMScope.recon_util import generate_base_grid, inverse_warping

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
        masks,
        global_mask=None, 
    ):
        ref_img = ref_img.unsqueeze(0).permute(0, 2, 3, 1)
        images = images.unsqueeze(0)

        images = torch.unbind(images, 1)
        iic_maps = torch.unbind(inv_inter_camera_maps, 0)
        warped_ss_maps = torch.unbind(warped_shift_slopes, 0)
        masks = torch.unbind(masks, 0)

        self.ssim_loss = 0
        self.reconstr_loss = 0
        self.smooth_loss = 0

        warped_img_list = []
        mask_list = []

        image_shape = (ref_img.shape[1], ref_img.shape[2])
        base_grid = generate_base_grid(image_shape)
        base_grid = base_grid.cuda()

        summed_warp = torch.zeros_like(ref_img)

        # TODO: why are we looping? I think we could do this in a batch
        # but I'll worry about that later
        for view_img, view_inv_inter_camera, view_warped_ss, ind_mask in zip(
            images, iic_maps, warped_ss_maps, masks
        ):
            view_img = view_img.permute(0, 2, 3, 1)  # [B, C, H, W] --> [B, H, W, C]
            # warp view_img to the ref_img using the dmap of the ref_img
            # for now, the reference camera has to be the one we calibrated with
            # but we can maybe change that in the future
            warped_img, mask = inverse_warping(
                view_img,
                depth_est,
                ref_camera_shift_slopes,
                view_inv_inter_camera,
                view_warped_ss,
                base_grid,
            )

            # multiply the two masks together
            mask = mask * ind_mask
            if global_mask is not None:
                mask = mask * global_mask

            warped_img = warped_img.permute(0, 2, 3, 1)

            summed_warp += warped_img

            warped_img_list.append(warped_img.cpu())
            mask_list.append(mask.cpu())

            loop_ssim = self.ssim(ref_img, warped_img, mask)
            self.ssim_loss += torch.mean(loop_ssim)
        
        # 2024/06/09 do the depth smoothness with summed warped images 
        # rather than the reference image
        self.smooth_loss += depth_smoothness(
            depth_est.unsqueeze(dim=-1), ref_img, self.smooth_lambda
        )

        self.reconstr_loss = 0
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
            "masks": mask_list,
            "warped_imgs": warped_img_list,
        }
        return self.unsup_loss, loss_dict, outputs_dict