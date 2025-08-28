import torch
import torch.nn as nn
import torch.nn.functional as F


class srcSSIM(nn.Module):
    def __init__(self):
        super(srcSSIM, self).__init__()
        self.ssim = SSIM()

    # TODO: eventually should manage sizes better
    def forward(self, warped_images, masks): 
        N = warped_images.shape[0] 
        full_SSIM = None 
        for i in range(N - 1):
            x = warped_images[[i]] 
            y = warped_images[i + 1:]
            m = masks[i + 1:]
            if len(y.shape) == 3:
                y = y[None]
                m = m[None]
            # TODO: think of more clever way to handle masks
            # so one image with big mask doesn't mess everything up 
            loop_SSIM = self.ssim(x, y, m)
            if full_SSIM is None:
                full_SSIM = torch.zeros((warped_images.shape[0], warped_images.shape[1],
                                        loop_SSIM.shape[2], loop_SSIM.shape[3]),
                                        dtype=loop_SSIM.dtype).to(loop_SSIM.device)
            full_SSIM[i] = torch.mean(loop_SSIM * (N - i - 1), axis=0) 
        return full_SSIM




class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images"""

    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool = nn.AvgPool2d(3, 1)
        self.mu_y_pool = nn.AvgPool2d(3, 1)
        self.sig_x_pool = nn.AvgPool2d(3, 1)
        self.sig_y_pool = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)
        self.mask_pool = nn.AvgPool2d(3, 1)
        # self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01**2
        self.C2 = 0.03**2

    # x, y, and mask should be [B, C, H, W]
    def forward(self, x, y, mask):
        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)
        sigma_x = self.sig_x_pool(x**2) - mu_x**2
        sigma_y = self.sig_y_pool(y**2) - mu_y**2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y
        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x**2 + mu_y**2 + self.C1) * (sigma_x + sigma_y + self.C2)
        SSIM_mask = self.mask_pool(mask.to(x.dtype))
        output = torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1) * SSIM_mask
        return output


def gradient_x(img):
    return img[:, :, :-1, :] - img[:, :, 1:, :]


def gradient_y(img):
    return img[:, :-1, :, :] - img[:, 1:, :, :]


def gradient(pred):
    D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
    D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    return D_dx, D_dy

def depth_smoothness(depth, img, lambda_wt=1):
    """Computes image-aware depth smoothness loss."""
    depth_dx = gradient_x(depth)
    depth_dy = gradient_y(depth)
    image_dx = gradient_x(img)
    image_dy = gradient_y(img)
    weights_x = torch.exp(
        -(lambda_wt * torch.mean(torch.abs(image_dx), 3, keepdim=True))
    )
    weights_y = torch.exp(
        -(lambda_wt * torch.mean(torch.abs(image_dy), 3, keepdim=True))
    )
    smoothness_x = depth_dx * weights_x
    smoothness_y = depth_dy * weights_y
    return torch.mean(torch.abs(smoothness_x)) + torch.mean(torch.abs(smoothness_y))