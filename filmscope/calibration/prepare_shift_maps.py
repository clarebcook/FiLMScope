# this file contains the functions for generating dense mappings from calibration information

import torch 
from filmscope.calibration import CalibrationInfoManager
from filmscope.util import generate_x_y_vectors, generate_A_matrix
import numpy as np
#from filmscope.recon_util import generate_base_grid
from torch.nn.functional import grid_sample
from torchvision.transforms import Resize
import torch.nn.functional as F

# make the base grid that wee need to use F.grid_sample
def generate_base_grid(image_shape):
    y, x = torch.meshgrid(
        torch.arange(0, image_shape[0], dtype=torch.float32) / image_shape[0],
        torch.arange(0, image_shape[1], dtype=torch.float32) / image_shape[1],
    )
    base_grid = torch.stack([x, y], dim=-1)
    base_grid = base_grid * 2 - 1

    return base_grid[None]

# load and organize the coefficients just for this purpose
def _get_organized_coeffs_from_file(calibration_filename, image_numbers, shift_type):
    info_manager = CalibrationInfoManager(calibration_filename)
    if shift_type == "slope":
        coeffs = info_manager.slope_coeffs
    elif shift_type == "inter_camera":
        coeffs = info_manager.inter_camera_shift_coeffs
    else:
        raise ValueError("shift_type must be one of 'slope' or 'inter_camera'")

    coeffs0 = None
    coeffs1 = None
    for i, num in enumerate(image_numbers):
        image_coeffs = coeffs[num]
        if coeffs0 is None:
            coeffs0 = np.zeros((len(image_numbers), len(image_coeffs[0])), np.float64)
            coeffs1 = coeffs0.copy()

        # here we flip from display coordinates to ARRAY coordinates
        coeffs0[i] = image_coeffs[1]
        coeffs1[i] = image_coeffs[0]

    return torch.asarray(coeffs0), torch.asarray(coeffs1)

# written with the help of perplexity.ai
# this currently expects C=1
def _fill_unfilled_pixels(image, kernel_size=3):
    B, C, H, W = image.shape
    filled_image = image.clone()
    # Create a mask for filled (non-zero) pixels
    mask = (image != 0).to(image.dtype)
    # Compute the sum of neighboring pixel values and the count of valid neighbors
    sum_kernel = torch.ones(
        (1, 1, kernel_size, kernel_size), device=image.device, dtype=image.dtype
    )
    neighbor_sum = F.conv2d(image, sum_kernel, padding=kernel_size // 2)
    neighbor_count = F.conv2d(mask, sum_kernel, padding=kernel_size // 2)
    # Avoid division by zero
    neighbor_count = torch.clamp(neighbor_count, min=1)
    # Compute the average of valid neighbors
    neighbor_avg = neighbor_sum / neighbor_count
    # Fill unfilled pixels with the average of surrounding pixels
    filled_image = torch.where(mask == 0, neighbor_avg, image)
    return filled_image

def _calculate_padding(flow):
    max_flow_x = torch.ceil(torch.abs(flow[:, 0]).max()).int().item()
    max_flow_y = torch.ceil(torch.abs(flow[:, 1]).max()).int().item()
    max_flow_x = max(1, max_flow_x)
    max_flow_y = max(1, max_flow_y)
    return (max_flow_x, max_flow_y)

# def _old_dense_iamge_warp(image, flows):
#     flows = flows * 2
#     flows[:, :, :, 1] /= image.shape[1]
#     flows[:, :, :, 0] /= image.shape[2]
#     flows = torch.flip(flows, dims=[-1])
#     flows = flows * -1
# 
#     base_grid = generate_base_grid((image.shape[1], image.shape[2]))
#     grid = base_grid + flows
# 
#     image = image.permute(0, 3, 1, 2)
#     result = grid_sample(
#         input=image,
#         grid=grid.to(image.dtype),
#         align_corners=False,
#         padding_mode="border",
#     )
#     return result.permute(0, 2, 3, 1)



# This function performs forward dense warping 
# for instance, flowx = flows[b, h, w, 0], flowy = flows[b, h, w, 1]
# output[b, h + flow, w + flowy, c] = image[b, h, w, c]
# this function was written with the help of perplexity.ai
# it currently does not have sub-pixel accuracy
# image should be shape (batch, height, width, channels)
# flows should be shape (batch, height, width, 2)
# flows is in pixels
def dense_image_warp(image, flows):
    flows = flows.permute(0, 3, 1, 2)
    image = image.permute(0, 3, 1, 2)
    
    padding_x, padding_y = _calculate_padding(flows)
    padded_image = F.pad(
        image, (padding_y, padding_y, padding_x, padding_x), mode="replicate"
    )
    padded_flow = F.pad(
        flows, (padding_y, padding_y, padding_x, padding_x, 0, 0), mode="replicate"
    )

    B, C, H, W = image.shape
    device = image.device
    pH, pW = H + 2 * padding_x, W + 2 * padding_y

    x_coords, y_coords = torch.meshgrid(
        torch.arange(pH), torch.arange(pW), indexing="ij"
    )
    coords = torch.stack((x_coords, y_coords)).float().to(device)
    coords = coords.unsqueeze(0).repeat(B, 1, 1, 1)
    # Add flow to coordinates
    new_coords = coords + padded_flow
    # Round to nearest pixel
    new_coords = torch.round(new_coords).long()
    # Create valid mask
    valid_mask = (
        (new_coords[:, 0] >= 0)
        & (new_coords[:, 0] < pH)
        & (new_coords[:, 1] >= 0)
        & (new_coords[:, 1] < pW)
    )
    valid_mask = valid_mask.unsqueeze(1)  # Add channel dimension
    # Clip coordinates to image boundaries
    new_coords[:, 0] = torch.clamp(new_coords[:, 0], 0, pH - 1)
    new_coords[:, 1] = torch.clamp(new_coords[:, 1], 0, pW - 1)
    # Flatten indices for scatter_add_
    flat_indices = new_coords[:, 0] * pW + new_coords[:, 1]
    output = torch.zeros_like(padded_image)
    for b in range(B):
        for c in range(C):
            flat_image = padded_image[b, c].flatten()
            flat_image = flat_image * valid_mask[b, 0].flatten()
            result = (
                output[b, c]
                .flatten()
                .scatter_(0, flat_indices[b].flatten(), flat_image)
                .view(pH, pW)
            )
            result = _fill_unfilled_pixels(result[None, None])
            output[b, c] = result.squeeze()
    # crop the output back down
    output = output[:, :, padding_x:-padding_x, padding_y:-padding_y]
    return output.permute(0, 2, 3, 1)



# this will take coefficients as generated in calibration
# and return an map of a given shape/offset/downsampling
# with the corresponding x/y shifts for each point
def _get_shifts_from_coeffs(coeffs0, coeffs1, image_shape, 
                            matrix_order, downsample=1,
                            offset=(0, 0), inverse=False, batch_size=None):
    X, Y = generate_x_y_vectors(image_shape[0], image_shape[1])
    X = (X + offset[0]) * downsample
    Y = (Y + offset[1]) * downsample
    # because of how A matrix is used in calibration
    # we have to flip X and Y to use it ?
    # it's pretty weird this has been sort of working
    matrix = torch.asarray(generate_A_matrix(matrix_order, X=Y, Y=X))

    vectors0 = torch.zeros((len(coeffs0), len(X)))
    vectors1 = torch.zeros_like(vectors0)

    for image_number in range(len(coeffs0)):
        vectors0[image_number] = torch.matmul(matrix, coeffs0[image_number])
        vectors1[image_number] = torch.matmul(matrix, coeffs1[image_number])

    # need to reshape the slope vectors before saving,
    # and make it one thing
    new_shape = [vectors0.shape[0], image_shape[0], image_shape[1], 1]
    shifts = torch.concatenate(
        (vectors0.reshape(new_shape), vectors1.reshape(new_shape)),
        axis=3,
    )

    shifts = shifts / downsample

    if batch_size is None:
        batch_size = len(coeffs0)
    if inverse:
        size = len(coeffs0)
        break_points = np.arange(0, size, batch_size)
        break_points = np.append(break_points, size)
        for start, stop in zip(break_points[:-1], break_points[1:]):
            partial_shifts = shifts[start:stop]
            partial_shifts = (
                dense_image_warp(partial_shifts, partial_shifts) * -1
            )
            partial_shifts = partial_shifts
            shifts[start:stop] = partial_shifts
    
    return shifts

# type is one of:
# "shift_slope", "warped_shift_slope", "inter_camera", "inv_inter_camera"
def generate_pixel_shift_maps(calibration_filename, type, downsample=1, image_numbers=None,
                              batch_size=5, offset=(0, 0)):
    manager = CalibrationInfoManager(calibration_filename)

    full_image_shape = manager.image_shape
    if image_numbers is None:
        image_numbers = [i for i in manager.inter_camera_shift_coeffs.keys()]

    image_shape = (
        int(full_image_shape[0] / downsample),
        int(full_image_shape[1] / downsample),
    )

    if type == "shift_slope" or type == "warped_shift_slope":
        coeff_type = "slope" 
        matrix_order = manager.slope_coeff_order
    elif type == "inter_camera" or type == "inv_inter_camera": 
        coeff_type = "inter_camera"
        matrix_order = manager.inter_cam_shift_coeff_order
    else:
        raise ValueError("""invalid map type, must be one of
                         "shift_slope", "warped_shift_slope",
                         "inter_camera", "inv_inter_camera" """)
    
    coeffs0, coeffs1 = _get_organized_coeffs_from_file(calibration_filename=calibration_filename,
                                                       image_numbers=image_numbers,
                                                       shift_type=coeff_type)
    
    if coeff_type == "inter_camera":
        inverse = type == "inv_inter_camera"
        maps = _get_shifts_from_coeffs(
            coeffs0, coeffs1, image_shape=image_shape, 
            matrix_order=matrix_order, downsample=downsample,
            offset=offset, inverse=inverse, batch_size=batch_size
        )
        return maps 
    
    slope_map = _get_shifts_from_coeffs(
        coeffs0, coeffs1, image_shape=image_shape, 
        matrix_order=matrix_order, downsample=downsample,
        offset=offset, batch_size=batch_size
    )
    
    if type == "shift_slope":
        return slope_map 

    # then we have to warp the shift slope maps
    # using the inter camera maps
    inter_cam_map = generate_pixel_shift_maps(calibration_filename=calibration_filename,
                                               type="inter_camera",
                                               downsample=downsample,
                                               image_numbers=image_numbers,
                                               batch_size=batch_size,
                                               offset=offset)
    size = len(coeffs0)
    warped_shift_slopes = torch.zeros_like(slope_map)
    break_points = np.arange(0, size, batch_size)
    break_points = np.append(break_points, size)
    for start, stop in zip(break_points[:-1], break_points[1:]):
        partial_warps = inter_cam_map[start:stop]
        partial_slopes = slope_map[start:stop]

        partial_warp_slopes = dense_image_warp(partial_slopes, partial_warps)
        warped_shift_slopes[start:stop] = partial_warp_slopes
    return warped_shift_slopes 
    
def generate_normalized_shift_maps(calibration_filename, image_shape, gen_downsample=8,
                                   *args, **kwargs):
    maps = generate_pixel_shift_maps(calibration_filename=calibration_filename,
                                     downsample=gen_downsample, *args, **kwargs) 

    maps[:, :, :, 0] /= maps.shape[1]
    maps[:, :, :, 1] /= maps.shape[2]
    maps = torch.flip(maps, dims=[-1]) 
    maps *= 2

    # reshape to match potentially downsampled image
    transform = Resize(image_shape) 
    maps_t = transform(maps.permute(0, 3, 1, 2))
    return maps_t.permute(0, 2, 3, 1)
