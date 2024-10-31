import numpy as np
import math
import xarray as xr 
from FiLMScope.calibration import generate_normalized_shift_maps
import cv2

from torch.utils.data import Dataset

from FiLMScope.util import load_dictionary, load_image_set


# crop_values should be (startx, endx, starty, endy) normalized to full image size
class FSDataset(Dataset):
    def __init__(
        self,
        image_filename,
        calibration_filename,
        image_numbers,
        downsample=1,
        crop_values=(0, 1, 0, 1),
        enforce_divisible=32, # set to -1 to not enforce
        frame_number=-1,
        blank_filename=None, # option to load in the image of just illumination reflections to subtract from data

        # if the arguments below are set to something other than None
        # it will override the provided "crop_values"
        # and do individual cropping for each image
        ref_crop_center=None,  # at least one of ref_crop_center or crop_centers should be None
        height_est=0,  # this should be set if ref_crop_center is set
        crop_centers=None,  # dictionary with image numbers as keys, (x, y) point in image coordinates as values
        crop_size=None,  # length 2 tuple with normalized crop size (i.e. values betwteen 0 and 1)
    ):

        self.blank_filename = blank_filename 
        
        self.image_numbers = np.asarray(image_numbers)
        self.image_filename = image_filename
        self.downsample = downsample
        self.frame_number = frame_number
        images = self.prep_images()
        
        # prepare the necessary maps
        # for legacy reasons these start out as numpy arrays
        # instead of torch tensors
        shape = (images.shape[0], images.shape[1], images.shape[2], 2)
        warped_shift_slope_maps = generate_normalized_shift_maps(
            calibration_filename, type="warped_shift_slope", downsample=downsample).numpy()
        inv_inter_camera_maps = generate_normalized_shift_maps(
            calibration_filename, type="inv_inter_camera", downsample=downsample).numpy()

        # identify the reference camera, which should be provided in every batch
        # and get the extra needed map
        reference_camera_num = load_dictionary(calibration_filename)["reference_camera"]

        # and load the additional needed map for the reference camera
        ref_camera_shift_slopes = generate_normalized_shift_maps(
            calibration_filename, type="shift_slope",
            image_numbers=[reference_camera_num], downsample=downsample).numpy()

        # figure out the size of the crop we will use
        if ref_crop_center is not None:
            centers = find_matching_points(
                ref_crop_center,
                height_est,
                warped_shift_slope_maps,
                inv_inter_camera_maps,
                ref_camera_shift_slopes,
            )
            crop_centers = {}
            for camera_number, crop_center in zip(image_numbers, centers):
                crop_centers[camera_number] = crop_center

        if crop_centers is None:
            full_startx = round(images.shape[1] * crop_values[0])
            full_endx = round(images.shape[1] * crop_values[1])
            full_starty = round(images.shape[2] * crop_values[2])
            full_endy = round(images.shape[2] * crop_values[3])
            lengthx = full_endx - full_startx
            lengthy = full_endy - full_starty
        else:
            lengthx = round(crop_size[0] * images.shape[1])
            lengthy = round(crop_size[1] * images.shape[2])

        # adjust that length if necessary
        if enforce_divisible != -1:
            diffx = next_multiple(lengthx, enforce_divisible) - lengthx
            diffy = next_multiple(lengthy, enforce_divisible) - lengthy

            lengthx = lengthx + diffx
            lengthy = lengthy + diffy

            if diffx != 0 and crop_centers is None:
                full_startx, full_endx = adjust_distance(
                    full_startx, full_endx, lengthx, 0, images.shape[1]
                )
            if diffy != 0 and crop_centers is None:
                full_starty, full_endy = adjust_distance(
                    full_starty, full_endy, lengthy, 0, images.shape[2]
                )
        if crop_centers is None:
            crop_startx = round(shape[1] * crop_values[0])
            crop_starty = round(shape[2] * crop_values[2])
            # I think I have to round up
            crop_endx = crop_startx + (full_endx - full_startx) 
            crop_endy = crop_starty + (full_endy - full_starty)

            self.full_crops = [full_startx, full_endx, full_starty, full_endy]
            self.map_crops = [crop_startx, crop_endx, crop_starty, crop_endy]

            images = images[:, full_startx:full_endx, full_starty:full_endy]

            warped_shift_slope_maps = crop_maps(warped_shift_slope_maps, self.map_crops)
            inv_inter_camera_maps = crop_maps(inv_inter_camera_maps, self.map_crops)
            ref_camera_shift_slopes = crop_maps(ref_camera_shift_slopes, self.map_crops)

            # TODO: this should use a different datatype
            masks = np.ones(
                inv_inter_camera_maps.shape[:3] + (1,), dtype=np.float32
            )
        # adjust inter camera maps based on individual crops, if necessary
        else:
            self.full_crops = {}
            self.map_crops = {}
            ref_center = crop_centers[reference_camera_num]

            # figure out the reference crop
            center = crop_centers[reference_camera_num]
            # based on lengthx, lengthy get the start and stop crop
            # for both full and downsampled
            center_pixels = (
                center[0] * images.shape[1],
                center[1] * images.shape[2],
            )

            startx = round(center_pixels[0] - lengthx / 2)
            starty = round(center_pixels[1] - lengthy / 2)
            endx = startx + lengthx
            endy = starty + lengthy

            ref_map_crops = [startx, endx, starty, endy]

            crop_images = np.zeros(
                (images.shape[0], lengthx, lengthy, images.shape[-1]), images.dtype
            )
            crop_warp_maps = np.zeros(
                (images.shape[0], lengthx, lengthy, 2),
                dtype=warped_shift_slope_maps.dtype,
            )
            crop_iis_maps = np.zeros(
                (images.shape[0], lengthx, lengthy, 2),
                dtype=inv_inter_camera_maps.dtype,
            )

            # TODO: should be some binary data type, not np.float32
            masks = np.ones(
                (images.shape[0], lengthx, lengthy, 1), dtype=np.float32
            )

            for i, image_number in enumerate(image_numbers):
                center = crop_centers[image_number]
                # based on lengthx, lengthy get the start and stop crop
                # for both full and downsampled
                center_pixels = (
                    center[0] * images.shape[1],
                    center[1] * images.shape[2],
                )

                startx = round(center_pixels[0] - lengthx / 2)
                starty = round(center_pixels[1] - lengthy / 2)
                endx = startx + lengthx
                endy = starty + lengthy
                
                # these are used to help create masks if the image crops fall outside the bounds
                # of the full images
                # if so, the images will be padded with zeros, and we'll save masks to keep track
                # of where those regions are
                # TODO: for now I'm ignoring the impact of these types of borders on the shift-and-sum volume
                # but eventually I should address that
                startx2 = max(startx, 0)
                starty2 = max(starty, 0) 
                endx2 = min(endx, images.shape[1] - 1) 
                endy2 = min(endy, images.shape[2] - 1)

                full_crops = [startx, endx, starty, endy]
                self.full_crops[image_number] = full_crops

                if endx2 != endx:
                    endx = endx2 - endx
                else:
                    endx = endx - startx
                if endy2 != endy:
                    endy = endy2 - endy
                else:
                    endy = endy - starty

                crop_images[i, startx2 - startx:endx, starty2 - starty:endy] = images[i, startx2:endx2, starty2:endy2]
                masks[i, :, :starty2 - starty] = 0
                masks[i, :, endy:] = 0
                masks[i, :startx2 - startx, :] = 0
                masks[i, endx:, :] = 0

                crop_iis_maps[i] = crop_maps(
                    inv_inter_camera_maps[i][None],
                    ref_map_crops,
                )
                crop_warp_maps[i] = crop_maps(
                    warped_shift_slope_maps[i][None], ref_map_crops
                )

                # find the shift from the reference center
                # and adjust the inter camera shift values accordingly
                shiftx = ref_center[0] - center[0]
                shifty = ref_center[1] - center[1]
                # adjust those shifts to be normalized for new image size
                ratiox = lengthx / images.shape[1]
                ratioy = lengthy / images.shape[2]
                shiftx = shiftx / ratiox
                shifty = shifty / ratioy
                # and divide by 2 to account for -1 to 1 normalizaiton
                shiftx = shiftx * 2
                shifty = shifty * 2

                crop_iis_maps[i, :, :, 0] = crop_iis_maps[i, :, :, 0] + shifty
                crop_iis_maps[i, :, :, 1] = crop_iis_maps[i, :, :, 1] + shiftx

                if image_number == reference_camera_num:
                    ref_camera_shift_slopes = crop_maps(
                        ref_camera_shift_slopes, ref_map_crops
                    )
            images = crop_images
            warped_shift_slope_maps = crop_warp_maps
            inv_inter_camera_maps = crop_iis_maps

        self.images = images.transpose([0, 3, 1, 2]).astype(np.float32)
        self.warped_shift_slope_maps = warped_shift_slope_maps
        self.inv_inter_camera_maps = inv_inter_camera_maps
        self.reference_camera = reference_camera_num
        self.ref_camera_shift_slopes = ref_camera_shift_slopes
        self.masks = masks.astype(bool)

    def __len__(self):
        return len(self.image_numbers)

    @property
    def reference_image(self):
        ref_camera_index = np.where(self.image_numbers == self.reference_camera)[0][0]
        image = self.images[ref_camera_index]
        return image

    def __getitem__(self, idx):
        return {
            "imgs": self.images[idx],
            "indices": idx,
            "image_numbers": self.image_numbers[idx],
            "warped_shift_slope_maps": self.warped_shift_slope_maps[idx],
            "inv_inter_camera_maps": self.inv_inter_camera_maps[idx],
            "masks": self.masks[idx], 
        }

    def prep_images(self):
        images_dict = load_image_set(
            filename=self.image_filename,
            image_numbers=self.image_numbers,
            downsample=self.downsample,
            frame_number=self.frame_number,
        )

        if self.blank_filename is not None:
            blank_images_dict = load_image_set(
                filename=self.blank_filename,
                image_numbers=self.image_numbers,
                downsample=self.downsample,
            )
            exposure = xr.open_dataset(self.image_filename).exposure.data.flatten()[0]
            blank_exposure = xr.open_dataset(self.blank_filename).exposure.data.flatten()[0]

        images = None
        for i, (image_num, image) in enumerate(images_dict.items()):
            if images is None:
                images = np.zeros(
                    (len(images_dict), image.shape[0], image.shape[1], 1),
                    dtype=np.float32,
                )

            if self.blank_filename is not None:
                blank_image = blank_images_dict[image_num] 
                blank_image = blank_image.astype(np.float32)

                # rehshape as blank image might not be same downsampled
                # the same as the dataset images
                blank_image = cv2.resize(blank_image, (image.shape[1], image.shape[0]))

                image = image.astype(np.float32)
                image = image / exposure - blank_image / (blank_exposure)
                image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255 
            
            images[i, :, :, 0] = image

        return images

    # this could easily be modified to allow for switching image sets
    def swap_frames(self, frame_number):
        if frame_number == self.frame_number:
            return
        
        self.frame_number = frame_number
        images = self.prep_images()

        # 2024/06/19
        # this is  new, and a lot of this is copy paste
        # should condense into a function that can be used here and in the __init__ func
        if isinstance(self.full_crops, dict):
            for image_number in self.image_numbers:
                startx, endx, starty, endy = self.full_crops[image_number]

                startx2 = max(startx, 0)
                starty2 = max(starty, 0) 
                endx2 = min(endx, images.shape[1] - 1) 
                endy2 = min(endy, images.shape[2] - 1)

                if endx2 != endx:
                    endx = endx2 - endx
                else:
                    endx = endx - startx
                if endy2 != endy:
                    endy = endy2 - endy
                else:
                    endy = endy - starty

                self.images[image_number, :, startx2 - startx:endx, starty2 - starty:endy] = images[image_number, startx2:endx2, starty2:endy2].transpose([2, 0, 1])
            return

        # crop
        full_startx = self.full_crops[0]
        full_endx = self.full_crops[1]
        full_starty = self.full_crops[2]
        full_endy = self.full_crops[3]
        images = images[:, full_startx:full_endx, full_starty:full_endy]

        self.images = images.transpose([0, 3, 1, 2]).astype(np.float32)


def next_multiple(number, multiple):
    remainder = number % multiple
    if remainder == 0:
        # The number itself is already divisible by 4
        return number
    else:
        # Calculate the next multiple of 4
        return number + (multiple - remainder)


def adjust_distance(x0, x1, new_diff, minx, maxx):
    adjust_amount = new_diff - (x1 - x0)
    lower_amt = math.floor(adjust_amount / 2)
    raise_amt = math.ceil(adjust_amount / 2)

    low_over = minx - (x0 - lower_amt)  # bad if this is positive
    high_over = (x1 + raise_amt) - maxx  # bad if this is positive

    if low_over + high_over > 0:
        raise ValueError("This adjustment cannot be done")

    if low_over > 0:
        lower_amt = lower_amt - low_over
        raise_amt = raise_amt + low_over
    if high_over > 0:
        lower_amt = lower_amt + high_over
        raise_amt = raise_amt - high_over

    new_x0 = x0 - lower_amt
    new_x1 = x1 + raise_amt

    assert new_x1 - new_x0 == new_diff

    return new_x0, new_x1

# crop maps according to provided values
# crop_values: [startx, endx, starty, endy]
# these should be in pixels to apply to original map
# this will return maps which can be used for
# images cropped the same amount
# maps should have shape [N, height, width, 2]
def crop_maps(maps, crop_values):
    start_shape = (maps.shape[1], maps.shape[2])

    startx = crop_values[0]
    endx = crop_values[1]
    starty = crop_values[2]
    endy = crop_values[3]

    # 1) crop the maps
    if startx >= 0 and starty >=0 and endx <= maps.shape[1] and endy <= maps.shape[2]:
        maps = maps[:, startx:endx, starty:endy]
    else:
        map_canvas = np.zeros((maps.shape[0], endx - startx, endy - starty, maps.shape[3]), dtype=maps.dtype)
        c_startx = max(startx, 0)
        c_starty = max(starty, 0)
        c_endx = min(endx, maps.shape[1])
        c_endy = min(endy, maps.shape[2]) 

        map_canvas[:, c_startx - startx:c_endx - startx,
                   c_starty - starty:c_endy - starty] = maps[:, c_startx:c_endx, c_starty:c_endy]
        maps = map_canvas

    end_shape = (maps.shape[1], maps.shape[2])

    # 2) scale values to new image size
    x_ratio = start_shape[0] / end_shape[0]
    y_ratio = start_shape[1] / end_shape[1]

    maps[:, :, :, 0] = maps[:, :, :, 0] * y_ratio
    maps[:, :, :, 1] = maps[:, :, :, 1] * x_ratio

    return maps

# this provides an approximate way to get good crops for the different images
# ref_loc - location in coordinates normalized 0 to 1 in the reference image
# height_est - estimated height in mm
def find_matching_points(
    ref_loc, height_est, warped_shift_slopes, inv_inter_cam_map, ref_warp_map
):
    ref_warp_map = ref_warp_map.squeeze()
    shape_pixels = warped_shift_slopes.shape[1:3]

    # shift reference location to reference plane, in reference camera coords
    ref_loc_pixels = (
        int(shape_pixels[0] * ref_loc[0]),
        int(shape_pixels[1] * ref_loc[1]),
    )
    ref_slope = ref_warp_map[ref_loc_pixels]
    ref_loc = (
        ref_loc[0] + height_est * ref_slope[1] / 2,
        ref_loc[1] + height_est * ref_slope[0] / 2,
    )
    ref_loc_pixels = (
        int(shape_pixels[0] * ref_loc[0]),
        int(shape_pixels[1] * ref_loc[1]),
    )

    coords = np.empty((warped_shift_slopes.shape[0], 2), dtype=np.float32)
    # loop through the maps
    for i, (warp_map, inter_map) in enumerate(
        zip(warped_shift_slopes, inv_inter_cam_map)
    ):
        # use the inter camera map to find reference plane coord in new camera
        shift = inter_map[ref_loc_pixels]
        camera_warp = warp_map[ref_loc_pixels]
        camera_loc = (
            ref_loc[0] + shift[1] / 2 - height_est * camera_warp[1] / 2,
            ref_loc[1] + shift[0] / 2 - height_est * camera_warp[0] / 2,
        )

        coords[i] = camera_loc

    return coords
