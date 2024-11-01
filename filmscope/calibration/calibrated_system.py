# This is roughly based on previous versions
# But we are building up slowly to also incorporate tensorflow stuff
# that we've been working out elsewhere
import numpy as np
from matplotlib import pyplot as plt

from filmscope.util import (
    get_crop_indices,
    generate_x_y_vectors,
    generate_A_matrix,
)
from filmscope.calibration import CalibrationInfoManager
from ._vertices_organizing_functions import get_vertices_matrices


class Filmscope_System:
    def __init__(self, calibration_filename):
        self.calib_manager = CalibrationInfoManager(calibration_filename)

        # these can be computed and filled in as needed
        # keys will be camera numbers
        self.camera_magnification_information = {}

    @property
    def slope_coeffs_dict(self):
        return self.calib_manager.slope_coeffs

    @property
    def camera_shifts_coeffs_dict(self):
        return self.calib_manager.inter_camera_shift_coeffs

    @property
    def slope_order(self):
        return self.calib_manager.slope_coeff_order

    @property
    def shift_order(self):
        return self.calib_manager.inter_cam_shift_coeff_order

    @property
    def reference_plane(self):
        return self.calib_manager.reference_plane

    @property
    def reference_camera(self):
        return self.calib_manager.reference_camera

    # pulling into a separate function to retain control
    # if I later make changes to how coefficients are stored
    def _get_base_shift_coeffs(self, camera_number):
        # IMPORTANT NOTE THAT WE SHOULD MAYBE ADDRESS AT SOME POINT
        # all of these coefficients were generated for vertices in their DISPLAY coordinates
        # so here we end up needing to switch which ones correspond to x and y directions
        base_offset_info = self.camera_shifts_coeffs_dict[camera_number]
        base_0_shifts_coeffs = base_offset_info[1]  # in pixels
        base_1_shifts_coeffs = base_offset_info[0]

        return base_0_shifts_coeffs, base_1_shifts_coeffs

    def _get_slope_coeffs(self, camera_number):
        slope_info = self.slope_coeffs_dict[camera_number]
        slope_coeffs_0 = slope_info[1]
        slope_coeffs_1 = slope_info[0]

        return slope_coeffs_0, slope_coeffs_1

    # return the shift slopes at each point combination of x_coords and y_coords
    # x_coords, y_coords, and the returned values are in array coordinates
    # not display coordinates
    # this isn't actually tested for x_coords and y_coords lists longer than 1
    def get_shift_slopes(self, camera_number, x_coords, y_coords):
        coeff0, coeff1 = self._get_slope_coeffs(camera_number)
        # TODO: maybe we don't always need to re-generate the matrix?
        slope_matrix = generate_A_matrix(self.shift_order, X=y_coords, Y=x_coords)

        v1 = np.matmul(slope_matrix, coeff1)
        v0 = np.matmul(slope_matrix, coeff0)

        return v0, v1

    # shift to reference camera at reference plane
    # same caveats apply as for "get_shift_slopes"
    # BUG: TODO: 2024/05/06 I'm realizing this function does not work
    # when the coordinates are ints... need to figure that out
    def get_pixel_shifts(self, camera_number, x_coords, y_coords):
        coeff0, coeff1 = self._get_base_shift_coeffs(camera_number)

        slope_matrix = generate_A_matrix(self.shift_order, X=y_coords, Y=x_coords)

        shift0 = np.matmul(slope_matrix, coeff0)
        shift1 = np.matmul(slope_matrix, coeff1)
        return shift0, shift1

    # TODO: I may need to get rid of this and replace with somehting using tensorflow
    # it's a bit of a legacy thing just to quickly conduct some tests

    # x_start and y_start are in full image coordinates
    def _generate_shift_vectors(
        self, camera_number, height_map_shape, x_start=0, y_start=0
    ):
        # get X and Y vectors
        X, Y = generate_x_y_vectors(height_map_shape[0], height_map_shape[1])
        X = X + x_start
        Y = Y + y_start

        # find the base offset
        base_0_shifts_coeffs, base_1_shifts_coeffs = self._get_base_shift_coeffs(
            camera_number
        )

        base_shift_matrix = generate_A_matrix(self.shift_order, Y=X, X=Y)
        base_shift_vector_0 = np.matmul(base_shift_matrix, base_0_shifts_coeffs)
        base_shift_vector_1 = np.matmul(base_shift_matrix, base_1_shifts_coeffs)

        # find the slopes using the camera's slopes and the reference camera slopes
        slope_coeffs_0, slope_coeffs_1 = self._get_slope_coeffs(camera_number)

        slope_matrix = generate_A_matrix(self.slope_order, Y=X, X=Y)
        slope_vector_0 = np.matmul(slope_matrix, slope_coeffs_0)
        slope_vector_1 = np.matmul(slope_matrix, slope_coeffs_1)

        return base_shift_vector_0, base_shift_vector_1, slope_vector_0, slope_vector_1

    # these locations are in full image coordinates
    # we are predicting the plane a point in space lies on
    # if it appears at pixel_loc0 in camera0 and pixel_loc1 in camera1
    # note that the unit of the returned plane should be in mm
    # if calibration was run correctly
    def plane_from_pixel_locs(self, cam_number0, cam_number1, pixel_loc0, pixel_loc1):
        (
            base_vec_00,
            base_vec_01,
            slope_vec_00,
            slope_vec_01,
        ) = self._generate_shift_vectors(
            cam_number0,
            height_map_shape=(1, 1),
            x_start=pixel_loc0[0],
            y_start=pixel_loc0[1],
        )

        (
            base_vec_10,
            base_vec_11,
            slope_vec_10,
            slope_vec_11,
        ) = self._generate_shift_vectors(
            cam_number1,
            height_map_shape=(1, 1),
            x_start=pixel_loc1[0],
            y_start=pixel_loc1[1],
        )

        # 1 and 0 here refer to the camera numbers, not direction
        def get_plane(loc1, loc0, base_shift1, base_shift0, slope1, slope0):
            return ((loc1 - loc0) + (base_shift1 - base_shift0)) / (
                slope1 - slope0
            ) # + self.reference_plane 2024/04/22 switched to not adding this

        plane0 = get_plane(
            pixel_loc1[0],
            pixel_loc0[0],
            base_vec_10[0],
            base_vec_00[0],
            slope_vec_10[0],
            slope_vec_00[0],
        )

        plane1 = get_plane(
            pixel_loc1[1],
            pixel_loc0[1],
            base_vec_11[0],
            base_vec_01[0],
            slope_vec_11[0],
            slope_vec_01[0],
        )

        return plane0, plane1

    # if might not make sense to keep this long term
    # but for now it is helpful to be able to provide a list of matched points for 2 cameras
    # and get back all the heights
    def get_points_heights(self, cam_num0, cam_num1, points_dict):
        points0 = points_dict[cam_num0]
        points1 = points_dict[cam_num1]

        heights0 = []
        heights1 = []
        for point0, point1 in zip(points0, points1):
            # print(point0, point1)
            if point0[0] is not None and point1[0] is not None:
                plane0, plane1 = self.plane_from_pixel_locs(
                    cam_number0=cam_num0,
                    cam_number1=cam_num1,
                    pixel_loc0=(point0[0], point0[1]),
                    pixel_loc1=(point1[0], point1[1]),
                )
                heights0.append(plane0)
                heights1.append(plane1)
            else:
                heights0.append(np.nan)
                heights1.append(np.nan)

        heights0 = np.asarray(heights0)
        heights1 = np.asarray(heights1)
        heights0 = heights0 - np.min(heights0[~np.isnan(heights0)])  # in mm
        heights1 = heights1 - np.min(heights1[~np.isnan(heights1)])  # in mm

        return heights0, heights1

    @staticmethod
    def display_point_heights(image, heights, points, patch_size=None, show=True):
        if patch_size is None:
            patch_size = (image.shape[0] / 50, image.shape[0] / 50)

        height_map = np.ones(image.shape) * np.nan

        for height, point in zip(heights, points):
            if np.isnan(height):
                continue
            s0, e0, s1, e1 = get_crop_indices(
                (point[0], point[1]),
                patch_size,
                height_map.shape,
                return_adjustments=False,
            )
            height_map[s0:e0, s1:e1] = height

        plt.figure()
        plt.imshow(image, cmap="gray")
        plt.imshow(height_map)
        cbar = plt.colorbar()
        cbar.set_label("Height (mm)")
        plt.show()

        return height_map

    # need to add functions about getting the magnification at a certain height
    # I would recommend computing the slope/intercept of the magnification for a given camera once
    # then using it from there
    # dimension is array coordinates (not display coorinates)
    def get_camera_magnification_information(
        self, camera_number, dim, return_plane_mags=False
    ):
        if (
            camera_number in self.camera_magnification_information
            and not return_plane_mags
        ):
            if dim in self.camera_magnification_information[camera_number]:
                return self.camera_magnification_information[camera_number][dim]

        # get the organized vertices for this camera
        vertices_dict = self.calib_manager.all_vertices
        cam_vertices_dict = {}
        for plane_number, vertices in vertices_dict.items():
            cam_vertices_dict[plane_number] = np.asarray(vertices[camera_number])

        vertices_matrices = get_vertices_matrices(
            cam_vertices_dict, self.calib_manager.expected_line_spacing
        )

        # find the average ditsances between vertices at each plane
        plane_distances = []
        for plane_vertices in vertices_matrices:
            all_distances = []
            # this is switching 0 and 1 because the vertices
            # are stored in display coorddinates
            vertices_range = plane_vertices.shape[abs(1 - dim)]
            for i in range(vertices_range):
                if dim == 0:
                    point_list = plane_vertices[:, i]
                else:
                    point_list = plane_vertices[i]

                diffs_squared = np.power(point_list[1:] - point_list[:-1], 2)
                distances = np.sqrt(np.sum(diffs_squared, axis=1))
                distances = distances[~np.isnan(distances)]
                if len(distances) > 0:
                    all_distances = np.concatenate((all_distances, distances))
            plane_distances.append(np.mean(all_distances))

        # get magnification from vertex distances
        pixel_size_m = self.calib_manager.pixel_size
        vertex_spacing_m = self.calib_manager.vertex_spacing_m
        plane_spacing_mm = self.calib_manager.plane_separation_mm
        magnifications = np.asarray(plane_distances) * pixel_size_m / vertex_spacing_m
        intercept = magnifications[self.calib_manager.reference_plane]

        y_vec = np.arange(len(magnifications)) * plane_spacing_mm
        slope_mm, _ = np.polyfit(y_vec, magnifications, 1)

        if camera_number not in self.camera_magnification_information:
            self.camera_magnification_information[camera_number] = {}

        self.camera_magnification_information[camera_number][dim] = [
            slope_mm,
            intercept,
        ]
        if return_plane_mags:
            return slope_mm, intercept, magnifications
        else:
            return slope_mm, intercept

    def get_magnification_at_plane(self, camera_number, plane_mm, dim):
        slope, intercept = self.get_camera_magnification_information(camera_number, dim)
        return intercept + slope * plane_mm

    # this function takes the shift in pixels of a point in two cameras
    # and returns the estimated x, y, and z shift of that point in space
    def get_displacement_vector(
        self,
        cam1_delta_pixels,
        cam2_delta_pixels,
        cam1_loc,
        cam2_loc,
        cam1_num,
        cam2_num,
        dim,
        axial_plane_mm,
    ):
        cam1_slope = self.get_shift_slopes(cam1_num, [cam1_loc[0]], [cam1_loc[1]])[dim]
        cam2_slope = self.get_shift_slopes(cam2_num, [cam2_loc[0]], [cam2_loc[1]])[dim]

        # for each camera the total lateral shift delta would be given by
        # delta[pixels] = slope[pixels/mm] * dz[mm] + magnification * dx[mm] * 1 pixel / pixel_size_mm
        # dx = (delta - slope * dz) * pixel_size_mm / magnification
        # so with two camers we can say:
        # (delta_cam0-slope_cam0*dz)*pixel_size_mm/magnificaiton_cam0 = (delta_cam1-slope_cam1*dz)*pixel_size_mm/magnification_cam1
        # -> (delta_cam0-slope_cam0*dz)/magnificaiton_cam0 = (delta_cam1-slope_cam1*dz)/magnification_cam1
        # -> delta_cam0/magnificaiton_cam0 - slope_cam0*dz/magnification_cam0 = delta_cam1/magnificaiton_cam1 - slope_cam1*dz/magnification_cam1
        # -> dz*(slope_cam1/magnification_cam1 - slope_cam0/magnification_cam0) = delta_cam1/magnification_cam1-delta_cam0/mag_cam0

        mag_cam1 = self.get_magnification_at_plane(cam1_num, axial_plane_mm, dim)
        mag_cam2 = self.get_magnification_at_plane(cam2_num, axial_plane_mm, dim)

        numerator = (
            cam1_delta_pixels / mag_cam1 - cam2_delta_pixels / mag_cam2
        )  # units of pixels
        denom = cam1_slope / mag_cam1 - cam2_slope / mag_cam2  # units of pixels / mm
        dz_mm = numerator / denom

        dlat_cam1_mm = (
            (cam1_delta_pixels - cam1_slope * dz_mm)
            * self.calib_manager.pixel_size
            * 1e3
            / mag_cam1
        )
        dlat_cam2_mm = (
            (cam2_delta_pixels - cam2_slope * dz_mm)
            * self.calib_manager.pixel_size
            * 1e3
            / mag_cam2
        )

        return dz_mm, dlat_cam1_mm, dlat_cam2_mm