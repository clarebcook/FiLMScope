import numpy as np
from matplotlib import pyplot as plt
import cv2


from ._parse_vertices_functions import (
    make_binary_image,
    condense_lines,
    find_approx_points,
    get_all_vertices,
)

from filmscope.util import display_with_lines, display_with_points
from .calibration_information_manager import CalibrationInfoManager


class SystemVertexParser:
    def __init__(
        self,
        calibration_filename,
        all_images,
        expected_vertex_spacing=None,
        camera_numbers=None,
        plane_numbers=None,
        display_downsample=4,
        lsf_range = None,
    ):
        # this is not being properly used yet
        self.display_downsample = display_downsample
        self.all_images = all_images

        self.calib_info_manager = CalibrationInfoManager(filename=calibration_filename)
        if expected_vertex_spacing is not None:
            self.calib_info_manager.expected_line_spacing = expected_vertex_spacing
        if self.calib_info_manager.expected_line_spacing is None:
            raise ValueError("expected_vertex_spacing must be provided if not already in calibration file.")

        # this is a little hacky
        # but I want to build in the ability to specify an lsf_range
        # without messing up my previous code which was using this value
        if lsf_range is None:
            self.calib_info_manager.lsf_range = int(self.calib_info_manager.expected_line_spacing / 2)
        else:
            self.calib_info_manager.lsf_range = lsf_range

        # also hacky, but get the first image
        # and use that to save the image shape
        key = [i for i in all_images[0].keys()][0] 
        self.calib_info_manager.image_shape = all_images[0][key].shape

        def make_nearest_odd(number):
            if number % 2 == 0:
                return number - 1
            else:
                return number

        self.default_threshold_values = {
            "adaptive_threshold_range": make_nearest_odd(self.calib_info_manager.expected_line_spacing),
            "blur_range": 15,
            "edge_thresh1": 50,
            "edge_thresh2": 150,
            "edge_aperture": 5,
            "line_thresh_per_pixel": 280 / 3000,
        }

        if camera_numbers is None:
            camera_numbers = np.arange(self.all_images.shape[1])
        self.camera_numbers = np.asarray(camera_numbers, dtype=int)
        self.camera_numbers = self.camera_numbers.astype(int)

        if plane_numbers is None:
            plane_numbers = np.arange(self.all_images.shape[0])
        self.plane_numbers = np.asarray(plane_numbers, dtype=int)
        self.plane_numbers = self.plane_numbers.astype(int)

    def save_all_parameters(self):
        self.calib_info_manager.save_all_info()
        #save_dictionary(self.params_dict, self.folder + self.parse_parameters_filename)

    def plane_index(self, number):
        return np.where(self.plane_numbers == number)[0][0]

    def display_images(self, plane_number=None, camera_number=None):
        if plane_number is not None:
            for i, image in self.all_images[self.plane_index(plane_number)].items():
                plt.figure()
                plt.imshow(
                    image[:: self.display_downsample, :: self.display_downsample]
                )
                plt.title(f"camera: {i}, plane: {plane_number}")
            return
        if camera_number is not None:
            for i, image in enumerate(self.plane_sorted_images[camera_number]):
                plt.figure()
                plt.imshow(
                    image[:: self.display_downsample, :: self.display_downsample]
                )
                plt.title(f"camera: {camera_number}, plane: {i}")

    def add_threshold_values(self, camera_number, plane_number, threshold_values):
        thresh_dict = self.calib_info_manager.vertex_calib_threshold_values
        if camera_number not in thresh_dict:
            thresh_dict[int(camera_number)] = {}
        thresh_dict[int(camera_number)][int(plane_number)] = threshold_values

    def add_ignored_image(self, camera_number, plane_number):
        image_dict = self.calib_info_manager.ignore_images
        if camera_number not in image_dict:
            image_dict[camera_number] = []
        image_dict[camera_number].append(plane_number)

    def check_ignore_image(self, plane_number, image_number):
        ignore_images = self.calib_info_manager.ignore_images
        possible_image_numbers = [num for num in ignore_images.keys()]
        if image_number not in possible_image_numbers:
            return False
        ignore_planes = ignore_images[image_number]
        if plane_number in ignore_planes:
            return True
        else:
            return False

    # getting into functions for actually parsing the vertices
    def _get_threshold_values(self, camera_number, plane_number):
        if not self.calib_info_manager.vertex_calib_threshold_values:
            return self.default_threshold_values

        threshold_dictionary = self.calib_info_manager.vertex_calib_threshold_values
        available_cameras = np.asarray([cam for cam in threshold_dictionary.keys()])
        closest_camera_index = np.where(
            np.abs(available_cameras - camera_number)
            == np.min(np.abs(available_cameras - camera_number))
        )[0][0]
        closest_camera = available_cameras[closest_camera_index]

        available_planes = np.asarray(
            [plane for plane in threshold_dictionary[closest_camera].keys()]
        )
        closest_plane_index = np.where(
            np.abs(available_planes - plane_number)
            == np.min(np.abs(available_planes - plane_number))
        )[0][0]
        closest_plane = available_planes[closest_plane_index]

        return threshold_dictionary[closest_camera][closest_plane]

    def find_lines(
        self,
        camera_number,
        plane_number,
        show=False,
        show_process=False,
        threshold_values=None,
        add_thresh_values_to_dict=True,
    ):
        if threshold_values is None:
            threshold_values = self._get_threshold_values(camera_number, plane_number)

        plane_index = self.plane_index(plane_number)
        image = self.all_images[plane_index][camera_number]
        binary = make_binary_image(image, threshold_values)

        edges = cv2.Canny(
            binary,
            threshold_values["edge_thresh1"],
            threshold_values["edge_thresh2"],
            threshold_values["edge_aperture"],
        )


        # this is a bit hacky, but I want to use different threshold values
        # for the two directions
        pixels0 = image.shape[0]
        pixels1 = image.shape[1]
        lines_thresh0 = cv2.HoughLines(edges, 2, np.pi / 180, int(threshold_values["line_thresh_per_pixel"] * pixels0))
        lines_thresh1 = cv2.HoughLines(edges, 2, np.pi / 180, int(threshold_values["line_thresh_per_pixel"] * pixels1)) 

        lines_dict0 = condense_lines(lines_thresh0, r_thresh=self.calib_info_manager.expected_line_spacing / 2)
        lines_dict1 = condense_lines(lines_thresh1, r_thresh=self.calib_info_manager.expected_line_spacing / 2)
        
        lines_dict = {
                "horizontal": lines_dict0["horizontal"],
                "vertical": lines_dict1["vertical"],
            }

        if plane_number not in self.calib_info_manager.detected_lines:
            self.calib_info_manager.detected_lines[plane_number] = {}
        self.calib_info_manager.detected_lines[plane_number][camera_number] = lines_dict

        if add_thresh_values_to_dict:
            self.add_threshold_values(camera_number, plane_number, threshold_values)

        if show_process:
            plt.figure()
            plt.imshow(binary)
            plt.title("Binary image")
            plt.figure()
            plt.imshow(edges)
            plt.title("Detected Lines")
            display_with_lines(
                image,
                lines_thresh0.squeeze(),
                xlength=image.shape[1],
                ylength=image.shape[0],
                display_downsample=self.display_downsample,
            )

            display_with_lines(
                image,
                lines_thresh1.squeeze(),
                xlength=image.shape[1],
                ylength=image.shape[0],
                display_downsample=self.display_downsample,
            )

        if show:
            all_lines = np.concatenate(
                (lines_dict["horizontal"], lines_dict["vertical"]), axis=0
            )
            # TODO: would be nice to put these on the same plot
            display_with_lines(
                image,
                all_lines,
                xlength=image.shape[1],
                ylength=image.shape[0],
                title=f"expanded lines for camera {camera_number}, plane {plane_number}",
                display_downsample=self.display_downsample,
            )
            display_with_lines(
                np.ones(image.shape),
                all_lines,
                xlength=image.shape[1],
                ylength=image.shape[0],
                title=f"expanded lines for camera {camera_number}, plane {plane_number}",
                display_downsample=self.display_downsample,
            )

    # this function finds lines for every image not yet included in the detected_lines dict
    def find_all_remaining_lines(self, show=True, max_display=20, threshold_values=None):
        num_displayed = 0
        for plane_number in self.plane_numbers:
            break_here = False
            for camera_number in self.camera_numbers:
                # we find all the expanded lines, but not vertices for some planes
                # if self.check_ignore_image(plane_number, camera_number):
                #    continue
                if (
                    plane_number in self.calib_info_manager.detected_lines
                    and camera_number in self.calib_info_manager.detected_lines[plane_number]
                ):
                    continue
                self.find_lines(camera_number, plane_number, show=show, threshold_values=threshold_values)
                num_displayed = num_displayed + 1
                if num_displayed >= max_display:
                    break_here = True
                    break
            if break_here:
                break

    def remove_line(
        self, camera_number, plane_number, direction, approx_loc, show=False
    ):
        lines = self.calib_info_manager.detected_lines[plane_number][camera_number][direction]
        lines = np.asarray(lines)
        line_locs = lines[:, 0]
        closest_index = np.argmin((line_locs - approx_loc) ** 2)
        deleted_line = lines[closest_index].tolist()

        lines = np.delete(lines, closest_index, axis=0)
        self.calib_info_manager.detected_lines[plane_number][camera_number][direction] = lines
        
        removed_lines_dict = self.calib_info_manager.removed_lines
        if plane_number not in removed_lines_dict:
            removed_lines_dict[plane_number] = {}
        if camera_number not in removed_lines_dict[plane_number]:
            removed_lines_dict[plane_number][camera_number] = []
        removed_lines_dict[plane_number][camera_number].append(deleted_line)

        if show:
            plane_index = self.plane_index(plane_number)
            image = self.all_images[plane_index][camera_number]
            lines_dict = self.calib_info_manager.detected_lines[plane_number][camera_number]
            all_lines = np.concatenate(
                (lines_dict["horizontal"], lines_dict["vertical"]), axis=0
            )
            # TODO: would be nice to put these on the same plot
            display_with_lines(
                image,
                all_lines,
                xlength=image.shape[1],
                ylength=image.shape[0],
                title=f"expanded lines for camera {camera_number}, plane {plane_number}",
                display_downsample=self.display_downsample,
            )

    def find_vertices(self, camera_number, plane_number, show=False):
        try:
            lines_dict = self.calib_info_manager.detected_lines[plane_number][camera_number]
        except Exception:
            raise Exception(
                f"Lines have not been found for cam {camera_number}, plane {plane_number}"
            )

        approx_points = find_approx_points(lines_dict)
        plane_index = self.plane_index(plane_number)
        image = self.all_images[plane_index][camera_number]

        threshold_values = self._get_threshold_values(camera_number, plane_number).copy()

        # 2024/05/30 test: when getting the vertices, we don't need to blur the binary image ?
        threshold_values["blur_range"] = 1
        vertices = get_all_vertices(
            image,
            approx_points,
            binary_threshold_values=threshold_values,
            show=show,
            lsf_range=self.calib_info_manager.lsf_range,
            display_title=f"Vertices for camera {camera_number}, plane {plane_number}",
            display_downsample=self.display_downsample,
        )

        all_vertices = self.calib_info_manager.all_vertices
        if plane_number not in all_vertices:
            all_vertices[int(plane_number)] = {}
        all_vertices[int(plane_number)][int(camera_number)] = vertices

    def find_all_remaining_vertices(self, show=True, max_display=20):
        num_displayed = 0
        all_vertices = self.calib_info_manager.all_vertices
        for plane_number in self.plane_numbers:
            for camera_number in self.camera_numbers:
                break_here = False
                if self.check_ignore_image(plane_number, camera_number):
                    continue
                if (
                    plane_number in all_vertices
                    and camera_number in all_vertices[plane_number]
                ):
                    continue
                self.find_vertices(camera_number, plane_number, show=show)
                num_displayed = num_displayed + 1
                if num_displayed >= max_display:
                    break_here = True
                    break
            if break_here:
                break

    def remove_point(self, camera_number, plane_number, approx_loc, show=False):
        vertices = self.calib_info_manager.all_vertices[plane_number][camera_number]
        vertices = np.asarray(vertices)
        closest_index = np.argmin(
            np.sqrt(
                (vertices[:, 0] - approx_loc[0]) ** 2
                + (vertices[:, 1] - approx_loc[1]) ** 2
            )
        )
        deleted_point = vertices[closest_index].tolist()
        vertices = np.delete(vertices, closest_index, axis=0)
        self.calib_info_manager.all_vertices[int(plane_number)][int(camera_number)] = vertices.tolist()

        removed_points_dict = self.calib_info_manager.removed_points
        if plane_number not in removed_points_dict:
            removed_points_dict[plane_number] = {}
        if camera_number not in removed_points_dict[plane_number]:
            removed_points_dict[plane_number][camera_number] = []
        removed_points_dict[plane_number][camera_number].append(deleted_point)

        self.calib_info_manager.removed_points
        if show:
            plane_index = self.plane_index(plane_number)
            image = self.all_images[plane_index][camera_number]
            display_with_points(
                image, np.asarray(vertices, dtype=np.uint16), radius=20, thickness=4
            )
    