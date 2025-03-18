import numpy as np
from matplotlib import pyplot as plt

from filmscope.util import (
    load_graph_images, display_with_points, least_squares_fit
)
from ._vertices_organizing_functions import align_vertices, get_vertices_matrices
from .calibration_information_manager import CalibrationInfoManager

class SystemCalibrator:
    
    def __init__(
        self,
        calibration_filename,
        camera_numbers=None,
        useable_plane_numbers=None,
        reference_plane=None,
        reference_camera=None,
        plane_separation_mm=None,
        ref_plane_image_folder=None,

    ):

        calibration_manager = CalibrationInfoManager(calibration_filename)
        if camera_numbers is None:
            camera_numbers = calibration_manager.image_numbers
        self.camera_numbers = camera_numbers
        
        if reference_plane is not None:
            calibration_manager.reference_plane = reference_plane 
        if calibration_manager.reference_plane is None:
            raise ValueError("Must provide reference plane")
        
        if reference_camera is not None:
            calibration_manager.reference_camera = reference_camera 
        if calibration_manager.reference_camera is None:
            raise ValueError("Must provide reference camera")
        
        if plane_separation_mm is not None:
            calibration_manager.plane_separation_mm = plane_separation_mm
        if calibration_manager.plane_separation_mm is None:
            raise ValueError("Must provide plane separation")
        self.calibration_manager = calibration_manager

        # BIG TODO: save in calibration information which plane numbers were used 
        if useable_plane_numbers is None:
            self.useable_plane_numbers = np.asarray([i for i in self.calibration_manager.all_vertices.keys()])
        else:
            self.useable_plane_numbers = useable_plane_numbers
        assert self.reference_plane in self.useable_plane_numbers
        
        # get the approximate inter-camera shifts
        self.inter_camera_shifts = {}
        ref_point = calibration_manager.approx_alignment_points[self.reference_camera]
        for camera_number, point in calibration_manager.approx_alignment_points.items():
            # these are switched back into DISPLAY coordinates
            self.inter_camera_shifts[camera_number] = (
                ref_point[1] - point[1],
                ref_point[0] - point[0],
            )

        if ref_plane_image_folder is not None:
            self.ref_plane_images = load_graph_images(
                ref_plane_image_folder,
                image_numbers=camera_numbers,
                plane_numbers=[reference_plane],
                calibration_filename=calibration_filename,
            )[0]
        else:
            self.ref_plane_images = None

    @property
    def reference_camera(self):
        return self.calibration_manager.reference_camera 
    
    @property
    def reference_plane(self):
        return self.calibration_manager.reference_plane
    
    @property
    def expected_spacing(self):
        return self.calibration_manager.expected_line_spacing
    
    @property
    def plane_separation_mm(self):
        return self.calibration_manager.plane_separation_mm

    ########## functions for inter-camera alignment ################
    def inter_camera_vertex_alignment(self, camera_number, show=False, shift_limit=60):
        reference_plane = self.calibration_manager.reference_plane 
        reference_camera = self.calibration_manager.reference_camera

        vertices = np.asarray(self.calibration_manager.all_vertices[reference_plane][camera_number])
        ref_vertices = np.asarray(
            self.calibration_manager.all_vertices[reference_plane][reference_camera]
        )

        expected_shift = self.inter_camera_shifts[camera_number]
        aligned_vertices, aligned_ref_vertices = align_vertices(
            vertices, ref_vertices, expected_shift, shift_limit=shift_limit
        )

        if show:
            ref_image = self.ref_plane_images[self.reference_camera]
            image = self.ref_plane_images[camera_number]
            display_with_points(
                image,
                aligned_vertices.astype(int),
                title=f"Camera {camera_number} with Aligned Vertices",
            )
            display_with_points(
                ref_image,
                aligned_ref_vertices.astype(int),
                title=f"reference camera with aligned vertices \n for camera {camera_number}",
            )

        return aligned_vertices, aligned_ref_vertices

    def all_inter_camera_vertex_alignment(self, show=False):
        self.all_aligned_reference_vertices = {}
        self.all_aligned_vertices = {}
        for camera_number in self.calibration_manager.all_vertices[self.reference_plane].keys():
            aligned_vertices, aligned_ref_vertices = self.inter_camera_vertex_alignment(
                camera_number, show=show, shift_limit=int(self.expected_spacing / 2),
            )

            self.all_aligned_vertices[camera_number] = aligned_vertices
            self.all_aligned_reference_vertices[camera_number] = aligned_ref_vertices

    # vertices and ref_vertices are assumed to be already aligned
    def fit_inter_camera_shifts(self, camera_number, order=2, show=False):
        vertices = self.all_aligned_vertices[camera_number]
        if len(vertices) == 0:
            return None
        ref_vertices = self.all_aligned_reference_vertices[camera_number]
        differences = ref_vertices - vertices

        X = vertices[:, 0].flatten()
        Y = vertices[:, 1].flatten()
        Z0 = differences[:, 0].flatten()
        Z1 = differences[:, 1].flatten()

        coeff0, A0, error0, z_predict0 = least_squares_fit(X, Y, Z0, order=order)
        squared_error0 = np.power(error0, 2)
        mse0 = np.mean(squared_error0)

        coeff1, A1, error1, z_predict1 = least_squares_fit(X, Y, Z1, order=order)
        squared_error1 = np.power(error1, 2)
        mse1 = np.mean(squared_error1)

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(121, projection="3d")
            ax.scatter(X, Y, Z0, label="Data")
            ax.scatter(X, Y, z_predict0, label="fit")
            ax.set_title("Direction 0, \n mse: %.3f pixels" % (mse0))
            ax.legend()

            ax = fig.add_subplot(122, projection="3d")
            ax.scatter(X, Y, Z1, label="Data")
            ax.scatter(X, Y, z_predict1, label="fit")
            ax.set_title("Direction 1, \n mse: %.3f pixels" % (mse1))
            ax.legend()

            fig.suptitle(
                f"camera {camera_number}, \n shifts to camera {self.reference_camera} at plane {self.reference_plane}"
            )

        return coeff0, coeff1

    def fit_all_inter_camera_shifts(self, order=2, show=False):
        self.calibration_manager.inter_cam_shift_coeff_order = order
        for camera_number in self.all_aligned_vertices.keys():
            coeffs = self.fit_inter_camera_shifts(camera_number, order=order, show=show)
            if coeffs is None:
                continue
            self.calibration_manager.inter_camera_shift_coeffs[camera_number] = [
                coeffs[0].tolist(),
                coeffs[1].tolist(),
            ]

    def save_inter_camera_coeffs(self):
        self.calibration_manager.save_all_info()

    def run_inter_camera_calibration(self, order=2, show=False):
        self.all_inter_camera_vertex_alignment()
        self.fit_all_inter_camera_shifts(order, show)
        self.save_inter_camera_coeffs()

    ########## functions for intra-camera alignment ################
    def get_camera_vertices_matrices(self, camera_number):
        all_vertices_dict = {}
        #for plane_number, cam_vertices in self.calibration_manager.all_vertices.items():
        for plane_number in self.useable_plane_numbers:
            cam_vertices = self.calibration_manager.all_vertices[plane_number]
            if camera_number not in cam_vertices:
                continue
            all_vertices_dict[plane_number] = np.asarray(cam_vertices[camera_number])
        return get_vertices_matrices(all_vertices_dict, self.calibration_manager.expected_line_spacing)

    def get_all_camera_vertices_matrices(self):
        self.all_camera_vertices_matrices = {}
        for camera_number in self.camera_numbers:
            vertices_matrices = self.get_camera_vertices_matrices(camera_number)
            self.all_camera_vertices_matrices[camera_number] = vertices_matrices

    def get_slopes_from_vertices_matrices(self, camera_number, show=False):
        vertices_matrices = self.all_camera_vertices_matrices[camera_number]
        matrix_shape = vertices_matrices.shape[1:3]
        slopes0 = np.zeros(matrix_shape)
        slopes0[:, :] = np.nan
        slopes1 = slopes0.copy()

        for i, j in np.ndindex(matrix_shape):
            vertex_line = vertices_matrices[:, i, j]
            good_indices = np.logical_not(np.isnan(vertex_line[:, 0]))

            # we need at least 2 non-nan points
            if len(np.where(good_indices == True)[0]) <= 1:
                continue

            index_start = np.where(good_indices == True)[0][0]
            index_end = np.where(good_indices == True)[0][-1]

            plane_range = index_end - index_start
            change0 = vertex_line[index_end, 0] - vertex_line[index_start, 0]
            change1 = vertex_line[index_end, 1] - vertex_line[index_start, 1]

            slopes0[i, j] = change0 / plane_range
            slopes1[i, j] = change1 / plane_range

        if show:
            X = vertices_matrices[self.reference_plane, :, :, 0].flatten()
            Y = vertices_matrices[self.reference_plane, :, :, 1].flatten()
            Z0 = slopes0.flatten()
            Z1 = slopes1.flatten()
            fig = plt.figure()
            ax = fig.add_subplot(121, projection="3d")
            ax.scatter(X, Y, Z0)
            ax.set_title("Slope 0 direction, pixels/plane")

            ax = fig.add_subplot(122, projection="3d")
            ax.scatter(X, Y, Z1)
            ax.set_title("Slope 1 direction, pixels/plane")

            fig.suptitle(f"camera {camera_number} slopes")
        return (
            slopes0,
            slopes1,
        )  # these are in pixels/plane shift, which will need to be converted to pixels/mm

    def fit_camera_slopes(self, camera_number, show=False, order=2):
        ref_plane_index = np.where(self.useable_plane_numbers==np.asarray(self.reference_plane))[0][0]
        vertices_matrix = self.all_camera_vertices_matrices[camera_number][
            ref_plane_index
        ]
        X = vertices_matrix[:, :, 0].flatten()
        Y = vertices_matrix[:, :, 1].flatten()

        slopes0, slopes1 = self.get_slopes_from_vertices_matrices(camera_number)
        slopes0_mm = slopes0 / self.plane_separation_mm
        slopes1_mm = slopes1 / self.plane_separation_mm

        Z0 = slopes0_mm.flatten()
        Z1 = slopes1_mm.flatten()

        # TODO: figure out how to get info for slopes when there aren't matching indices in the reference plane
        good_indices = np.logical_not(np.isnan(Z0))
        X = X[good_indices]
        Y = Y[good_indices]
        Z0 = Z0[good_indices]
        Z1 = Z1[good_indices]

        # definitely a more efficient way to do this, I need to understand when X and Z
        # don't have same "good_indices"
        good_indices = np.logical_not(np.isnan(X))
        X = X[good_indices]
        Y = Y[good_indices]
        Z0 = Z0[good_indices]
        Z1 = Z1[good_indices]

        # this is important: right now, X and Y are in DISPLAY coordinates
        # Z0 is also the 0 dim in display, and Z1 is 1 dim in display
        # so coeff0 and coeff1 are the coefficients for display directions
        # but also key: the A matrix will have to be generated using display x and y
        # at every point going forward. This will take some careful re-working. 
        coeff0, A0, error0, z_predict0 = least_squares_fit(X, Y, Z0, order=order)
        squared_error0 = np.power(error0, 2)
        mse0 = np.mean(squared_error0)

        coeff1, A1, error1, z_predict1 = least_squares_fit(X, Y, Z1, order=order)
        squared_error1 = np.power(error1, 2)
        mse1 = np.mean(squared_error1)

        if show:
            fig = plt.figure()
            ax = fig.add_subplot(121, projection="3d")
            ax.scatter(X, Y, Z0, label="Data")
            ax.scatter(X, Y, z_predict0, label="fit")
            ax.set_title("Slope Direction 0, \n mse: %.3f (pixels/mm)^2" % (mse0))
            ax.legend()

            ax = fig.add_subplot(122, projection="3d")
            ax.scatter(X, Y, Z1, label="Data")
            ax.scatter(X, Y, z_predict1, label="fit")
            ax.set_title("Direction 1, \n mse: %.3f (pixels/mm)^2" % (mse1))
            ax.legend()

            fig.suptitle(f"camera {camera_number} shift slopes")

        return coeff0, coeff1

    def fit_all_camera_slopes(self, order=2, show=False):
        self.calibration_manager.slope_coeff_order = order
        for camera_number in self.camera_numbers:
            coeffs = self.fit_camera_slopes(
                camera_number=camera_number, order=order, show=show
            )
            self.calibration_manager.slope_coeffs[int(camera_number)] = [
                coeffs[0].tolist(),
                coeffs[1].tolist(),
            ]

    def save_slope_coeffs(self):
        self.calibration_manager.save_all_info()

    def run_slope_calibration(self, order=2, show=False):
        self.get_all_camera_vertices_matrices()
        self.fit_all_camera_slopes(order, show)
        self.save_slope_coeffs()