import numpy as np
from functools import partial
from filmscope.util import load_dictionary, save_dictionary, make_keys_ints
import os


class PropertyDictMeta(type):
    def __new__(cls, name, bases, dct):
        basic_properties = dct.get("_basic_properties_", [])
        dict_properties = dct.get("_dict_properties_", [])
        list_properties = dct.get("_list_properties_", [])

        for prop_name in basic_properties + dict_properties + list_properties:
            dct[prop_name] = property(
                partial(cls._get_property, prop_name=prop_name),
                partial(cls._set_property, prop_name=prop_name),
            )

        return super().__new__(cls, name, bases, dct)

    @staticmethod
    def _set_property(obj, value, prop_name):
        obj._property_dict[prop_name] = value
        obj.save_all_info()

    @staticmethod
    def _get_property(obj, prop_name):
        return obj._property_dict[prop_name]

    def save_all_info(self):
        return


class CalibrationInfoManager(metaclass=PropertyDictMeta):
    # properties that are a single value
    _basic_properties_= ['expected_line_spacing', 'reference_plane', 'reference_camera',
                         'slope_coeff_order', 'inter_cam_shift_coeff_order',
                         'lsf_range', 'plane_separation_mm', 'pixel_size', 
                         'vertex_spacing_m', 'image_shape']
    
    # and properties that are dictionaries 
    _dict_properties_ = ['vertex_calib_threshold_values', "plane_names",
                         "crop_indices", "detected_lines", "all_vertices",
                         "ignore_images", "removed_lines",
                         "removed_points", "approx_alignment_points",
                         "inter_camera_shift_coeffs", "slope_coeffs"] 

    _property_dict = {}

    def __init__(self, filename, legacy_folder=None, allow_save=True):
        self.filename = filename
        self.allow_save = allow_save
        self._image_numbers = None
        # check if the filename exists, and if so, load it
        if os.path.exists(self.filename):
            self.load_all_info()

        # fill in the property dict
        for prop in self._basic_properties_:
            if prop not in self._property_dict:
                self._property_dict[prop] = None
        for prop in self._dict_properties_:
            if prop not in self._property_dict:
                self._property_dict[prop] = {}

        # if legacy folder is provided, load that information
        # maybe give a warning if you're overwriting information loaded from "filename"
        if legacy_folder is not None:
            self.load_legacy_information(legacy_folder)
            #if os.path.exists(filename):
            #    raise Warning("Information from legacy folder is overwriting saved information")

    # this may need to be modified
    # probably different types of files need to be loaded differently
    # and you'd need to check if they actually exist
    # for instance, slope_coeffs contains additional information
    def load_legacy_information(self, legacy_folder):
        basic_legacy_dict = {
            "plane_names": "/plane_name_dict",
            "crop_indices": "/crop_indices",
            "all_vertices": "/identified_vertices",
            "approx_alignment_points": "/approx_alignment_points",
        }

        for key, filename in basic_legacy_dict.items():
            if os.path.exists(legacy_folder + filename):
                self._property_dict[key] = load_dictionary(legacy_folder + filename)

        if os.path.exists(legacy_folder + "/vertex_parse_parameters"):
            dictionary = load_dictionary(legacy_folder + "/vertex_parse_parameters")
            self.vertex_calib_threshold_values = dictionary["custom_threshold_values"]
            self.ignore_images = dictionary["ignore_images"]

        def sort_coeff_dict(filename, param_name, order_param_name):
            dictionary = load_dictionary(legacy_folder + filename)
            self._property_dict[order_param_name] = dictionary.pop("order")
            self.reference_camera = dictionary.pop("reference_camera")
            self.reference_plane = dictionary.pop("reference_plane")
            self._property_dict[param_name] = dictionary

        sort_coeff_dict('/inter_camera_shift_coeffs', 'inter_camera_shift_coeffs',
                        'inter_cam_shift_coeff_order')
        sort_coeff_dict('/slope_coeffs', 'slope_coeffs', 'slope_coeff_order')
        self.save_all_info()

    @property 
    def properties(self):
        properties = (self._basic_properties_ + self._dict_properties_ + 
                      ["filename", "image_numbers", "plane_numbers", "is_single_image"])
        return properties

    @property 
    # this is using the identifed vertices to identify
    # which image numbers were used in calibration
    # If the value is not explicitly set, it will use the image numbers
    # of all identified vertices 
    def image_numbers(self):
        if self._image_numbers is None:
            plane0 = self.plane_numbers[0] 
            vert_dict = self.all_vertices[plane0]
            return [i for i in vert_dict]
        else:
            return self._image_numbers

    @image_numbers.setter
    def image_numbers(self, val):
        self._image_numbers = np.asarray(val)

    @property 
    def plane_numbers(self):
        return [i for i in self.plane_names.keys()]
    
    @property 
    def is_single_image(self):
        return not len(self.crop_indices) == 0

    def load_all_info(self):
        # load the dictionary
        self._property_dict = load_dictionary(self.filename, keys_are_ints=False)
        # and modify whatever is needed to get the keys in the right format
        for property, prop_value in self._property_dict.items():
            self._property_dict[property] = make_keys_ints(prop_value)
        return

    def save_all_info(self):
        # copy dict so we can keep desired arrays as numpy arrays
        if not self.allow_save:
            return
        save_dict = self._property_dict.copy()
        save_dictionary(save_dict, self.filename)
