
from datetime import datetime

def _get_conditional_indices(start0, end0, start1, end1, image_shape): 
    start0 = max(0, start0) 
    end0 = min(end0, image_shape[0]) 
    start1 = max(0, start1) 
    end1 = min(end1, image_shape[1]) 
    return int(start0), int(end0), int(start1), int(end1)

def get_crop_indices(midpoint, crop_shape, image_shape, return_adjustments=True):
    _start0 = int(midpoint[0] - crop_shape[0] / 2)
    _end0 = int(midpoint[0] + crop_shape[0] / 2)
    _start1 = int(midpoint[1] - crop_shape[1] / 2)
    _end1 = int(midpoint[1] + crop_shape[1] / 2)
    start0, end0, start1, end1 = _get_conditional_indices(_start0, _end0, _start1, _end1, image_shape) 
    if not return_adjustments: 
        return start0, end0, start1, end1 
    
    return start0, end0, start1, end1, start0 - _start0, end0 - _end0, start1 - _start1, end1 - _end1

def get_timestamp():
    str_format = "%Y%m%d_%H%M%S" 
    time_string = datetime.now().strftime(str_format) 
    return time_string
