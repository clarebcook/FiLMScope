import numpy as np 
import json 
import tqdm as tqdm 


#### functions for loading and saving dictionaries


# makes sure integer keys are not of type np.int32 or np.int64
# this recusrively goes through possible multi-layer dictionary
def _clean_dict_keys_for_saving(dictionary):
    if type(dictionary) != dict:
        return dictionary

    temp_dict = {}
    for key, item in dictionary.items():
        item = _clean_dict_keys_for_saving(item)
        if type(key) == np.int32 or type(key) == np.int64:
            temp_dict[int(key)] = item
        else:
            temp_dict[key] = item
    return temp_dict

# function to make sure every numpy array in a value or dict
# is converted to a list for saving
def _recursive_numpy_to_list(item):
    # if the item is a numpy array, make it a list and return
    if type(item) == np.ndarray:
        return item.tolist()
    # if it's not a dictionary, return the item
    if type(item) != dict:
        return item

    # if it is a dictionary, recursively call function for every entry
    for key, value in item.items():
        item[key] = _recursive_numpy_to_list(value)

    return item

# just some shortcut functions for loading/saving dictionaries
def save_dictionary(dictionary, save_filename, clean=True):
    if clean:
        dictionary = _clean_dict_keys_for_saving(dictionary)
        dictionary = _recursive_numpy_to_list(dictionary)
    with open(save_filename, "w") as fp:
        json.dump(dictionary, fp)


# dictionaries with integer keys get converted to strings
# this recursively changes those keys back to ints
def make_keys_ints(dictionary):
    if type(dictionary) != dict:
        return dictionary

    temp_dict = {}
    for key, item in dictionary.items():
        item = make_keys_ints(item)
        if key.isdigit():
            temp_dict[int(key)] = item
        else:
            temp_dict[key] = item
    return temp_dict


def load_dictionary(filename, keys_are_ints=True):
    rawfile = open(filename)
    dictionary = json.load(rawfile)
    if keys_are_ints:
        dictionary = make_keys_ints(dictionary)
    return dictionary
