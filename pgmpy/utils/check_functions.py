"""
Contains simple check functions
"""
import numpy as np


def _check_1d_array_object(parameter, name_param):
    """
    Checks whether given parameter is a 1d array like object, and returns a numpy array object
    """
    if isinstance(parameter, (np.ndarray, list, tuple, np.matrix)):
        parameter = np.array(parameter)
        if parameter.ndim != 1:
            raise TypeError("{} should be a 1d array type object".format(name_param))
    else:
        raise TypeError("{} should be a 1d array type object".format(name_param))

    return parameter


def _check_length_equal(param_1, param_2, name_param_1, name_param_2):
    """
    Raises an error when the length of given two arguments is not equal
    """
    if len(param_1) != len(param_2):
        raise ValueError(
            "Length of {} must be same as Length of {}".format(
                name_param_1, name_param_2
            )
        )
