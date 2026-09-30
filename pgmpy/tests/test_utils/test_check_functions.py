import pytest
import numpy as np

from pgmpy.utils.check_functions import _check_1d_array_object, _check_length_equal

def test_check_1d_array_object_valid():
    # Valid 1D array-like inputs
    list_param = [1, 2, 3]
    tuple_param = (1, 2, 3)
    np_param = np.array([1, 2, 3])
    
    res_list = _check_1d_array_object(list_param, "list_param")
    res_tuple = _check_1d_array_object(tuple_param, "tuple_param")
    res_np = _check_1d_array_object(np_param, "np_param")
    
    assert isinstance(res_list, np.ndarray)
    assert isinstance(res_tuple, np.ndarray)
    assert isinstance(res_np, np.ndarray)
    
    np.testing.assert_array_equal(res_list, np.array([1, 2, 3]))
    np.testing.assert_array_equal(res_tuple, np.array([1, 2, 3]))
    np.testing.assert_array_equal(res_np, np.array([1, 2, 3]))

def test_check_1d_array_object_invalid():
    # Invalid types (not array-like)
    with pytest.raises(TypeError, match="invalid_type should be a 1d array type object"):
        _check_1d_array_object(10, "invalid_type")
        
    with pytest.raises(TypeError, match="invalid_type should be a 1d array type object"):
        _check_1d_array_object("string_param", "invalid_type")

def test_check_1d_array_object_multidim():
    # 2D array
    param_2d = np.array([[1, 2], [3, 4]])
    with pytest.raises(TypeError, match="param_2d should be a 1d array type object"):
        _check_1d_array_object(param_2d, "param_2d")

def test_check_length_equal():
    # Equal lengths
    param1 = [1, 2, 3]
    param2 = ["a", "b", "c"]
    _check_length_equal(param1, param2, "param1", "param2")  # Should not raise
    
    # Unequal lengths
    param3 = [1, 2]
    with pytest.raises(ValueError, match="Length of param1 must be same as Length of param3"):
        _check_length_equal(param1, param3, "param1", "param3")
