import numpy as np
import pytest
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config
from pgmpy.utils import compat_fns

torch = _safe_import("torch")


@pytest.fixture(autouse=True)
def reset_config():
    """Reset pgmpy config to defaults after each test."""
    yield
    config.set_backend("numpy")


def test_size_numpy():
    arr = np.array([[1, 2], [3, 4]])
    assert compat_fns.size(arr) == 4


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_size_torch():
    config.set_backend("torch")
    arr = torch.tensor([[1, 2], [3, 4]])
    assert compat_fns.size(arr) == 4


def test_copy_numpy():
    arr = np.array([1, 2, 3])
    arr_copy = compat_fns.copy(arr)
    assert np.array_equal(arr, arr_copy)
    assert arr is not arr_copy

    val = 5.0
    val_copy = compat_fns.copy(val)
    assert val == val_copy


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_copy_torch():
    config.set_backend("torch")
    arr = torch.tensor([1, 2, 3])
    arr_copy = compat_fns.copy(arr)
    assert torch.equal(arr, arr_copy)
    assert arr is not arr_copy

    val = 5.0
    val_copy = compat_fns.copy(val)
    assert torch.equal(torch.tensor(5.0), val_copy)


def test_tobytes_numpy():
    arr = np.array([1, 2])
    assert compat_fns.tobytes(arr) == arr.tobytes()


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_tobytes_torch():
    config.set_backend("torch")
    arr = torch.tensor([1, 2])
    assert compat_fns.tobytes(arr) == arr.numpy(force=True).tobytes()


def test_max_numpy():
    arr = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(compat_fns.max(arr, axis=[0]), np.array([3, 4]))
    np.testing.assert_array_equal(compat_fns.max(arr), 4)


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_max_torch():
    config.set_backend("torch")
    arr = torch.tensor([[1, 2], [3, 4]])
    assert torch.equal(compat_fns.max(arr, axis=[0]), torch.tensor([3, 4]))
    assert compat_fns.max(arr) == 4


def test_einsum_numpy():
    a = np.array([1, 2])
    b = np.array([3, 4])
    # Dot product
    assert compat_fns.einsum("i,i->", a, b) == 11


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_einsum_torch():
    config.set_backend("torch")
    a = torch.tensor([1, 2])
    b = torch.tensor([3, 4])
    assert compat_fns.einsum("i,i->", a, b) == 11


def test_argmax_numpy():
    arr = np.array([1, 5, 2])
    assert compat_fns.argmax(arr) == 1


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_argmax_torch():
    config.set_backend("torch")
    arr = torch.tensor([1, 5, 2])
    assert compat_fns.argmax(arr) == 1


def test_stack_numpy():
    a = np.array([1, 2])
    b = np.array([3, 4])
    stacked = compat_fns.stack((a, b))
    np.testing.assert_array_equal(stacked, np.array([[1, 2], [3, 4]]))


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_stack_torch():
    config.set_backend("torch")
    a = torch.tensor([1, 2])
    b = torch.tensor([3, 4])
    stacked = compat_fns.stack((a, b))
    assert torch.equal(stacked, torch.tensor([[1, 2], [3, 4]]))


def test_to_numpy():
    arr = [1, 2, 3]
    np_arr = compat_fns.to_numpy(arr)
    assert isinstance(np_arr, np.ndarray)
    np.testing.assert_array_equal(np_arr, np.array([1, 2, 3]))

    # Test decimals
    arr_float = [1.123, 2.456]
    np_arr_round = compat_fns.to_numpy(arr_float, decimals=1)
    np.testing.assert_array_equal(np_arr_round, np.array([1.1, 2.5]))


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_to_numpy_torch():
    config.set_backend("torch")
    arr = torch.tensor([1, 2, 3])
    np_arr = compat_fns.to_numpy(arr)
    assert isinstance(np_arr, np.ndarray)
    np.testing.assert_array_equal(np_arr, np.array([1, 2, 3]))


def test_ravel_f_numpy():
    arr = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(compat_fns.ravel_f(arr), np.array([1, 3, 2, 4]))


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_ravel_f_torch():
    config.set_backend("torch")
    arr = torch.tensor([[1, 2], [3, 4]])
    np.testing.assert_array_equal(compat_fns.ravel_f(arr), np.array([1, 3, 2, 4]))


def test_ones_numpy():
    ones_arr = compat_fns.ones(3)
    np.testing.assert_array_equal(ones_arr, np.array([1.0, 1.0, 1.0]))


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_ones_torch():
    config.set_backend("torch")
    ones_arr = compat_fns.ones(3)
    assert torch.equal(ones_arr, torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64))


def test_get_compute_backend():
    assert compat_fns.get_compute_backend() is np


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_get_compute_backend_torch():
    config.set_backend("torch")
    assert compat_fns.get_compute_backend() is torch


def test_unique_numpy():
    arr = np.array([1, 1, 2, 3, 3])
    uniques = compat_fns.unique(arr)
    np.testing.assert_array_equal(uniques, np.array([1, 2, 3]))


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_unique_torch():
    config.set_backend("torch")
    arr = torch.tensor([1, 1, 2, 3, 3])
    uniques = compat_fns.unique(arr)
    assert torch.equal(uniques, torch.tensor([1, 2, 3]))


def test_flip_numpy():
    arr = np.array([1, 2, 3])
    np.testing.assert_array_equal(compat_fns.flip(arr, axis=0), np.array([3, 2, 1]))


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_flip_torch():
    config.set_backend("torch")
    arr = torch.tensor([1, 2, 3])
    assert torch.equal(compat_fns.flip(arr, axis=0), torch.tensor([3, 2, 1]))


def test_transpose_numpy():
    arr = np.array([[1, 2], [3, 4]])
    np.testing.assert_array_equal(compat_fns.transpose(arr, axis=(1, 0)), np.array([[1, 3], [2, 4]]))


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_transpose_torch():
    config.set_backend("torch")
    arr = torch.tensor([[1, 2], [3, 4]])
    assert torch.equal(compat_fns.transpose(arr, axis=(1, 0)), torch.tensor([[1, 3], [2, 4]]))


def test_exp_numpy():
    arr = np.array([0, 1])
    np.testing.assert_array_equal(compat_fns.exp(arr), np.array([1.0, np.e]))


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_exp_torch():
    config.set_backend("torch")
    arr = torch.tensor([0, 1.0])
    assert torch.allclose(compat_fns.exp(arr), torch.tensor([1.0, np.e]))


def test_sum_numpy():
    arr = np.array([1, 2, 3])
    assert compat_fns.sum(arr) == 6


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_sum_torch():
    config.set_backend("torch")
    arr = torch.tensor([1, 2, 3])
    assert compat_fns.sum(arr) == 6


def test_allclose_numpy():
    arr1 = np.array([1.0, 2.0])
    arr2 = np.array([1.0, 2.0001])
    assert compat_fns.allclose(arr1, arr2, atol=1e-3)
    assert not compat_fns.allclose(arr1, arr2, atol=1e-5)


@pytest.mark.skipif(not _check_soft_dependencies("torch", severity="none"), reason="requires torch")
def test_allclose_torch():
    config.set_backend("torch")
    arr1 = torch.tensor([1.0, 2.0])
    arr2 = torch.tensor([1.0, 2.0001])
    assert compat_fns.allclose(arr1, arr2, atol=1e-3)
    assert not compat_fns.allclose(arr1, arr2, atol=1e-5)
