"""Common API for torch and numpy backends."""

from copy import deepcopy

import numpy as np
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config

torch = _safe_import("torch")


def _is_torch_tensor(obj):
    if not _check_soft_dependencies("torch", severity="none"):
        return False

    return isinstance(obj, torch.Tensor)


def size(arr):
    if isinstance(arr, np.ndarray):
        return arr.size
    else:
        return arr.nelement()


def copy(arr):
    if config.get_backend() == "numpy":
        if isinstance(arr, np.ndarray):
            return np.array(arr)
        elif isinstance(arr, (int, float)):
            return deepcopy(arr)
        raise Exception(
            f"Invalid backend ({config.get_backend()}) for data type {type(arr)}"
        )
    else:
        if isinstance(arr, torch.Tensor):
            return arr.detach().clone()
        else:
            return torch.tensor(
                arr, dtype=config.get_dtype(), device=config.get_device()
            )


def tobytes(arr):
    if isinstance(arr, np.ndarray):
        return arr.tobytes()
    else:
        return arr.numpy(force=True).tobytes()


def max(arr, axis=None):
    if axis is not None:
        axis = tuple(axis)

    if isinstance(arr, np.ndarray):
        return np.max(arr, axis=axis)
    else:
        return torch.amax(arr, dim=axis)


def einsum(*args):
    if config.get_backend() == "numpy":
        return np.einsum(*args)
    else:
        return torch.einsum(*args)


def argmax(arr):
    if isinstance(arr, np.ndarray):
        return np.argmax(arr)
    else:
        return torch.argmax(arr)


def stack(arr_iter):
    if config.get_backend() == "numpy":
        return np.stack(tuple(arr_iter))
    else:
        return torch.stack(tuple(arr_iter))


def to_numpy(arr, decimals=None):
    if _is_torch_tensor(arr):
        if arr.device.type.startswith("cuda"):
            arr = arr.cpu().detach().numpy()
        else:
            arr = arr.numpy(force=True)

    if decimals is None:
        return np.array(arr)
    else:
        return np.array(arr).round(decimals)


def ravel_f(arr):
    if isinstance(arr, np.ndarray):
        return arr.ravel("F")
    else:
        return to_numpy(arr).ravel("F")


def ones(n):
    if config.get_backend() == "numpy":
        return np.ones(n, dtype=config.get_dtype())

    else:
        return torch.ones(n, dtype=config.get_dtype(), device=config.get_device())


def get_compute_backend():
    if config.get_backend() == "numpy":
        return np
    else:
        import torch

        return torch


def unique(arr, axis=0, return_counts=False, return_inverse=False):
    if isinstance(arr, np.ndarray):
        return np.unique(
            arr, axis=axis, return_counts=return_counts, return_inverse=return_inverse
        )
    else:
        return torch.unique(
            arr, return_inverse=return_inverse, return_counts=return_counts, dim=axis
        )


def flip(arr, axis=0):
    if isinstance(arr, np.ndarray):
        return np.flip(arr, axis=axis)
    else:
        return torch.flip(arr, dims=axis)


def transpose(arr, axis):
    if isinstance(arr, np.ndarray):
        return np.transpose(arr, axes=axis)
    else:
        return torch.permute(arr, dims=axis)


def exp(arr):
    if isinstance(arr, np.ndarray):
        return np.exp(arr)
    else:
        return arr.exp()


def sum(arr):
    if isinstance(arr, np.ndarray):
        return arr.sum()
    else:
        return torch.sum(arr)


def allclose(arr1, arr2, atol):
    if isinstance(arr1, np.ndarray) and isinstance(arr2, np.ndarray):
        return np.allclose(arr1, arr2, atol=atol)
    else:
        if isinstance(arr1, np.ndarray):
            arr1 = torch.tensor(
                arr1, dtype=config.get_dtype(), device=config.get_device()
            )
        else:
            arr1 = arr1.detach().clone()
        if isinstance(arr2, np.ndarray):
            arr2 = torch.tensor(
                arr2, dtype=config.get_dtype(), device=config.get_device()
            )
        else:
            arr2 = arr2.detach().clone()
        return torch.allclose(
            arr1,
            arr2,
            atol=atol,
        )


def zeros(size, dtype=None):
    if config.get_backend() == "numpy":
        return np.zeros(size, dtype=dtype or config.get_dtype())
    else:
        return torch.zeros(
            size, dtype=dtype or config.get_dtype(), device=config.get_device()
        )


def random_choice(values, size=1, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
        if torch.is_tensor(p) or torch.is_tensor(values):
            torch.manual_seed(seed)

    if isinstance(values, np.ndarray) or (
        isinstance(p, np.ndarray) if p is not None else False
    ):
        if isinstance(values, torch.Tensor):
            values = to_numpy(values)
        if p is not None and isinstance(p, torch.Tensor):
            p = to_numpy(p)
        return np.random.choice(values, size=size, p=p)

    elif isinstance(values, torch.Tensor) or (
        isinstance(p, torch.Tensor) if p is not None else False
    ):
        if not isinstance(values, torch.Tensor):
            values = torch.tensor(
                values, dtype=config.get_dtype(), device=config.get_device()
            )

        if p is None:
            idx = torch.randint(
                low=0, high=len(values), size=(size,), device=values.device
            )
            return values[idx]
        else:
            if not isinstance(p, torch.Tensor):
                p = torch.tensor(
                    p, dtype=config.get_dtype(), device=config.get_device()
                )

            cumsum = torch.cumsum(p, dim=0)
            rand_vals = torch.rand(size, device=p.device)
            idx = torch.searchsorted(cumsum, rand_vals)
            return values[idx]

    else:
        return np.random.choice(values, size=size, p=p)
