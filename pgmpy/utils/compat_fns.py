## Redefines function for pytorch and numpy backends, so that they have same behavior

import numpy as np
import torch


def size(arr):
    if isinstance(arr, np.ndarray):
        return arr.size
    else:
        return arr.nelement()


def copy(arr):
    if isinstance(arr, np.ndarray):
        return np.array(arr)
    else:
        return torch.clone(arr)


def tobytes(arr):
    if isinstance(arr, np.ndarray):
        return arr.tobytes()
    else:
        return arr.numpy(force=True).tobytes()


def max(arr, axis):
    if isinstance(arr, np.ndarray):
        return np.max(arr, axis=tuple(axis))
    else:
        return torch.amax(arr, dim=axis)
