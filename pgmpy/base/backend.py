import numpy as np
import torch

from pgmpy.config import backend

def barray(arr, dtype):
    arr = np.array(arr, dtype=dtype)
    if backend == 'pytorch':
        return torch.from_numpy(arr, dtype=dtype)
    else:
        return arr


def bsum(arr, axis):
    if backend == 'pytorch':
        return torch.sum(arr, dim=axis)
    else:
        return np.sum(arr, axis=axis)


def bmax(arr, axis):
    if backend == 'pytorch':
        return torch.max(arr, dim=axis)[0]
    else:
        return np.max(arr, dim=axis)


def bdiv(arr, value):
    if backend == 'pytorch':
        return torch.div(arr, value)
    else:
        return np.divide(arr, value)


