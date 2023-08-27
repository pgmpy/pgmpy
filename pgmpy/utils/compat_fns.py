## Redefines function for pytorch and numpy backends, so that they have same behavior

import numpy as np
import torch


def size(arr):
    from pgmpy import config

    if config.BACKEND == "numpy":
        return arr.size
    else:
        return arr.nelement()


def copy(arr):
    from pgmpy import config

    if config.BACKEND == "numpy":
        return np.array(arr)
    else:
        return torch.clone(arr)


def tobytes(arr):
    from pgmpy import config

    if config.BACKEND == "numpy":
        return arr.tobytes()
    else:
        return arr.numpy(force=True).tobytes()


def max(arr, axis=None):
    from pgmpy import config

    if axis is not None:
        axis = tuple(axis)

    if config.BACKEND == "numpy":
        return np.max(arr, axis=axis)
    else:
        return torch.amax(arr, dim=axis)


def einsum(*args):
    from pgmpy import config

    if config.BACKEND == "numpy":
        return np.einsum(*args)
    else:
        return torch.einsum(*args)


def argmax(arr):
    from pgmpy import config

    if config.BACKEND == "numpy":
        return np.argmax(arr)
    else:
        return torch.argmax(arr)


def stack(arr_iter):
    from pgmpy import config

    if config.BACKEND == "numpy":
        return np.stack(arr_iter)
    else:
        return torch.stack(tuple(arr_iter))
