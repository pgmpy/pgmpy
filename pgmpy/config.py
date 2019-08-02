import torch
import numpy as np


# Check if pandas is available
try:
    import pandas

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Check if GPU is available
def set_device(device_="cpu"):
    """
    Set the device to be used for pytorch.

    Parameters
    ----------
    device_: str (cpu | gpu)
        Device to use for computation.
    """
    global device
    if device_ == "cpu":
        device = torch.device("cpu")
    elif device == "gpu":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise ValueError("No cuda supported GPU available")
    else:
        raise ValueError("Unknown device. Please select between cpu or gpu")


# Set backend to use numpy or pytorch
def set_backend(backend_="numpy"):
    """
    Sets the backend framework to use.

    Parameters
    ----------
    backend: str (numpy | pytorch)
        The backend to use for computation.
    """
    global backend

    if backend_ in ["numpy", "pytorch"]:
        backend = backend_
    else:
        raise ValueError("Unknown backend. Please select between numpy or pytorch")


def set_dtype(dtype_="float32"):
    """
    Globally set dtype for all arrays.

    Parameters
    ----------
    dtype: str
        Options: float32 | float64 | float16 | uint8 | int8 | int16 | int32 | int64
    """
    global dtype

    if dtype_ in [
        "float32",
        "float64",
        "float16",
        "uint8",
        "int8",
        "int16",
        "int32",
        "int64",
    ]:
        if backend == "pytorch":
            dtype = getattr(torch, dtype_)
        elif backend == "numpy":
            dtype = getattr(np, dtype_)

    else:
        raise ValueError(
            "Argument for dtype_ not recognized. Please check docstring for acceptable values"
        )
