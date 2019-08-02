import torch


# Check if pandas is available
try:
    import pandas

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Check if GPU is available
def set_device(device_='cpu'):
    """
    Set the device to be used for pytorch.

    Parameters
    ----------
    device_: str (cpu | gpu)
        Device to use for computation.
    """
    global device
    if device_ == 'cpu':
        device = torch.device("cpu")
    elif device == 'gpu':
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            raise ValueError("No cuda supported GPU available")
    else:
        raise ValueError("Unknown device. Please select between cpu or gpu")

# Set backend to use numpy or pytorch
def set_backend(backend_='numpy'):
    """
    Sets the backend framework to use.

    Parameters
    ----------
    backend: str (numpy | pytorch)
        The backend to use for computation.
    """
    global backend

    if backend_ in ['numpy', 'pytorch']:
        backend = backend_
    else:
        raise ValueError("Unknown backend. Please select between numpy or pytorch")

def set_dtype(dtype_='float'):
    """
    Globally set dtype for all arrays.

    Parameters
    ----------
    dtype: str
    """
    global dtype

    if dtype_ == 'float':
        dtype = torch.float
