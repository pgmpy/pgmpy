# Global variable for setting whether to show progress bars or not
SHOW_PROGRESS = True

def no_progress():
    """
    If called sets the global variable `SHOW_PROGRESS` to False resulting in no
    progress bars anywhere.
    """
    global SHOW_PROGRESS
    SHOW_PROGRESS = False

def get_progress():
    """
    Returns the current state of the SHOW_PROGRESS variable.
    """
    global SHOW_PROGRESS
    return(SHOW_PROGRESS)

# Check if GPU is available.
import torch

CUDA = torch.cuda.is_available()

# Check if GPU is available.
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

if CUDA:
    DTYPE = torch.float64
else:
    DTYPE = "float64"

BACKEND = 'numpy'

def get_backend():
    global BACKEND
    return(BACKEND)

def set_backend(backend):
    global BACKEND
    BACKEND = backend

def set_device(device):
    """
    Sets the device for pytorch.
    """
    global DEVICE
    torch.device(device)

def get_device():
    """
    Returns the current device.
    """
    global DEVICE
    return(DEVICE)


def set_dtype(dtype):
    """
    Sets the dtype globally.
    """
    global DTYPE
    DTYPE = dtype


def get_dtype():
    """
    Returns the current global dtype.
    """
    global DTYPE
    return(DTYPE)
