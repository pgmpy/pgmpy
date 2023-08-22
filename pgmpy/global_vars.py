# TODO: This variables being set in this file should move to setup.py
import torch

CUDA = torch.cuda.is_available()

# Check if GPU is available
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# Set a global variable whether to show progress bar or not.
SHOW_PROGRESS = True
if CUDA:
    DTYPE = torch.float64
else:
    DTYPE = "float64"


def no_progress():
    """
    If called sets the global variable `SHOW_PROGRESS` to False resulting in no
    progress bars anywhere.
    """
    global SHOW_PROGRESS
    SHOW_PROGRESS = False


def set_dtype(dtype):
    """
    Sets the dtype globally.
    """
    global DTYPE
    DTYPE = dtype
