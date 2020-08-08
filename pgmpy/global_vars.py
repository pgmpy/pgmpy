# TODO: This variables being set in this file should move to setup.py


try:
    import torch

    # Check if GPU is available
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    dtype = torch.float
except ImportError:  # pragma: no cover
    torch = None
    device = None
    dtype = None


# This module initializes flags for optional dependencies
try:
    import pandas

    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False
    pandas = None


# Set a global variable whether to show progress bar or not.
SHOW_PROGRESS = True


def no_progress():
    """
    If called sets the global variable `SHOW_PROGRESS` to False resulting in no
    progress bars anywhere.
    """
    global SHOW_PROGRESS
    SHOW_PROGRESS = False
