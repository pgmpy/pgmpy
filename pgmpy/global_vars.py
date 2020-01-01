# TODO: This variables being set in this file should move to setup.py


try:  # pragma: no cover
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
try:  # pragma: no cover
    import pandas

    HAS_PANDAS = True
except ImportError:  # pragma: no cover
    HAS_PANDAS = False
    pandas = None
