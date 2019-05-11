# TODO: This variables being set in this file should move to setup.py

import torch


# This module initializes flags for optional dependencies
try:
    import pandas

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

# Check if GPU is available
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

dtype = torch.float
