import logging

import numpy as np
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pgmpy")


class Config:
    def __init__(self):
        """
        Default configuration initilization.
        """
        self.BACKEND = "numpy"
        self.DTYPE = "float64"
        self.DEVICE = None
        self.SHOW_PROGRESS = True

    def set_device(self, device=None):
        """
        Sets the device if using pytorch backend.

        Parameters
        ----------
        device: str (default: None)
            Either 'cuda': to create arrays on GPU, or 'cpu' to create arrays on CPU.
            If None, sets to cuda if GPU is available else uses CPU.
        """
        if self.BACKEND == "numpy":
            raise ValueError(
                f"Current backend is numpy. Device can only be set for torch backend"
            )

        if device is None:
            if torch.cuda.is_available():
                self.DEVICE = torch.device("cuda:0")
            else:
                self.DEVICE = torch.device("cpu")
        else:
            if not device.startswith(("cuda", "cpu")):
                raise ValueError(
                    f"device must be either 'cuda', 'cuda:x' or 'cpu'. Got: {device}"
                )
            elif device.startswith("cuda"):
                if torch.cuda.is_available():
                    self.DEVICE = torch.device(device)
            else:
                self.DEVICE = torch.device(device)

    def get_device(self):
        """
        Returns the current backend device.
        """
        return self.DEVICE

    def set_backend(self, backend, device=None, dtype=None):
        """
        Setup the compute backend.

        Parameters
        ----------
        backend: str (numpy or torch)
            Sets the compute backend to `backend`.

        device: str (default: None)
            Sets the device for torch backend. For numpy backend, sets device=None.
            If None, sets device to the first gpu if available else cpu.

        dtype: Instance of numpy.dtype or torch.dtype (default: None)
            Sets the dtype for arrays. If None, sets to either numpy.float64 or
            torch.float64 depending on the backend.
        """
        if backend not in ["numpy", "torch"]:
            raise ValueError(
                f"backend can either be `numpy` or `torch`. Got: {backend}"
            )

        if backend == "numpy":
            self.BACKEND = "numpy"
            self.DEVICE = None
        else:
            self.BACKEND = "torch"
            self.set_device(device)
        self.set_dtype(dtype=dtype)

    def get_backend(self):
        """
        Returns the current backend.
        """
        return self.BACKEND

    def set_show_progress(self, show_progress):
        """
        Sets a global variable to (not) show progress bars.

        Parameters
        ----------
        show_progress: boolean
            If True, shows progress bars, else doesn't.
        """
        if show_progress not in [True, False]:
            raise ValueError(f"show_progress must be a boolean. Got: {show_progress}")

        self.SHOW_PROGRESS = show_progress

    def get_show_progress(self):
        """
        Returns boolean whether to show progress bar or not.
        """
        return self.SHOW_PROGRESS

    def set_dtype(self, dtype=None):
        """
        Sets the dtype for value matrices.

        Parameters
        ----------
        dtype: Instance of numpy.dtype of torch.dtype. (default: None)
            Sets the dtype to `dtype`. If None set to either numpy.float64 or torch.float64 depending on the backend.
        """
        if self.BACKEND == "numpy":
            if dtype is None:
                self.DTYPE = "float64"
            elif not isinstance(dtype, np.dtype):
                raise ValueError(
                    f"Backend is set to numpy, dtype must be an instance of np.dtype. Got: {type(dtype)}"
                )
            else:
                self.DTYPE = dtype

        elif self.BACKEND == "torch":
            if dtype is None:
                self.DTYPE = torch.float64
            elif not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"Backend is set to torch, dtype must be an instance of torch.dtype. Got: {type(dtype)}"
                )
            else:
                self.DTYPE = dtype

    def get_dtype(self):
        """
        Returns the dtype.
        """
        return self.DTYPE

    def get_compute_backend(self):
        if self.BACKEND == "numpy":
            return np

        else:
            return torch


config = Config()
