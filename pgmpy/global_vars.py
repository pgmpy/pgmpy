import numpy as np
import torch


class Config:
    def __init__(self):
        """
        Default configuration initilization
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
        return self.DEVICE

    def set_backend(self, backend, device=None):
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

    def get_backend(self):
        return self.BACKEND

    def set_show_progress(self, show_progress):
        if show_progress not in [True, False]:
            raise ValueError(f"show_progress must be a boolean. Got: {show_progress}")

        self.SHOW_PROGRESS = show_progress

    def get_show_progress(self):
        return self.SHOW_PROGRESS

    def set_dtype(self, dtype):
        if self.BACKEND == "numpy":
            if not isinstance(dtype, np.dtype):
                raise ValueError(
                    f"Backend is set to numpy, dtype must be an instance of np.dtype. Got: {type(dtype)}"
                )

            self.DTYPE = dtype

        elif self.BACKEND == "torch":
            if not isinstance(dtype, torch.dtype):
                raise ValueError(
                    f"Backend is set to torch, dtype must be an instance of torch.dtype. Got: {type(dtype)}"
                )

            self.DTYPE = dtype

    def get_dtype(self):
        return self.DTYPE


config = Config()
