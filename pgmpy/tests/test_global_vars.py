import unittest

import numpy as np
import torch

from pgmpy import config


class TestConfig(unittest.TestCase):
    def test_defaults(self):
        self.assertEqual(config.BACKEND, "numpy")
        self.assertEqual(config.get_backend(), "numpy")

        self.assertEqual(config.DTYPE, "float64")
        self.assertEqual(config.get_dtype(), "float64")

        self.assertEqual(config.DEVICE, None)
        self.assertEqual(config.get_device(), None)

        self.assertEqual(config.SHOW_PROGRESS, True)
        self.assertEqual(config.get_show_progress(), True)

    def test_torch_cpu(self):
        config.set_backend(backend="torch", device="cpu", dtype=torch.float32)

        self.assertEqual(config.BACKEND, "torch")
        self.assertEqual(config.get_backend(), "torch")

        self.assertEqual(config.DTYPE, torch.float32)
        self.assertEqual(config.get_dtype(), torch.float32)

        self.assertEqual(config.DEVICE, torch.device("cpu"))
        self.assertEqual(config.get_device(), torch.device("cpu"))

        self.assertEqual(config.SHOW_PROGRESS, True)
        self.assertEqual(config.get_show_progress(), True)

    @unittest.skipIf(not torch.cuda.is_available(), "No GPU")
    def test_torch_gpu(self):
        config.set_backend(backend="torch", device="cuda", dtype=torch.float32)

        self.assertEqual(config.BACKEND, "torch")
        self.assertEqual(config.get_backend(), "torch")

        self.assertEqual(config.DTYPE, torch.float32)
        self.assertEqual(config.get_dtype(), torch.float32)

        self.assertEqual(config.DEVICE, torch.device("cuda"))
        self.assertEqual(config.get_device(), torch.device("cuda"))

        self.assertEqual(config.SHOW_PROGRESS, True)
        self.assertEqual(config.get_show_progress(), True)

    def test_no_progress(self):
        config.set_show_progress(show_progress=False)

        self.assertEqual(config.BACKEND, "numpy")
        self.assertEqual(config.get_backend(), "numpy")

        self.assertEqual(config.DTYPE, "float64")
        self.assertEqual(config.get_dtype(), "float64")

        self.assertEqual(config.DEVICE, None)
        self.assertEqual(config.get_device(), None)

        self.assertEqual(config.SHOW_PROGRESS, False)
        self.assertEqual(config.get_show_progress(), False)

    def tearDown(self):
        config.set_backend("numpy")
        config.set_show_progress(show_progress=True)
