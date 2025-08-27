import logging
import unittest

import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from pgmpy.utils._safe_import import _safe_import

torch = _safe_import("torch")

from pgmpy import config
from pgmpy.global_vars import DuplicateFilter


class TestConfig:
    def assertEqual(self, x, y):
        assert x == y

    def test_defaults(self):
        self.assertEqual(config.BACKEND, "numpy")
        self.assertEqual(config.get_backend(), "numpy")

        self.assertEqual(config.DTYPE, "float64")
        self.assertEqual(config.get_dtype(), "float64")

        self.assertEqual(config.DEVICE, None)
        self.assertEqual(config.get_device(), None)

        self.assertEqual(config.SHOW_PROGRESS, True)
        self.assertEqual(config.get_show_progress(), True)

    @pytest.mark.skipif(
        not _check_soft_dependencies("torch", severity="none"),
        reason="test only if torch is available",
    )
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

    @pytest.mark.skipif(
        not _check_soft_dependencies("torch", severity="none")
        or not torch.cuda.is_available(),
        reason="test only if torch and torch.cuda are available",
    )
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

    @pytest.mark.skipif(
        not _check_soft_dependencies("torch", severity="none")
        or not torch.cuda.is_available(),
        reason="test only if torch and torch.cuda are available",
    )
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


class TestDuplicateFilter(unittest.TestCase):
    def test_duplicate_filter(self):
        test_logger = logging.getLogger("test_logger")
        test_logger.setLevel(logging.INFO)

        for handler in test_logger.handlers[:]:
            test_logger.removeHandler(handler)
        for filter_ in test_logger.filters[:]:
            test_logger.removeFilter(filter_)

        captured_logs = []

        class ListHandler(logging.Handler):
            def emit(self, record):
                captured_logs.append(record.getMessage())

        handler = ListHandler()
        test_logger.addHandler(handler)

        # Add duplicate filter
        duplicate_filter = DuplicateFilter()
        test_logger.addFilter(duplicate_filter)

        test_logger.info("First message")  # Should pass
        test_logger.info("First message")  # Should be filtered out (duplicate)
        test_logger.info("Second message")  # Should pass
        test_logger.info("Second message")  # Should be filtered out (duplicate)
        test_logger.info("Second message")  # Should be filtered out (duplicate)
        test_logger.info("First message")  # Should pass (not consecutive duplicate)
        test_logger.info("First message")  # Should be filtered out (duplicate)
        test_logger.info("Third message")  # Should pass

        expected_logs = [
            "First message",
            "Second message",
            "First message",
            "Third message",
        ]

        self.assertEqual(captured_logs, expected_logs)
