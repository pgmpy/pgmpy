import logging

import pytest
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config
from pgmpy.global_vars import DuplicateFilter

torch = _safe_import("torch")


@pytest.fixture(autouse=True)
def reset_config():
    """Reset pgmpy config to defaults after each test."""
    yield
    config.set_backend("numpy")
    config.set_show_progress(show_progress=True)


class TestConfig:
    def test_defaults(self):
        assert config.BACKEND == "numpy"
        assert config.get_backend() == "numpy"

        assert config.DTYPE == "float64"
        assert config.get_dtype() == "float64"

        assert config.DEVICE is None
        assert config.get_device() is None

        assert config.SHOW_PROGRESS is True
        assert config.get_show_progress() is True

    @pytest.mark.skipif(
        not _check_soft_dependencies("torch", severity="none"),
        reason="test only if torch is available",
    )
    def test_torch_cpu(self):
        config.set_backend(backend="torch", device="cpu", dtype=torch.float32)

        assert config.BACKEND == "torch"
        assert config.get_backend() == "torch"

        assert config.DTYPE == torch.float32
        assert config.get_dtype() == torch.float32

        assert config.DEVICE == torch.device("cpu")
        assert config.get_device() == torch.device("cpu")

        assert config.SHOW_PROGRESS is True
        assert config.get_show_progress() is True

    @pytest.mark.skipif(
        not _check_soft_dependencies("torch", severity="none")
        or not torch.cuda.is_available(),
        reason="test only if torch and torch.cuda are available",
    )
    def test_torch_gpu(self):  # pragma: no cover
        config.set_backend(backend="torch", device="cuda", dtype=torch.float32)

        assert config.BACKEND == "torch"
        assert config.get_backend() == "torch"

        assert config.DTYPE == torch.float32
        assert config.get_dtype() == torch.float32

        assert config.DEVICE == torch.device("cuda")
        assert config.get_device() == torch.device("cuda")

        assert config.SHOW_PROGRESS is True
        assert config.get_show_progress() is True

    def test_no_progress(self):
        config.set_show_progress(show_progress=False)

        assert config.BACKEND == "numpy"
        assert config.get_backend() == "numpy"

        assert config.DTYPE == "float64"
        assert config.get_dtype() == "float64"

        assert config.DEVICE is None
        assert config.get_device() is None

        assert config.SHOW_PROGRESS is False
        assert config.get_show_progress() is False


class TestDuplicateFilter:
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

        assert captured_logs == expected_logs
