import pytest
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config

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
        not _check_soft_dependencies("torch", severity="none") or not torch.cuda.is_available(),
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
