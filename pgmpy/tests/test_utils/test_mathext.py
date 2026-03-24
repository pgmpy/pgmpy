import numpy as np
import numpy.testing as np_test
import pytest
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy import config
from pgmpy.utils.mathext import sample_discrete, sample_discrete_maps

torch = _safe_import("torch")


def _raise_if_called(*args, **kwargs):
    raise AssertionError("numpy.random.choice should not be used in torch backend sampling")


@pytest.mark.skipif(
    not _check_soft_dependencies("torch", severity="none"),
    reason="test only if torch is available",
)
class TestTorchNativeSampling:
    @pytest.fixture(autouse=True)
    def set_torch_backend(self):
        prev_backend = config.get_backend()
        prev_device = config.get_device()
        prev_dtype = config.get_dtype()

        config.set_backend("torch")
        yield
        config.set_backend(prev_backend, device=str(prev_device) if prev_device is not None else None, dtype=prev_dtype)

    def test_sample_discrete_does_not_use_numpy_choice(self, monkeypatch):
        monkeypatch.setattr(np.random, "choice", _raise_if_called)

        weights = torch.tensor([0.2, 0.5, 0.3], dtype=config.get_dtype(), device=config.get_device())

        samples = sample_discrete(np.array([0, 1, 2]), weights, size=16, seed=7)
        repeat_samples = sample_discrete(np.array([0, 1, 2]), weights, size=16, seed=7)

        np_test.assert_array_equal(samples, repeat_samples)
        assert set(samples.tolist()).issubset({0, 1, 2})

    def test_sample_discrete_maps_does_not_use_numpy_choice(self, monkeypatch):
        monkeypatch.setattr(np.random, "choice", _raise_if_called)

        weight_indices = torch.tensor([0, 1, 0, 1, 1, 0], dtype=torch.long, device=config.get_device())
        index_to_weight = {
            0: torch.tensor([0.2, 0.8], dtype=config.get_dtype(), device=config.get_device()),
            1: torch.tensor([0.7, 0.3], dtype=config.get_dtype(), device=config.get_device()),
        }

        samples = sample_discrete_maps(np.array([0, 1]), weight_indices, index_to_weight, size=6, seed=11)
        repeat_samples = sample_discrete_maps(np.array([0, 1]), weight_indices, index_to_weight, size=6, seed=11)

        np_test.assert_array_equal(samples, repeat_samples)
        assert set(samples.tolist()).issubset({0, 1})
