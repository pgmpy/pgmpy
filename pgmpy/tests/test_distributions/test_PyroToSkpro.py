import math

import numpy as np
import pandas as pd
import pytest
from skpro.distributions.laplace import Laplace
from skpro.distributions.normal import Normal

from pgmpy.distributions.converter.PyroToSkpro import PyroToSkpro

torch = pytest.importorskip("torch")
pyro = pytest.importorskip("pyro")


@pytest.fixture
def samples():
    samples = {
        "obs": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [5.0, 6.0],
            ]
        )
    }
    return samples


class TestPyroToSkpro:
    def test_normal(self, samples):

        converter = PyroToSkpro("normal")
        normal = converter.convert(samples, pd.RangeIndex(0, 2), ["normal"], "obs")

        assert isinstance(normal, Normal)
        assert normal.shape == (2, 1)
        np.testing.assert_allclose(
            normal.mu,
            np.array(
                [
                    [3.0],
                    [4.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            normal.sigma,
            np.array(
                [
                    [1.6329932],
                    [1.6329932],
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_add(self, samples):
        def _convert_laplace(samples, index, columns, name="obs"):
            y_samples = samples[name]
            mean = y_samples.mean(dim=0).detach().reshape(-1, 1).cpu().numpy()
            scale = (y_samples.std(dim=0, unbiased=False) / math.sqrt(2)).detach().reshape(-1, 1).cpu().numpy()

            return Laplace(
                mu=mean,
                scale=scale,
                index=index,
                columns=columns,
            )

        converter = PyroToSkpro("laplace").add("laplace", _convert_laplace)
        laplace = converter.convert(samples, pd.RangeIndex(0, 2), ["laplace"], "obs")

        assert isinstance(laplace, Laplace)
        assert laplace.shape == (2, 1)
        np.testing.assert_allclose(
            laplace.mu,
            np.array(
                [
                    [3.0],
                    [4.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            laplace.scale,
            np.array(
                [
                    [1.1547006],
                    [1.1547006],
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_set(self, samples):
        def _convert_laplace(samples, index, columns, name="obs"):
            y_samples = samples[name]
            mean = y_samples.mean(dim=0).detach().reshape(-1, 1).cpu().numpy()
            scale = (y_samples.std(dim=0, unbiased=False) / math.sqrt(2)).detach().reshape(-1, 1).cpu().numpy()

            return Laplace(
                mu=mean,
                scale=scale,
                index=index,
                columns=columns,
            )

        converter = PyroToSkpro()
        converter.add("laplace", _convert_laplace)
        converter.set("laplace")
        laplace = converter.convert(samples, pd.RangeIndex(0, 2), ["laplace"], "obs")

        assert isinstance(laplace, Laplace)
        assert laplace.shape == (2, 1)
        np.testing.assert_allclose(
            laplace.mu,
            np.array(
                [
                    [3.0],
                    [4.0],
                ],
                dtype=np.float32,
            ),
        )
        np.testing.assert_allclose(
            laplace.scale,
            np.array(
                [
                    [1.1547006],
                    [1.1547006],
                ],
                dtype=np.float32,
            ),
            rtol=1e-6,
            atol=1e-7,
        )

    def test_remove(self, samples):
        def _convert_laplace(samples, index, columns, name="obs"):
            return ...

        converter = PyroToSkpro()
        converter.add("laplace", _convert_laplace)
        converter.set("laplace")
        converter.remove("laplace")
        with pytest.raises(ValueError):
            converter.convert(samples, pd.RangeIndex(0, 2), ["laplace"], "obs")

    def test_get(self, samples):
        def _convert_laplace(samples, index, columns, name="obs"):
            return ...

        converter = PyroToSkpro()

        assert converter.get() == ["normal"]

        converter.add("laplace", _convert_laplace)
        assert converter.get() == ["normal", "laplace"]
