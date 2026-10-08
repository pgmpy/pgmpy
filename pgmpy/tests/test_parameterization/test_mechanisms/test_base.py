import numpy as np
import pandas as pd
import pytest
from skbase._exceptions import NotFittedError
from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

from pgmpy.parameterization import BaseMechanism

Normal = _safe_import("skpro.distributions.Normal")

pytestmark = pytest.mark.skipif(
    not _check_soft_dependencies("skpro", severity="none"), reason="execute only if required dependency present"
)


class SumMechanism(BaseMechanism):
    """Y = the sum of the parents + U, with U standard normal."""

    _tags = {"variable_type": ["continuous"], "python_dependencies": "skpro"}

    def _fit(self, X, y, sample_weight):
        pass

    def _predict_proba(self, X):
        means = X.sum(axis=1).to_numpy(dtype=float).reshape(-1, 1)
        return Normal(mu=means, sigma=1.0, index=X.index, columns=[self.variable_])

    def _sample_noise(self, n_samples, random_state):
        return np.random.default_rng(random_state).normal(size=n_samples)

    def _forward(self, X, noise):
        return pd.DataFrame({self.variable_: X.sum(axis=1).to_numpy(dtype=float) + noise}, index=X.index)

    def _abduct(self, X, y):
        return pd.DataFrame({self.variable_: y - X.sum(axis=1).to_numpy(dtype=float)}, index=X.index)


@pytest.fixture
def data():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=5), "b": rng.normal(size=5)}, index=list("pqrst"))
    return X, (X.sum(axis=1) + rng.normal(size=5)).rename("y")


class TestBaseMechanism:
    def test_structural_methods(self, data):
        X, y = data
        mechanism = SumMechanism().fit(X, y)

        # The forward pass on the abducted noise gives the observations back, and with other parent values, their
        # counterfactuals. Without noise, predict is the usual point prediction.
        noise = mechanism.abduct(X, y)
        assert (noise.index.tolist(), noise.columns.tolist()) == (X.index.tolist(), ["y"])
        pd.testing.assert_frame_equal(mechanism.predict(X, noise=noise), y.to_frame())
        np.testing.assert_allclose(mechanism.predict(X.assign(a=X["a"] + 1), noise=noise)["y"], y + 1)
        np.testing.assert_allclose(mechanism.predict(X)["y"], X.sum(axis=1))

        # The noise doesn't depend on the parents, so sample_noise only takes how many values to draw; an array passes
        # them on by position. sample draws the noise and runs the forward pass, so a seed gives the same values.
        u = mechanism.sample_noise(len(X), random_state=0)
        assert u.shape == (len(X),)
        pd.testing.assert_frame_equal(mechanism.sample(X, random_state=0), mechanism.predict(X, noise=u))
        drawn = mechanism.sample(X, n_samples=3, random_state=0)
        expected = np.tile(X.sum(axis=1), 3) + mechanism.sample_noise(3 * len(X), random_state=0)
        np.testing.assert_allclose(drawn["y"], expected)
        assert drawn.index.tolist()[:2] == [(0, "p"), (0, "q")]
        root = SumMechanism().fit(None, y)
        np.testing.assert_allclose(root.sample(n_samples=4, random_state=0)["y"], root.sample_noise(4, random_state=0))

    def test_checks(self, data):
        X, y = data
        with pytest.raises(NotFittedError):
            SumMechanism().sample_noise(3)
        mechanism = SumMechanism().fit(X, y)

        # Noise and observations need one finite value per row: arrays by position, pandas objects with X's index.
        for bad in (np.zeros(len(X) + 1), np.full(len(X), np.nan), y.set_axis(range(len(y))), y.to_frame().assign(z=1)):
            with pytest.raises(ValueError):
                mechanism.predict(X, noise=bad)
            with pytest.raises(ValueError):
                mechanism.abduct(X, bad)
        with pytest.raises(ValueError, match="parents"):
            mechanism.predict(None, noise=np.zeros(len(X)))
        # A root takes a DataFrame without columns for the forward pass and abduction, as for predict.
        root = SumMechanism().fit(None, y)
        with pytest.raises(ValueError, match="predict needs X"):
            root.predict(None, noise=np.zeros(len(X)))
        with pytest.raises(ValueError, match="abduct needs X"):
            root.abduct(None, y)
        for n_samples in (-1, 2.5, True, None):
            with pytest.raises(ValueError, match="n_samples"):
                mechanism.sample_noise(n_samples)
