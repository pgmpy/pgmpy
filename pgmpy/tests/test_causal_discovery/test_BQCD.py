import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from pgmpy.causal_discovery import BQCD


class EmpiricalQuantileRegressor:
    """Minimal regressor used to exercise the custom quantile-regressor API."""

    def __init__(self, quantile):
        self.quantile = quantile

    def fit(self, X, y):
        self.prediction_ = float(np.quantile(y, self.quantile))
        return self

    def predict(self, X):
        x = np.asarray(X, dtype=float).reshape(-1)
        return np.full(x.shape, self.prediction_) + 1e-3 * x


class NonFiniteQuantileRegressor:
    def fit(self, X, y):
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)


def _nonlinear_data(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    y = x + (0.05 + 2.0 * x**2) * rng.normal(size=n)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    assert BQCD().get_params() == {
        "n_quantiles": 3,
        "quantile_regressor": None,
        "seed": None,
    }


def test_fit_recovers_direction():
    est = BQCD(n_quantiles=3, seed=0).fit(_nonlinear_data(n=200, seed=0))

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ == pytest.approx(0.7952320308862181, rel=1e-4)
    assert est.backward_score_ == pytest.approx(0.7981598437714771, rel=1e-4)
    assert est.forward_score_ < est.backward_score_
    assert len(est.quantile_levels_) == 3
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.score(true_graph=est.causal_graph_, metric="SHD") == 0


def test_fit_recovers_backward_direction():
    data = _nonlinear_data()[["Y", "X"]]
    est = BQCD(n_quantiles=3, seed=0).fit(data)

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.backward_score_ < est.forward_score_


def test_small_sample_uses_single_quantile():
    est = BQCD(n_quantiles=7, seed=0).fit(_nonlinear_data(n=199))

    assert est.quantile_levels_ == pytest.approx([0.5])
    assert est.quadrature_weights_ == pytest.approx([1.0])
    assert len(est.forward_estimators_) == 1


def test_custom_quantile_regressor():
    calls = []

    def quantile_regressor(quantile):
        calls.append(float(quantile))
        return EmpiricalQuantileRegressor(quantile)

    est = BQCD(n_quantiles=2, quantile_regressor=quantile_regressor).fit(_nonlinear_data())

    assert len(calls) == 4
    assert all(isinstance(regressor, EmpiricalQuantileRegressor) for regressor in est.forward_estimators_)


def test_fixed_seed_is_reproducible():
    data = _nonlinear_data(seed=2)
    first = BQCD(n_quantiles=3, seed=7).fit(data)
    second = BQCD(n_quantiles=3, seed=7).fit(data)

    assert first.forward_score_ == second.forward_score_
    assert first.backward_score_ == second.backward_score_


def test_clone_preserves_n_quantiles():
    cloned = clone(BQCD(n_quantiles=2, seed=3))

    assert cloned.get_params() == {"n_quantiles": 2, "quantile_regressor": None, "seed": 3}


@pytest.mark.parametrize(
    ("kwargs", "data", "match"),
    [
        ({"n_quantiles": 0}, pd.DataFrame({"X": [0.0, 1.0], "Y": [1.0, 2.0]}), "positive integer"),
        ({"n_quantiles": True}, pd.DataFrame({"X": [0.0, 1.0], "Y": [1.0, 2.0]}), "positive integer"),
        (
            {"quantile_regressor": "not-callable"},
            pd.DataFrame({"X": [0.0, 1.0], "Y": [1.0, 2.0]}),
            "None or callable",
        ),
        ({}, pd.DataFrame({"X": [1.0, 1.0, 1.0], "Y": [1.0, 2.0, 3.0]}), "constant"),
        ({}, pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 3.0], "Z": [2.0, 1.0, 0.0]}), "exactly two variables"),
        ({}, pd.DataFrame({"X": list("aabbab"), "Y": list("xyxyxy")}), "continuous"),
        (
            {"quantile_regressor": lambda quantile: NonFiniteQuantileRegressor()},
            _nonlinear_data(),
            "non-finite predictions",
        ),
        (
            {"quantile_regressor": lambda quantile: EmpiricalQuantileRegressor(quantile)},
            pd.DataFrame({"X": [0.0, 1.0, 2.0, 3.0], "Y": [0.0, 1.0, 2.0, 3.0]}),
            "both directions produced the same score",
        ),
    ],
)
def test_invalid_input_raises(kwargs, data, match):
    with pytest.raises(ValueError, match=match):
        BQCD(**kwargs).fit(data)


def test_numpy_integer_n_quantiles():
    est = BQCD(n_quantiles=np.int64(2), seed=0).fit(_nonlinear_data())

    assert len(est.quantile_levels_) == 2


def test_homoscedastic_data():
    """BQCD relies on heteroscedastic noise; homoscedastic noise produces near-zero or tied score differences."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=200)
    y = x + rng.normal(size=200)  # Homoscedastic noise: constant noise variance
    data = pd.DataFrame({"X": x, "Y": y})

    est = BQCD(n_quantiles=3, seed=0).fit(data)
    assert hasattr(est, "forward_score_") and hasattr(est, "backward_score_")
