import importlib

import numpy as np
import pandas as pd
import pytest
from sklearn.base import BaseEstimator, clone

from pgmpy.causal_discovery import PNL
from pgmpy.causal_discovery.PNL import NormFlow


class PolynomialDisturbanceEstimator(BaseEstimator):
    """Small deterministic estimator used to test PNL orchestration."""

    def fit(self, X, y=None):
        X = np.asarray(X)
        self.function_ = np.polynomial.Polynomial.fit(X[:, 0], X[:, 1], deg=3)
        return self

    def predict(self, X):
        X = np.asarray(X)
        return X[:, 1] - self.function_(X[:, 0])


class NonFiniteDisturbanceEstimator(BaseEstimator):
    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.full(len(X), np.nan)


class SeededPolynomialDisturbanceEstimator(PolynomialDisturbanceEstimator):
    def __init__(self, seed=None):
        self.seed = seed


def _squared_dependence_score(cause, disturbance):
    return abs(np.corrcoef(np.asarray(cause) ** 2, np.asarray(disturbance) ** 2)[0, 1])


@pytest.fixture
def nonlinear_data():
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, 500)
    y = x**3 + 0.2 * rng.normal(size=x.size)
    return pd.DataFrame({"X": x, "Y": y})


def test_init():
    assert PNL().get_params() == {
        "estimator": None,
        "scoring_method": "independence",
        "seed": None,
    }


def test_fit_recovers_direction(nonlinear_data):
    est = PNL(
        estimator=PolynomialDisturbanceEstimator(),
        scoring_method=_squared_dependence_score,
    ).fit(nonlinear_data)

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ < est.backward_score_
    assert est.adjacency_matrix_.loc["X", "Y"] == 1
    assert est.score(true_graph=est.causal_graph_, metric="SHD") == 0


def test_non_finite_disturbance_raises():
    data = pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 4.0]})

    with pytest.raises(ValueError, match="non-finite disturbance"):
        PNL(estimator=NonFiniteDisturbanceEstimator()).fit(data)


@pytest.mark.parametrize(
    ("score", "match"),
    [
        (np.nan, "dependence score is non-finite"),
        (0.0, "both directions produced the same score"),
    ],
)
def test_non_finite_or_tied_direction_scores_raise(nonlinear_data, score, match):
    with pytest.raises(ValueError, match=match):
        PNL(
            estimator=PolynomialDisturbanceEstimator(),
            scoring_method=lambda cause, disturbance: score,
        ).fit(nonlinear_data)


def test_clone_preserves_scoring_method_instance():
    from pgmpy.causal_discovery.bivariate_scores import IndependenceScore

    cloned = clone(PNL(scoring_method=IndependenceScore(criterion="p_value")))

    assert isinstance(cloned.scoring_method, IndependenceScore)
    assert cloned.scoring_method.criterion == "p_value"


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (pd.DataFrame({"X": [1.0, 1.0, 1.0], "Y": [1.0, 2.0, 3.0]}), "constant"),
        (pd.DataFrame({"X": [0.0, 1.0, 2.0], "Y": [1.0, 2.0, 3.0], "Z": [2.0, 1.0, 0.0]}), "exactly two variables"),
        (pd.DataFrame({"X": [0.0, 1.0, np.nan], "Y": [1.0, 2.0, 3.0]}), None),
        (pd.DataFrame({"X": list("aabbab"), "Y": list("xyxyxy")}), "continuous"),
    ],
)
def test_invalid_input_raises(data, match):
    with pytest.raises(ValueError, match=match):
        PNL(estimator=PolynomialDisturbanceEstimator()).fit(data)


def test_seed_is_passed_to_default_estimators(monkeypatch, nonlinear_data):
    pnl_module = importlib.import_module("pgmpy.causal_discovery.PNL")
    monkeypatch.setattr(pnl_module, "NormFlow", SeededPolynomialDisturbanceEstimator)

    est = PNL(seed=42, scoring_method=_squared_dependence_score).fit(nonlinear_data)

    assert est.forward_estimator_.seed == 42
    assert est.backward_estimator_.seed == 42


def test_normflow_module_fit_and_serialization():
    torch = pytest.importorskip("torch")
    x = np.linspace(-1.0, 1.0, 32)
    data = np.column_stack((x, x**3))[:, ::-1]
    data_t = torch.tensor(np.ascontiguousarray(data), dtype=torch.float32)

    model = NormFlow(hidden_dim=4, n_components=3, max_iter=3, seed=7)
    assert isinstance(model, torch.nn.Module)
    assert {"inner_model_.0.weight", "post_bias_", "mean_", "scale_", "fitted_"} <= model.state_dict().keys()
    assert not model.fitted_.item()

    torch.manual_seed(123)
    rng_state = torch.random.get_rng_state().clone()
    first = model.fit(data)
    second = NormFlow(hidden_dim=4, n_components=3, max_iter=3, seed=7).fit(data)
    restored = NormFlow(hidden_dim=4, n_components=3, max_iter=3, seed=8)
    restored.load_state_dict(first.state_dict())

    assert torch.equal(torch.random.get_rng_state(), rng_state)
    assert first(data_t).shape == (len(data), 1)
    assert first.predict(data).shape == (len(data),)
    assert float(first.post_bias_.detach().std()) > 0
    assert torch.all(first._post_transform(torch.tensor([[0.0]]))[1] > 0)
    np.testing.assert_allclose(first.predict(data), second.predict(data))
    torch.testing.assert_close(restored(data_t), first(data_t))


class PostNonlinearDisturbanceEstimator(BaseEstimator):
    """Estimator that inverts post-nonlinearity g(Y) before extracting additive disturbance."""

    def fit(self, X, y=None):
        X = np.asarray(X)
        y_inv = X[:, 1] ** 3
        self.function_ = np.polynomial.Polynomial.fit(X[:, 0], y_inv, deg=3)
        return self

    def predict(self, X):
        X = np.asarray(X)
        y_inv = X[:, 1] ** 3
        return y_inv - self.function_(X[:, 0])


def test_post_nonlinear_data():
    """Verify PNL direction recovery when explicit post-nonlinearity g(z) != z is applied."""
    rng = np.random.default_rng(0)
    x = rng.uniform(-2.0, 2.0, 500)
    z = x**3 + 0.2 * rng.normal(size=x.size)
    y = np.cbrt(z)  # Non-trivial post-nonlinear transformation g(z) = z^(1/3)
    data = pd.DataFrame({"X": x, "Y": y})

    est = PNL(
        estimator=PostNonlinearDisturbanceEstimator(),
        scoring_method=_squared_dependence_score,
    ).fit(data)

    assert list(est.causal_graph_.edges()) == [("X", "Y")]
    assert est.forward_score_ < est.backward_score_
