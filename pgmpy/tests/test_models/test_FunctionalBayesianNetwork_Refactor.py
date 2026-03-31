import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import MaximumLikelihoodEstimator as MLE
from pgmpy.factors.hybrid.Adapters import LinearGaussianAdapter, TabularAdapter
from pgmpy.factors.hybrid.FunctionalCPD_Refactor import FunctionalCPD
from pgmpy.factors.hybrid.SkproAdapter import SkproAdapter
from pgmpy.models.FunctionalBayesianNetwork_Refactor import FunctionalBayesianNetwork as FunctionalBN


def test_fit_learns_mixed_tabular_and_linear_cpds():
    rng = np.random.default_rng(42)
    n_samples = 1000

    a_data = rng.integers(0, 2, size=n_samples)
    b_data = rng.integers(0, 2, size=n_samples)
    c_data = rng.normal(loc=5.0, scale=2.0, size=n_samples)
    d_data = 2.5 * a_data - 1.5 * b_data + 3.0 * c_data + rng.normal(loc=0.0, scale=1.0, size=n_samples)

    data = pd.DataFrame({"A": a_data, "B": b_data, "C": c_data, "D": d_data})

    model = FunctionalBN([("A", "D"), ("B", "D"), ("C", "D")])

    cpd_a = FunctionalCPD(variable="A", tag="tabular", estimator=MLE)
    cpd_b = FunctionalCPD(variable="B", tag="tabular", estimator=MLE)
    cpd_c = FunctionalCPD(variable="C", tag="linear", estimator="MLE")
    cpd_d = FunctionalCPD(variable="D", tag="linear", estimator="MLE")

    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)
    model.fit(data)

    fitted_a = model.get_cpds("A").fitted_cpd_
    fitted_b = model.get_cpds("B").fitted_cpd_
    fitted_c = model.get_cpds("C").fitted_cpd_
    fitted_d = model.get_cpds("D").fitted_cpd_

    np.testing.assert_allclose(fitted_a.get_values().reshape(-1), [0.5, 0.5], atol=0.08)
    np.testing.assert_allclose(fitted_b.get_values().reshape(-1), [0.5, 0.5], atol=0.08)

    np.testing.assert_allclose(fitted_c.beta, [5.0], atol=0.2)
    np.testing.assert_allclose(fitted_c.std, 2.0, atol=0.2)

    np.testing.assert_allclose(fitted_d.beta[0], 0.0, atol=0.25)
    np.testing.assert_allclose(fitted_d.beta[1:], [2.5, -1.5, 3.0], atol=0.2)
    np.testing.assert_allclose(fitted_d.std, 1.0, atol=0.15)


def test_fit_uses_tabular_and_linear_adapters():
    rng = np.random.default_rng(21)
    data = pd.DataFrame(
        {
            "A": rng.integers(0, 2, size=80),
            "B": rng.normal(size=80),
            "C": rng.normal(size=80),
        }
    )

    model = FunctionalBN([("A", "C"), ("B", "C")])
    cpd_a = FunctionalCPD(variable="A", tag="tabular", estimator=MLE)
    cpd_b = FunctionalCPD(variable="B", tag="linear", estimator="MLE")
    cpd_c = FunctionalCPD(variable="C", tag="linear", estimator="MLE")
    model.add_cpds(cpd_a, cpd_b, cpd_c)
    model.fit(data)

    fitted_a = model.get_cpds("A")
    fitted_b = model.get_cpds("B")
    assert isinstance(fitted_a.adapter_, TabularAdapter)
    assert isinstance(fitted_b.adapter_, LinearGaussianAdapter)
    assert "FunctionalCPD(tabular)" in repr(fitted_a)
    assert "FunctionalCPD(linear)" in repr(fitted_b)


class DummySkproRegressor:
    def fit(self, X, y):
        self.was_fit = True
        self.X_shape = X.shape
        self.y_shape = y.shape
        self.y_mean = float(np.mean(y))
        return self


def test_fit_uses_skpro_adapter_for_external_model():
    n_samples = 20
    rng = np.random.default_rng(11)
    data = pd.DataFrame(
        {
            "A": rng.normal(size=n_samples),
            "B": rng.normal(size=n_samples),
            "grade": rng.normal(loc=5.0, scale=1.0, size=n_samples),
        }
    )

    model = FunctionalBN([("A", "grade"), ("B", "grade")])
    cpd_a = FunctionalCPD(variable="A", tag="linear", estimator="MLE")
    cpd_b = FunctionalCPD(variable="B", tag="linear", estimator="MLE")
    cpd_grade = FunctionalCPD(
        variable="grade",
        tag=["skpro.BayesianLinearRegressor"],
        estimator=DummySkproRegressor(),
    )
    model.add_cpds(cpd_a, cpd_b, cpd_grade)
    model.fit(data)

    fitted_grade = model.get_cpds("grade")
    assert isinstance(fitted_grade.fitted_cpd_, SkproAdapter)
    assert fitted_grade.fitted_cpd_.model.was_fit is True
    assert fitted_grade.fitted_cpd_.model.X_shape == (n_samples, 2)
    assert fitted_grade.fitted_cpd_.model.y_shape == (n_samples,)


def test_fit_with_bayesian_linear_regressor_uses_fast_mocked_fit(monkeypatch):
    pytest.importorskip("skpro")
    pytest.importorskip("pymc_marketing")

    from skpro.regression.bayesian import BayesianLinearRegressor

    BayesianLinearRegressor._tags["python_dependencies"] = [
        "pymc",
        "pymc-marketing",
        "arviz>=0.18.0",
    ]

    recorded = {}

    def _mock_fit(self, X, y):
        recorded["x_shape"] = X.shape
        recorded["y_shape"] = y.shape
        recorded["columns"] = list(X.columns)
        return self

    monkeypatch.setattr(BayesianLinearRegressor, "fit", _mock_fit, raising=True)

    rng = np.random.default_rng(42)
    n_samples = 80
    a_data = rng.integers(0, 2, size=n_samples)
    b_data = rng.integers(0, 2, size=n_samples)
    c_data = rng.normal(loc=5.0, scale=2.0, size=n_samples)
    d_data = 2.5 * a_data - 1.5 * b_data + 3.0 * c_data + rng.normal(loc=0.0, scale=1.0, size=n_samples)

    data = pd.DataFrame({"A": a_data, "B": b_data, "C": c_data, "D": d_data})

    model = FunctionalBN([("A", "D"), ("B", "D"), ("C", "D")])
    cpd_a = FunctionalCPD(variable="A", tag="tabular", estimator=MLE)
    cpd_b = FunctionalCPD(variable="B", tag="tabular", estimator=MLE)
    cpd_c = FunctionalCPD(variable="C", tag="skpro", estimator=BayesianLinearRegressor())
    cpd_d = FunctionalCPD(variable="D", tag="linear", estimator="MLE")

    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d)
    model.fit(data)

    assert recorded["x_shape"] == (n_samples, 1)
    assert recorded["y_shape"] == (n_samples,)
    assert recorded["columns"] == ["_const_"]
