from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import LinearGaussianBayesianEstimator
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


@pytest.fixture
def estimator_one_parent():
    """Y = 2*X + 1 + N(0, 0.5²)."""
    model = LinearGaussianBayesianNetwork([("X", "Y")])
    np.random.seed(1)
    x = np.random.randn(300)
    y = 2.0 * x + 1.0 + np.random.randn(300) * 0.5
    return LinearGaussianBayesianEstimator(model, pd.DataFrame({"X": x, "Y": y}))


@pytest.fixture
def estimator_two_parents():
    """Z = 1.5*X1 - 0.8*X2 + 0.5 + N(0, 0.3²)."""
    model = LinearGaussianBayesianNetwork([("X1", "Z"), ("X2", "Z")])
    np.random.seed(42)
    x1, x2 = np.random.randn(500), np.random.randn(500)
    z = 1.5 * x1 - 0.8 * x2 + 0.5 + np.random.randn(500) * 0.3
    return LinearGaussianBayesianEstimator(
        model, pd.DataFrame({"X1": x1, "X2": x2, "Z": z})
    )


@pytest.fixture
def estimator_get_params():
    """Y = 3*X + N(0, 0.5²)."""
    model = LinearGaussianBayesianNetwork([("X", "Y")])
    np.random.seed(7)
    x = np.random.randn(200)
    y = 3.0 * x + np.random.randn(200) * 0.5
    return LinearGaussianBayesianEstimator(model, pd.DataFrame({"X": x, "Y": y}))


@pytest.fixture
def estimator_posterior():
    """Small deterministic dataset."""
    model = LinearGaussianBayesianNetwork([("X", "Y")])
    v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    return LinearGaussianBayesianEstimator(model, pd.DataFrame({"X": v, "Y": v}))


class TestInit:

    def test_init(self):
        """Valid model stores data; invalid model raises error."""
        model = LinearGaussianBayesianNetwork([("X", "Y")])
        np.random.seed(42)
        data = pd.DataFrame({"X": np.random.randn(100), "Y": np.random.randn(100)})

        est = LinearGaussianBayesianEstimator(model, data)
        assert isinstance(est, LinearGaussianBayesianEstimator)
        pd.testing.assert_frame_equal(est.data, data)

        with pytest.raises(NotImplementedError):
            LinearGaussianBayesianEstimator(MagicMock(), data)


class TestEstimateCpd:

    def test_one_parent_structure_and_accuracy(self, estimator_one_parent):
        """Single-parent regression recovers true parameters.
        Also verifies the root node (no-parent) branch produces an intercept-only CPD.
        """
        # root node: no parents → intercept only
        cpd_root = estimator_one_parent.estimate_cpd("X")
        assert isinstance(cpd_root, LinearGaussianCPD)
        assert cpd_root.variable == "X"
        assert cpd_root.evidence == []
        assert np.asarray(cpd_root.beta).reshape(-1).shape == (1,)
        assert cpd_root.std > 0

        # child node: one parent → intercept + slope
        cpd = estimator_one_parent.estimate_cpd("Y")
        beta = np.asarray(cpd.beta).reshape(-1)

        assert isinstance(cpd, LinearGaussianCPD)
        assert cpd.variable == "Y"
        assert cpd.evidence == ["X"]
        assert beta.shape == (2,)
        assert cpd.std > 0
        assert beta[0] == pytest.approx(1.0, abs=0.2)
        assert beta[1] == pytest.approx(2.0, abs=0.2)
        assert cpd.std == pytest.approx(0.5, abs=0.15)

    def test_two_parents_sorted_and_accurate(self, estimator_two_parents):
        """Parent ordering and coefficients are correct."""
        cpd = estimator_two_parents.estimate_cpd("Z")
        beta = np.asarray(cpd.beta).reshape(-1)

        assert cpd.evidence == ["X1", "X2"]
        assert beta.shape == (3,)
        assert beta[0] == pytest.approx(0.5, abs=0.15)
        assert beta[1] == pytest.approx(1.5, abs=0.15)
        assert beta[2] == pytest.approx(-0.8, abs=0.15)

    def test_informative_prior_shifts_posterior(self, estimator_one_parent):
        """Tight prior should pull posterior intercept."""
        cpd_flat = estimator_one_parent.estimate_cpd("Y")
        cpd_informed = estimator_one_parent.estimate_cpd(
            "Y", B0=np.array([-50.0, 0.0]), V0=np.eye(2) * 1e-6
        )
        assert (
            np.asarray(cpd_informed.beta).reshape(-1)[0]
            < np.asarray(cpd_flat.beta).reshape(-1)[0]
        )

    def test_prior_validation_raises(self, estimator_one_parent):
        """Invalid prior dimensions should raise errors."""
        with pytest.raises(ValueError):
            estimator_one_parent.estimate_cpd("Y", B0=np.array([0.0]), V0=np.eye(2))
        with pytest.raises(ValueError):
            estimator_one_parent.estimate_cpd("Y", B0=np.zeros(2), V0=np.eye(3))


class TestGetParameters:

    def test_output_values_and_optional_args(self, estimator_get_params):
        """Check structure, correctness, dict priors, and parallel execution."""
        model = estimator_get_params.model

        params = estimator_get_params.get_parameters()
        assert isinstance(params, list)
        assert len(params) == len(model.nodes())
        assert all(isinstance(p, LinearGaussianCPD) for p in params)

        params_map = {p.variable: p for p in params}
        for node in model.nodes():
            direct = estimator_get_params.estimate_cpd(node)
            np.testing.assert_array_almost_equal(
                np.asarray(params_map[node].beta).reshape(-1),
                np.asarray(direct.beta).reshape(-1),
            )
            assert params_map[node].std == pytest.approx(direct.std, rel=1e-6)

        # dict priors
        params_dict = estimator_get_params.get_parameters(
            B0={"X": np.array([0.0]), "Y": np.array([0.0, 0.0])},
            V0={"X": np.eye(1) * 5, "Y": np.eye(2) * 5},
            alpha_0={"X": 3.0, "Y": 3.0},
            beta_0={"X": 2.0, "Y": 2.0},
        )
        assert len(params_dict) == len(model.nodes())

        # parallel execution
        single = {p.variable: p for p in estimator_get_params.get_parameters(n_jobs=1)}
        parallel = {
            p.variable: p for p in estimator_get_params.get_parameters(n_jobs=2)
        }
        for node in model.nodes():
            np.testing.assert_array_almost_equal(
                np.asarray(single[node].beta).reshape(-1),
                np.asarray(parallel[node].beta).reshape(-1),
            )

    def test_prior_validation_raises(self, estimator_get_params):
        """Invalid B0/V0 shapes should raise errors."""
        with pytest.raises(ValueError):
            estimator_get_params.get_parameters(B0=np.zeros(3))
        with pytest.raises(ValueError):
            estimator_get_params.get_parameters(
                B0={"X": np.array([0.0, 9.9]), "Y": np.zeros(2)}
            )
        with pytest.raises(ValueError):
            estimator_get_params.get_parameters(V0={"X": np.eye(3), "Y": np.eye(2)})


class TestPosteriorBehavior:

    def test_prior_sensitivity(self, estimator_posterior):
        """Flat prior → OLS; tight prior → dominates data; larger alpha_0 → smaller std."""
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        # flat prior converges to sample mean
        cpd_flat = estimator_posterior.estimate_cpd(
            "X", B0=np.array([0.0]), V0=np.eye(1) * 1e6
        )
        assert np.asarray(cpd_flat.beta).reshape(-1)[0] == pytest.approx(
            np.mean(v), abs=1e-3
        )

        # tight prior anchored at 100 dominates the 5 data points
        cpd_tight = estimator_posterior.estimate_cpd(
            "X", B0=np.array([100.0]), V0=np.eye(1) * 1e-6
        )
        assert np.asarray(cpd_tight.beta).reshape(-1)[0] > 50.0

        # larger alpha_0 tightens variance
        cpd_low = estimator_posterior.estimate_cpd("X", alpha_0=2.0, beta_0=0.1)
        cpd_high = estimator_posterior.estimate_cpd("X", alpha_0=200.0, beta_0=0.1)
        assert cpd_high.std < cpd_low.std

    def test_edge_cases(self):
        """Large n recovers slope; n=1 still works."""
        np.random.seed(3)
        x = np.random.randn(10_000)
        y = 0.5 * x + np.random.randn(10_000)
        cpd = LinearGaussianBayesianEstimator(
            LinearGaussianBayesianNetwork([("X", "Y")]),
            pd.DataFrame({"X": x, "Y": y}),
        ).estimate_cpd("Y")
        assert np.asarray(cpd.beta).reshape(-1)[1] == pytest.approx(0.5, abs=0.05)

        cpd_single = LinearGaussianBayesianEstimator(
            LinearGaussianBayesianNetwork([("X", "Y")]),
            pd.DataFrame({"X": [1.0], "Y": [2.0]}),
        ).estimate_cpd("Y")
        assert isinstance(cpd_single, LinearGaussianCPD)
