import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor

from pgmpy.models import DiscreteBayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.parameter_estimator import (
    DiscreteBayesianEstimator,
    DiscreteEM,
    DiscreteMLE,
    LinearGaussianMLE,
)


@pytest.fixture
def discrete_data():
    data = pd.DataFrame(
        data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0]}
    )
    model = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
    yield {"data": data, "model": model}
    get_reusable_executor().shutdown(wait=True)


@pytest.fixture
def gaussian_data():
    rng = np.random.default_rng(42)
    data = pd.DataFrame(
        rng.normal(0, 1, (100, 3)), columns=["x1", "x2", "x3"]
    )
    model = LinearGaussianBayesianNetwork([("x1", "x2"), ("x2", "x3")])
    return {"data": data, "model": model}


@pytest.fixture
def em_data():
    data = pd.DataFrame(
        data={
            "A": [0, 0, 1, 1, 0, 1, 0, 1, 0, 1],
            "B": [0, 1, 0, 1, 0, 1, 1, 0, 0, 1],
        }
    )
    model = DiscreteBayesianNetwork(
        [("A", "C"), ("B", "C")], latents={"C"}
    )
    return {"data": data, "model": model}


# --- Test: summary before fit raises ValueError ---


def test_summary_before_fit_raises():
    estimator = DiscreteMLE()
    with pytest.raises(ValueError, match="has not been fitted yet"):
        estimator.summary()


def test_summary_before_fit_raises_bayesian():
    estimator = DiscreteBayesianEstimator()
    with pytest.raises(ValueError, match="has not been fitted yet"):
        estimator.summary()


def test_summary_before_fit_raises_gaussian():
    estimator = LinearGaussianMLE()
    with pytest.raises(ValueError, match="has not been fitted yet"):
        estimator.summary()


# --- Test: summary returns a string ---


def test_discrete_mle_summary_returns_string(discrete_data):
    estimator = DiscreteMLE().fit(discrete_data["model"], discrete_data["data"])
    result = estimator.summary()
    assert isinstance(result, str)


def test_gaussian_summary_returns_string(gaussian_data):
    estimator = LinearGaussianMLE().fit(gaussian_data["model"], gaussian_data["data"])
    result = estimator.summary()
    assert isinstance(result, str)


# --- Test: summary contains header info ---


def test_discrete_mle_summary_contains_header(discrete_data):
    estimator = DiscreteMLE().fit(discrete_data["model"], discrete_data["data"])
    result = estimator.summary()

    assert "Parameter Estimation Summary" in result
    assert "DiscreteMLE" in result
    assert "DiscreteBayesianNetwork" in result
    assert "Nodes:" in result
    assert "Edges:" in result
    assert "Samples:" in result


def test_gaussian_summary_contains_header(gaussian_data):
    estimator = LinearGaussianMLE().fit(gaussian_data["model"], gaussian_data["data"])
    result = estimator.summary()

    assert "Parameter Estimation Summary" in result
    assert "LinearGaussianMLE" in result
    assert "LinearGaussianBayesianNetwork" in result
    assert "Nodes:" in result
    assert "Samples:" in result


# --- Test: summary contains CPD info ---


def test_discrete_mle_summary_contains_cpds(discrete_data):
    estimator = DiscreteMLE().fit(discrete_data["model"], discrete_data["data"])
    result = estimator.summary()

    assert "Parameters" in result
    # CPD tables should be present for all variables
    for var in ["A", "B", "C"]:
        assert var in result


def test_gaussian_summary_contains_equations(gaussian_data):
    estimator = LinearGaussianMLE().fit(gaussian_data["model"], gaussian_data["data"])
    result = estimator.summary()

    assert "Parameters" in result
    # Gaussian CPDs show equations like P(x1) = N(...)
    assert "P(x1)" in result
    assert "P(x2 | x1)" in result
    assert "P(x3 | x2)" in result


# --- Test: compact mode ---


def test_discrete_mle_summary_compact(discrete_data):
    estimator = DiscreteMLE().fit(discrete_data["model"], discrete_data["data"])
    full = estimator.summary(compact=False)
    compact = estimator.summary(compact=True)

    # Compact should be shorter than full
    assert len(compact) < len(full)
    # Compact should still contain the header
    assert "DiscreteMLE" in compact
    # Compact uses repr which includes TabularCPD
    assert "TabularCPD" in compact


# --- Test: Bayesian estimator summary contains prior info ---


def test_bayesian_summary_contains_prior_bdeu(discrete_data):
    estimator = DiscreteBayesianEstimator(
        prior_type="BDeu", equivalent_sample_size=10
    ).fit(discrete_data["model"], discrete_data["data"])
    result = estimator.summary()

    assert "Prior" in result
    assert "BDeu" in result
    assert "Equivalent Sample Size" in result
    assert "10" in result


def test_bayesian_summary_contains_prior_k2(discrete_data):
    estimator = DiscreteBayesianEstimator(prior_type="K2").fit(
        discrete_data["model"], discrete_data["data"]
    )
    result = estimator.summary()

    assert "Prior" in result
    assert "K2" in result


# --- Test: EM summary contains convergence info ---


def test_em_summary_contains_convergence(em_data):
    estimator = DiscreteEM(max_iter=5, show_progress=False).fit(
        em_data["model"], em_data["data"]
    )
    result = estimator.summary()

    assert "Convergence" in result
    assert "Max Iterations" in result
    assert "Iterations Run" in result
    assert "Converged" in result
    assert "Tolerance (atol)" in result
    assert "M-step Estimator" in result


def test_em_summary_contains_latent_variables(em_data):
    estimator = DiscreteEM(max_iter=5, show_progress=False).fit(
        em_data["model"], em_data["data"]
    )
    result = estimator.summary()

    assert "Latent Variables" in result
    assert "C" in result


# --- Test: discrete-specific details ---


def test_discrete_summary_contains_state_counts(discrete_data):
    estimator = DiscreteMLE().fit(discrete_data["model"], discrete_data["data"])
    result = estimator.summary()

    assert "Variables:" in result
    assert "State Counts:" in result


# --- Test: gaussian-specific details ---


def test_gaussian_summary_contains_variables(gaussian_data):
    estimator = LinearGaussianMLE().fit(gaussian_data["model"], gaussian_data["data"])
    result = estimator.summary()

    assert "Variables:" in result
