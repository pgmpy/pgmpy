import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery.TOPIC import TOPIC


@pytest.fixture
def linear_non_gaussian_data():
    rng = np.random.default_rng(0)
    n = 2000
    A = rng.laplace(0.0, 1.0, n)
    B = 1.5 * A + rng.laplace(0.0, 0.5, n)
    C = -1.2 * A + rng.laplace(0.0, 0.5, n)
    D = 0.8 * B + 0.7 * C + rng.laplace(0.0, 0.5, n)
    return pd.DataFrame({"A": A, "B": B, "C": C, "D": D})


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [TOPIC(return_type="dag")],
    expected_failed_checks=expected_failed_checks,
)
def test_topic_compatibility(estimator, check):
    check(estimator)


def test_topic_recovers_linear_non_gaussian(linear_non_gaussian_data):
    est = TOPIC(scoring_method="bic-g").fit(linear_non_gaussian_data)

    # Attributes set by fit.
    assert est.n_features_in_ == linear_non_gaussian_data.shape[1]
    assert list(est.feature_names_in_) == list(linear_non_gaussian_data.columns)
    assert est.causal_graph_ is not None
    assert est.adjacency_matrix_.shape == (4, 4)
    assert sorted(est.topological_order_) == ["A", "B", "C", "D"]

    # All true edges should be recovered.
    learned = set(est.causal_graph_.edges())
    true_edges = {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")}
    assert true_edges.issubset(learned)

    # Topological order respects the true ancestor relations.
    order = est.topological_order_
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")


def test_topic_ignores_pairs_scoring_minus_inf_both_ways():
    rng = np.random.default_rng(0)
    n = 500
    A = rng.normal(size=n)
    C = rng.normal(size=n)
    B = A + C + 0.5 * rng.normal(size=n)
    K = np.where(A > 0, "hi", "lo").astype(object)
    K[0] = "rare"
    data = pd.DataFrame({"A": A, "B": B, "C": C, "K": pd.Categorical(K)})

    with_k = TOPIC(scoring_method="bic-cg", show_progress=False).fit(data)
    without_k = TOPIC(scoring_method="bic-cg", show_progress=False).fit(data[["A", "B", "C"]])
    assert [var for var in with_k.topological_order_ if var != "K"] == without_k.topological_order_
    assert set(with_k.causal_graph_.edges()) == set(without_k.causal_graph_.edges())
