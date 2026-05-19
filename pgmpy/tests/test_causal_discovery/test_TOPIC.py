import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery.TOPIC import TOPIC
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork


@pytest.fixture
def linear_gaussian_data():
    model = LinearGaussianBayesianNetwork([("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")])
    model.add_cpds(
        LinearGaussianCPD("A", [0.0], 1.0),
        LinearGaussianCPD("B", [0.0, 1.5], 0.5, ["A"]),
        LinearGaussianCPD("C", [0.0, -1.2], 0.5, ["A"]),
        LinearGaussianCPD("D", [0.0, 0.8, 0.7], 0.5, ["B", "C"]),
    )
    return model.simulate(n_samples=2000, seed=0)


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


@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
def test_topic_recovers_known_structure(linear_gaussian_data, scoring_method):
    est = TOPIC(scoring_method=scoring_method).fit(linear_gaussian_data)

    assert est.n_features_in_ == linear_gaussian_data.shape[1]
    assert list(est.feature_names_in_) == list(linear_gaussian_data.columns)
    assert est.causal_graph_ is not None
    assert est.adjacency_matrix_.shape == (4, 4)
    assert sorted(est.topological_order_) == ["A", "B", "C", "D"]

    # AIC can learn extra edges because of its less strict penalty term.
    learned = set(est.causal_graph_.edges())
    true_edges = {("A", "B"), ("A", "C"), ("B", "D"), ("C", "D")}
    assert true_edges.issubset(learned)

    # Topological order respects the true ancestor relations.
    order = est.topological_order_
    assert order.index("A") < order.index("B")
    assert order.index("A") < order.index("C")
    assert order.index("B") < order.index("D")
    assert order.index("C") < order.index("D")
