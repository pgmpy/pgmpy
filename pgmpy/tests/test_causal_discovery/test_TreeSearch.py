"""
Tests for the sklearn-compatible TreeSearch class in pgmpy.causal_discovery.
"""

import numpy as np
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import TreeSearch
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [TreeSearch(estimator_type="chow-liu", show_progress=False)],
    expected_failed_checks=expected_failed_checks,
)
def test_treesearch_compatibility(estimator, check):
    check(estimator)


@pytest.fixture(scope="module")
def chow_liu_data():
    np.random.seed(0)
    model = DiscreteBayesianNetwork(
        [("A", "B"), ("A", "C"), ("B", "D"), ("B", "E"), ("C", "F")]
    )
    cpd_a = TabularCPD("A", 2, [[0.4], [0.6]])
    cpd_b = TabularCPD(
        "B",
        3,
        [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_c = TabularCPD(
        "C", 2, [[0.3, 0.4], [0.7, 0.6]], evidence=["A"], evidence_card=[2]
    )
    cpd_d = TabularCPD(
        "D",
        3,
        [[0.5, 0.3, 0.1], [0.4, 0.4, 0.8], [0.1, 0.3, 0.1]],
        evidence=["B"],
        evidence_card=[3],
    )
    cpd_e = TabularCPD(
        "E",
        2,
        [[0.3, 0.5, 0.2], [0.7, 0.5, 0.8]],
        evidence=["B"],
        evidence_card=[3],
    )
    cpd_f = TabularCPD(
        "F",
        3,
        [[0.3, 0.6], [0.5, 0.2], [0.2, 0.2]],
        evidence=["C"],
        evidence_card=[2],
    )

    model.add_cpds(cpd_a, cpd_b, cpd_c, cpd_d, cpd_e, cpd_f)
    inference = BayesianModelSampling(model)
    return inference.forward_sample(size=10000)


@pytest.fixture(scope="module")
def tan_data():
    np.random.seed(0)
    model = DiscreteBayesianNetwork(
        [
            ("A", "R"),
            ("A", "B"),
            ("A", "C"),
            ("A", "D"),
            ("A", "E"),
            ("R", "B"),
            ("R", "C"),
            ("R", "D"),
            ("R", "E"),
        ]
    )
    cpd_a = TabularCPD("A", 2, [[0.7], [0.3]])
    cpd_r = TabularCPD(
        "R",
        3,
        [[0.6, 0.2], [0.3, 0.5], [0.1, 0.3]],
        evidence=["A"],
        evidence_card=[2],
    )
    cpd_b = TabularCPD(
        "B",
        3,
        [
            [0.1, 0.1, 0.2, 0.2, 0.7, 0.1],
            [0.1, 0.3, 0.1, 0.2, 0.1, 0.2],
            [0.8, 0.6, 0.7, 0.6, 0.2, 0.7],
        ],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    cpd_c = TabularCPD(
        "C",
        2,
        [[0.7, 0.2, 0.2, 0.5, 0.1, 0.3], [0.3, 0.8, 0.8, 0.5, 0.9, 0.7]],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    cpd_d = TabularCPD(
        "D",
        3,
        [
            [0.3, 0.8, 0.2, 0.8, 0.4, 0.7],
            [0.4, 0.1, 0.4, 0.1, 0.1, 0.1],
            [0.3, 0.1, 0.4, 0.1, 0.5, 0.2],
        ],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    cpd_e = TabularCPD(
        "E",
        2,
        [[0.5, 0.6, 0.6, 0.5, 0.5, 0.4], [0.5, 0.4, 0.4, 0.5, 0.5, 0.6]],
        evidence=["A", "R"],
        evidence_card=[2, 3],
    )
    model.add_cpds(cpd_a, cpd_r, cpd_b, cpd_c, cpd_d, cpd_e)
    inference = BayesianModelSampling(model)
    return inference.forward_sample(size=10000)


def test_treesearch_chow_liu_expected_edges(chow_liu_data):
    est = TreeSearch(
        estimator_type="chow-liu",
        root_node="A",
        show_progress=False,
    )
    est.fit(chow_liu_data)

    expected_edges = {
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("B", "E"),
        ("C", "F"),
    }
    assert set(est.causal_graph_.nodes()) == set(["A", "B", "C", "D", "E", "F"])
    assert set(est.causal_graph_.edges()) == expected_edges
    assert est.adjacency_matrix_.shape == (6, 6)
    assert hasattr(est, "n_features_in_")
    assert hasattr(est, "feature_names_in_")


def test_treesearch_tan_expected_edges(tan_data):
    est = TreeSearch(
        estimator_type="tan",
        class_node="A",
        root_node="R",
        show_progress=False,
    )
    est.fit(tan_data)

    expected_edges = {
        ("A", "B"),
        ("A", "C"),
        ("A", "D"),
        ("A", "E"),
        ("A", "R"),
        ("R", "B"),
        ("R", "C"),
        ("R", "D"),
        ("R", "E"),
    }
    assert set(est.causal_graph_.nodes()) == set(["A", "B", "C", "D", "E", "R"])
    assert set(est.causal_graph_.edges()) == expected_edges


def test_tan_requires_class_node(tan_data):
    est = TreeSearch(estimator_type="tan", show_progress=False)
    with pytest.raises(ValueError, match="class_node"):
        est.fit(tan_data)
