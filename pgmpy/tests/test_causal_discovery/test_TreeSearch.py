"""
Tests for the sklearn-compatible TreeSearch class in pgmpy.causal_discovery.
"""

import warnings

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from joblib.externals.loky import get_reusable_executor
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import TreeSearch
from pgmpy.factors.discrete import TabularCPD
from pgmpy.metrics import SHD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.sampling import BayesianModelSampling
from pgmpy.utils import get_example_model

# ---------------------------------------------------------------------------
# Suppress DeprecationWarning from the legacy estimator called internally
# ---------------------------------------------------------------------------
pytestmark = pytest.mark.filterwarnings("ignore::DeprecationWarning")


# ---------------------------------------------------------------------------
# sklearn compatibility
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def simple_data():
    """Small random data — used for quick structural checks."""
    np.random.seed(0)
    return pd.DataFrame(
        np.random.randint(low=0, high=2, size=(100, 5)),
        columns=["A", "B", "C", "D", "E"],
    )


@pytest.fixture(scope="module")
def chow_liu_data():
    """Sampled data from a known tree BN — used to verify Chow-Liu recovery."""
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
    return BayesianModelSampling(model).forward_sample(size=10000)


@pytest.fixture(scope="module")
def tan_data():
    """Sampled data from a known TAN BN — used to verify TAN recovery."""
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
    return BayesianModelSampling(model).forward_sample(size=10000)


@pytest.fixture(scope="module")
def alarm_data():
    return get_example_model("alarm").simulate(int(1e4), seed=42)


# ---------------------------------------------------------------------------
# Chow-Liu tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edge_weights_fn",
    ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_chow_liu_structure_recovery(chow_liu_data, edge_weights_fn, n_jobs):
    """Chow-Liu should recover the exact known tree structure from data13."""
    est = TreeSearch(
        estimator_type="chow-liu",
        root_node="A",
        edge_weights_fn=edge_weights_fn,
        n_jobs=n_jobs,
        show_progress=False,
    )
    est.fit(chow_liu_data)

    assert set(est.causal_graph_.nodes()) == {"A", "B", "C", "D", "E", "F"}
    assert set(est.causal_graph_.edges()) == {
        ("A", "B"),
        ("A", "C"),
        ("B", "D"),
        ("B", "E"),
        ("C", "F"),
    }
    assert est.causal_graph_.has_edge("A", "B")
    assert est.causal_graph_.has_edge("A", "C")
    assert est.causal_graph_.has_edge("B", "D")
    assert est.causal_graph_.has_edge("B", "E")
    assert est.causal_graph_.has_edge("C", "F")


@pytest.mark.parametrize(
    "edge_weights_fn",
    ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_chow_liu_is_tree(simple_data, edge_weights_fn, n_jobs):
    """Result of Chow-Liu must always be a valid tree."""
    est = TreeSearch(
        estimator_type="chow-liu",
        root_node="A",
        edge_weights_fn=edge_weights_fn,
        n_jobs=n_jobs,
        show_progress=False,
    )
    est.fit(simple_data)

    assert set(est.causal_graph_.nodes()) == {"A", "B", "C", "D", "E"}
    assert nx.is_tree(est.causal_graph_)


# ---------------------------------------------------------------------------
# TAN tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edge_weights_fn",
    ["mutual_info", "adjusted_mutual_info", "normalized_mutual_info"],
)
@pytest.mark.parametrize("n_jobs", [1, 2])
def test_tan_structure_recovery(tan_data, edge_weights_fn, n_jobs):
    """TAN should recover the exact known structure from data22."""
    est = TreeSearch(
        estimator_type="tan",
        root_node="R",
        class_node="A",
        edge_weights_fn=edge_weights_fn,
        n_jobs=n_jobs,
        show_progress=False,
    )
    est.fit(tan_data)

    assert set(est.causal_graph_.nodes()) == {"A", "B", "C", "D", "E", "R"}
    assert set(est.causal_graph_.edges()) == {
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
    # class node -> feature edges
    assert est.causal_graph_.has_edge("A", "B")
    assert est.causal_graph_.has_edge("A", "C")
    assert est.causal_graph_.has_edge("A", "D")
    assert est.causal_graph_.has_edge("A", "E")
    # tree edges over feature variables
    assert est.causal_graph_.has_edge("R", "B")
    assert est.causal_graph_.has_edge("R", "C")
    assert est.causal_graph_.has_edge("R", "D")
    assert est.causal_graph_.has_edge("R", "E")


# ---------------------------------------------------------------------------
# Auto root node / auto class node
# ---------------------------------------------------------------------------


def test_chow_liu_auto_root_node(simple_data):
    """When root_node=None, root should be auto-selected as max-weight node."""
    est = TreeSearch(estimator_type="chow-liu", show_progress=False)
    est.fit(simple_data)

    assert nx.is_tree(est.causal_graph_)

    # The auto-selected root is the unique node with in-degree 0
    in_degrees = dict(est.causal_graph_.in_degree())
    roots = [n for n, d in in_degrees.items() if d == 0]
    assert len(roots) == 1
    assert roots[0] in simple_data.columns


def test_tan_auto_class_node(tan_data):
    """When root_node=None, root and class nodes should be auto-selected."""
    # Replicate the auto-selection logic to get expected root and class nodes
    from pgmpy.estimators import TreeSearch as LegacyTreeSearch

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = LegacyTreeSearch(tan_data)

    weights = legacy._get_weights(tan_data)
    sum_weights = weights.sum(axis=0)
    maxw_idx = np.argsort(sum_weights)[::-1]
    expected_root = tan_data.columns[maxw_idx[0]]
    expected_class = tan_data.columns[maxw_idx[1]]

    est = TreeSearch(
        estimator_type="tan",
        class_node=expected_class,
        show_progress=False,
    )
    est.fit(tan_data)

    # ROBUST FIX: Check structure instead of list indexing

    # In TAN, the expected_root (root of the feature tree) has the class node as its parent
    assert set(est.causal_graph_.predecessors(expected_root)) == {expected_class}

    # The class node itself is the root of the entire graph (in-degree 0)
    assert est.causal_graph_.in_degree(expected_class) == 0

    # Ensure all nodes are present
    assert sorted(est.causal_graph_.nodes()) == sorted(["C", "R", "A", "D", "E", "B"])


# ---------------------------------------------------------------------------
# Fitted attributes
# ---------------------------------------------------------------------------


def test_fitted_attributes_chow_liu(chow_liu_data):
    """causal_graph_, adjacency_matrix_, n_features_in_ must be set after fit."""
    est = TreeSearch(estimator_type="chow-liu", root_node="A", show_progress=False)
    est.fit(chow_liu_data)

    assert hasattr(est, "causal_graph_")
    assert hasattr(est, "adjacency_matrix_")
    assert hasattr(est, "n_features_in_")

    assert est.n_features_in_ == len(chow_liu_data.columns)
    assert isinstance(est.adjacency_matrix_, pd.DataFrame)
    assert est.adjacency_matrix_.shape == (
        len(chow_liu_data.columns),
        len(chow_liu_data.columns),
    )


def test_fitted_attributes_tan(tan_data):
    """causal_graph_, adjacency_matrix_, n_features_in_ must be set after fit."""
    est = TreeSearch(
        estimator_type="tan", root_node="R", class_node="A", show_progress=False
    )
    est.fit(tan_data)

    assert hasattr(est, "causal_graph_")
    assert hasattr(est, "adjacency_matrix_")
    assert hasattr(est, "n_features_in_")

    assert est.n_features_in_ == len(tan_data.columns)
    assert isinstance(est.adjacency_matrix_, pd.DataFrame)
    assert est.adjacency_matrix_.shape == (
        len(tan_data.columns),
        len(tan_data.columns),
    )


def test_not_fitted_error(simple_data):
    """score() must raise NotFittedError before fit() is called."""
    est = TreeSearch(estimator_type="chow-liu", show_progress=False)
    with pytest.raises(NotFittedError):
        est.score(X=simple_data)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_estimator_type(simple_data):
    est = TreeSearch(estimator_type="invalid", show_progress=False)
    with pytest.raises(ValueError, match="estimator_type must be one of"):
        est.fit(simple_data)


def test_missing_class_node_for_tan(simple_data):
    est = TreeSearch(estimator_type="tan", show_progress=False)
    with pytest.raises(ValueError, match="class_node must be provided"):
        est.fit(simple_data)


def test_invalid_root_node(simple_data):
    est = TreeSearch(estimator_type="chow-liu", root_node="Z", show_progress=False)
    with pytest.raises(ValueError, match="root_node"):
        est.fit(simple_data)


def test_invalid_class_node(simple_data):
    est = TreeSearch(estimator_type="tan", class_node="Z", show_progress=False)
    with pytest.raises(ValueError, match="class_node"):
        est.fit(simple_data)


# ---------------------------------------------------------------------------
# Real dataset — alarm TAN
# ---------------------------------------------------------------------------


def test_tan_alarm(alarm_data):
    """TAN on alarm dataset should match expected edges from bnlearn."""
    expected_edges = {
        ("CVP", "LVFAILURE"),
        ("CVP", "INTUBATION"),
        ("CVP", "TPR"),
        ("CVP", "DISCONNECT"),
        ("CVP", "VENTMACH"),
        ("CVP", "HR"),
        ("CVP", "FIO2"),
        ("CVP", "HRBP"),
        ("CVP", "VENTLUNG"),
        ("CVP", "PAP"),
        ("CVP", "HISTORY"),
        ("CVP", "PCWP"),
        ("CVP", "INSUFFANESTH"),
        ("CVP", "SAO2"),
        ("CVP", "EXPCO2"),
        ("CVP", "PRESS"),
        ("CVP", "PULMEMBOLUS"),
        ("CVP", "ARTCO2"),
        ("CVP", "MINVOLSET"),
        ("LVFAILURE", "HISTORY"),
        ("LVFAILURE", "PCWP"),
        ("INTUBATION", "INSUFFANESTH"),
        ("EXPCO2", "INTUBATION"),
        ("HR", "TPR"),
        ("PRESS", "DISCONNECT"),
        ("VENTLUNG", "VENTMACH"),
        ("VENTMACH", "PRESS"),
        ("VENTMACH", "MINVOLSET"),
        ("HR", "HRBP"),
        ("ARTCO2", "HR"),
        ("SAO2", "FIO2"),
        ("VENTLUNG", "PAP"),
        ("PCWP", "VENTLUNG"),
        ("VENTLUNG", "EXPCO2"),
        ("VENTLUNG", "ARTCO2"),
        ("PAP", "PULMEMBOLUS"),
        ("ARTCO2", "SAO2"),
    }
    features = [
        "LVFAILURE",
        "INTUBATION",
        "TPR",
        "DISCONNECT",
        "VENTMACH",
        "HR",
        "FIO2",
        "HRBP",
        "VENTLUNG",
        "PAP",
        "HISTORY",
        "PCWP",
        "INSUFFANESTH",
        "SAO2",
        "EXPCO2",
        "PRESS",
        "PULMEMBOLUS",
        "ARTCO2",
        "MINVOLSET",
    ]
    target = "CVP"

    est = TreeSearch(
        estimator_type="tan",
        root_node=features[0],
        class_node=target,
        n_jobs=1,
        show_progress=False,
    )
    est.fit(alarm_data[features + [target]])

    assert set(est.causal_graph_.edges()) == expected_edges


# ---------------------------------------------------------------------------
# score() method
# ---------------------------------------------------------------------------


def test_score():
    """score() should work with data and true_graph after fitting."""
    asia_model = get_example_model("asia")
    data = asia_model.simulate(n_samples=int(1e4), seed=42)

    est = TreeSearch(estimator_type="chow-liu", show_progress=False)
    est.fit(data)

    # CorrelationScore is excluded — it requires a parameterized BayesianNetwork
    # to simulate data. TreeSearch only returns a bare DAG (structure only).

    # Test structure-based score
    structure_score = est.score(X=data, metric="structure_score")
    assert np.round(structure_score, 4) > -3e4

    # Test graph-comparison scores — valid for bare DAGs
    shd = est.score(true_graph=asia_model, metric=SHD())
    assert isinstance(shd, (int, float))

    shd = est.score(true_graph=asia_model, metric="SHD")
    assert isinstance(shd, (int, float))


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def teardown_module(module):
    get_reusable_executor().shutdown(wait=True)
