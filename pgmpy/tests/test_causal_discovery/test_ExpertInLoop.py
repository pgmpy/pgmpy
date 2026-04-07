"""
Tests for the sklearn-compatible ExpertInLoop class in pgmpy.causal_discovery
"""

import logging
import sys
from functools import partial
from unittest.mock import patch

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertInLoop
from pgmpy.ci_tests._base import _BaseCITest
from pgmpy.estimators import ExpertKnowledge


def simple_orient(var1, var2, **kwargs):
    """Simple orientation function (module-level for pickling support)."""
    return (var1, var2) if var1 < var2 else (var2, var1)


def make_estimator():
    """Create an ExpertInLoop estimator with a simple orientation function."""
    return ExpertInLoop(orientation_fn=simple_orient, show_progress=False)


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
        "check_positive_only_tag_during_fit": (
            "Fails due to numpy internal cast issue on estimator test data when X -= X.mean()"
        ),
    }


@parametrize_with_checks(
    [make_estimator()],
    expected_failed_checks=expected_failed_checks,
)
def test_expertinloop_compatibility(estimator, check):
    check(estimator)


# --- Fixtures ---
@pytest.fixture
def adult_data():
    """Load and preprocess the adult dataset."""
    df = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult_proc.csv", index_col=0)
    df.Age = pd.Categorical(
        df.Age,
        categories=["<21", "21-30", "31-40", "41-50", "51-60", "61-70", ">70"],
        ordered=True,
    )
    df.Education = pd.Categorical(
        df.Education,
        categories=[
            "Preschool",
            "1st-4th",
            "5th-6th",
            "7th-8th",
            "9th",
            "10th",
            "11th",
            "12th",
            "HS-grad",
            "Some-college",
            "Assoc-voc",
            "Assoc-acdm",
            "Bachelors",
            "Prof-school",
            "Masters",
            "Doctorate",
        ],
        ordered=True,
    )
    df.HoursPerWeek = pd.Categorical(df.HoursPerWeek, categories=["<=20", "21-30", "31-40", ">40"], ordered=True)
    df.Workclass = pd.Categorical(df.Workclass, ordered=False)
    df.MaritalStatus = pd.Categorical(df.MaritalStatus, ordered=False)
    df.Occupation = pd.Categorical(df.Occupation, ordered=False)
    df.Relationship = pd.Categorical(df.Relationship, ordered=False)
    df.Race = pd.Categorical(df.Race, ordered=False)
    df.Sex = pd.Categorical(df.Sex, ordered=False)
    df.NativeCountry = pd.Categorical(df.NativeCountry, ordered=False)
    df.Income = pd.Categorical(df.Income, ordered=False)
    return df


@pytest.fixture
def adult_data_small(adult_data):
    return adult_data[["Age", "Education", "Race", "Sex", "Income"]]


@pytest.fixture
def descriptions():
    """Descriptions of the variables in the adult dataset."""
    return {
        "Age": "The age of a person",
        "Workclass": "The workplace where the person is employed such as Private industry, or self employed",
        "Education": "The highest level of education the person has finished",
        "MaritalStatus": "The marital status of the person",
        "Occupation": "The kind of job the person does. For example, sales, craft repair, clerical",
        "Relationship": "The relationship status of the person",
        "Race": "The ethnicity of the person",
        "Sex": "The sex or gender of the person",
        "HoursPerWeek": "The number of hours per week the person works",
        "NativeCountry": "The native country of the person",
        "Income": "The income i.e. amount of money the person makes",
    }


@pytest.fixture
def orientations_small():
    """Pre-defined orientations for small dataset tests."""
    return {
        ("Education", "Income"),
        ("Race", "Education"),
        ("Age", "Education"),
    }


@pytest.fixture
def true_dag_edges():
    """True edges for adult dataset."""
    return [
        # Education-related paths
        ("Age", "Education"),
        ("Race", "Education"),
        ("NativeCountry", "Education"),
        # Income-related paths
        ("Education", "Income"),
        ("Occupation", "Income"),
        ("HoursPerWeek", "Income"),
        ("MaritalStatus", "Income"),
        # Occupation-related paths
        ("Age", "Occupation"),
        ("Education", "Occupation"),
        ("Sex", "Occupation"),
        ("Workclass", "Occupation"),
        # HoursPerWeek-related paths
        ("Age", "HoursPerWeek"),
        ("Workclass", "HoursPerWeek"),
        ("Occupation", "HoursPerWeek"),
        ("Education", "HoursPerWeek"),
        # Relationship and MaritalStatus paths
        ("Age", "MaritalStatus"),
        ("Sex", "MaritalStatus"),
        ("MaritalStatus", "Relationship"),
        ("Age", "Relationship"),
        ("Sex", "Relationship"),
        # Other reasonable connections
        ("Race", "NativeCountry"),
        ("Workclass", "MaritalStatus"),
        ("Workclass", "Relationship"),
    ]


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
def test_estimate(adult_data, true_dag_edges):
    """Test basic estimation with oracle orientation function."""
    true_dag = nx.DiGraph(true_dag_edges)
    true_dag.add_nodes_from(adult_data.columns)

    def oracle_orient(var1, var2, **kwargs):
        """Orientation function that knows the 'true' structure."""
        if true_dag.has_edge(var1, var2):
            return (var1, var2)
        elif true_dag.has_edge(var2, var1):
            return (var2, var1)
        else:
            return None

    estimator = ExpertInLoop(
        orientation_fn=oracle_orient,
        pval_threshold=0.05,
        effect_size_threshold=0.05,
        show_progress=False,
    )
    estimator.fit(adult_data)

    for u, v in estimator.causal_graph_.edges():
        assert true_dag.has_edge(u, v)

    assert nx.is_directed_acyclic_graph(estimator.causal_graph_)


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
def test_estimate_with_orientations(adult_data_small, orientations_small):
    """Test estimation with pre-specified orientations."""
    estimator = ExpertInLoop(
        orientation_fn=simple_orient,
        orientations=orientations_small,
        pval_threshold=0.1,
        effect_size_threshold=0.1,
        show_progress=False,
    )
    estimator.fit(adult_data_small)

    # Check that pre-specified orientations are present in the graph
    for edge in orientations_small:
        assert edge in estimator.causal_graph_.edges(), f"Pre-specified orientation {edge} not found in learned graph"


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
def test_estimate_with_cache(adult_data_small, orientations_small):
    """Test estimation with cached orientations."""
    # Created estimator with pre-specified orientations
    estimator = ExpertInLoop(
        orientation_fn=simple_orient,
        orientations=orientations_small,
        use_cache=True,
        pval_threshold=0.1,
        effect_size_threshold=0.1,
        show_progress=False,
    )
    # fit() re-initializes the cache but uses `orientations` parameter during fit
    estimator.fit(adult_data_small)

    assert orientations_small.issubset(set(estimator.causal_graph_.edges()))
    # Cache should be populated after fit()
    assert len(estimator.orientation_cache_) > 0


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
def test_estimate_with_custom_orient_fn(adult_data_small):
    """Test estimation with custom orientation function."""

    def custom_orient(var1, var2, **kwargs):
        # orient edges from alphabetically first to second
        if var1 < var2:
            return (var1, var2)
        else:
            return (var2, var1)

    estimator = ExpertInLoop(
        orientation_fn=custom_orient,
        pval_threshold=0.1,
        effect_size_threshold=0.1,
        show_progress=False,
    )
    estimator.fit(adult_data_small)

    # Check that all edges are oriented from alphabetically lower to higher
    for edge in estimator.causal_graph_.edges():
        assert edge[0] < edge[1]

    # Check that orientations were cached
    assert len(estimator.orientation_cache_) > 0
    for edge in estimator.orientation_cache_:
        assert edge[0] < edge[1]


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
def test_estimate_with_orient_fn_kwargs(adult_data_small):
    """Test that orientation function works with different configurations."""

    def make_orient_fn(reverse_alphabetical=False):
        """Create an orientation function with specific configuration."""

        def orient_fn(var1, var2):
            # Use the captured reverse_alphabetical parameter
            if reverse_alphabetical:
                if var1 > var2:
                    return (var1, var2)
                else:
                    return (var2, var1)
            else:
                if var1 < var2:
                    return (var1, var2)
                else:
                    return (var2, var1)

        return orient_fn

    # Test with reverse_alphabetical=True
    estimator = ExpertInLoop(
        orientation_fn=make_orient_fn(reverse_alphabetical=True),
        pval_threshold=0.1,
        effect_size_threshold=0.1,
        show_progress=False,
    )
    estimator.fit(adult_data_small)

    # Check that all edges are oriented from alphabetically higher to lower
    for edge in estimator.causal_graph_.edges():
        assert edge[0] > edge[1]


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
def test_combined_expert_knowledge(adult_data):
    """Test combination of forbidden edges, required edges, and temporal order."""
    expert_knowledge = ExpertKnowledge(
        forbidden_edges=[("Age", "Income")],
        required_edges=[("Education", "Income")],
        temporal_order=[["Age", "Race"], ["Education"], ["Income", "HoursPerWeek"]],
    )

    estimator = ExpertInLoop(
        expert_knowledge=expert_knowledge,
        effect_size_threshold=0.0001,
        orientation_fn=simple_orient,
        show_progress=False,
    )
    estimator.fit(adult_data)

    # Check forbidden edges
    assert ("Age", "Income") not in estimator.causal_graph_.edges()

    # Check temporal order
    for u, v in estimator.causal_graph_.edges():
        if u in expert_knowledge.temporal_ordering and v in expert_knowledge.temporal_ordering:
            u_order = expert_knowledge.temporal_ordering[u]
            v_order = expert_knowledge.temporal_ordering[v]
            assert u_order <= v_order, f"Edge {u}->{v} violates temporal order"


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
def test_edge_orientation_priority(adult_data):
    """Test that edge orientation follows the correct priority order."""
    expert_knowledge = ExpertKnowledge(temporal_order=[["Age", "Race"], ["Education"], ["Income", "HoursPerWeek"]])

    # Define orientations that should take precedence over temporal order
    orientations = {("Income", "Education")}  # Opposite of temporal order

    estimator = ExpertInLoop(
        expert_knowledge=expert_knowledge,
        orientations=orientations,
        effect_size_threshold=0.0001,
        orientation_fn=simple_orient,
        show_progress=False,
    )
    estimator.fit(adult_data)

    # Check that specified orientations take precedence
    if ("Income", "Education") in estimator.causal_graph_.edges():
        assert ("Education", "Income") not in estimator.causal_graph_.edges()


def test_fitted_attributes():
    """Test that fitted attributes are properly set."""
    # Create simple data
    np.random.seed(42)
    data = pd.DataFrame(
        {"A": np.random.choice([0, 1], 100), "B": np.random.choice([0, 1], 100)},
        dtype="category",
    )

    def simple_orient(var1, var2, **kwargs):
        return (var1, var2) if var1 < var2 else (var2, var1)

    estimator = ExpertInLoop(
        orientation_fn=simple_orient,
        effect_size_threshold=0.0001,
        show_progress=False,
    )
    estimator.fit(data)

    # Check fitted attributes
    assert hasattr(estimator, "causal_graph_")
    assert hasattr(estimator, "adjacency_matrix_")
    assert hasattr(estimator, "variables_")
    assert hasattr(estimator, "orientation_cache_")
    assert hasattr(estimator, "n_features_in_")
    assert hasattr(estimator, "feature_names_in_")

    assert estimator.n_features_in_ == 2
    assert set(estimator.variables_) == {"A", "B"}


def test_adjacency_matrix():
    """Test that adjacency matrix is correctly formed."""
    np.random.seed(42)
    data = pd.DataFrame(
        {
            "A": np.random.choice([0, 1], 100),
            "B": np.random.choice([0, 1], 100),
            "C": np.random.choice([0, 1], 100),
        },
        dtype="category",
    )

    def simple_orient(var1, var2, **kwargs):
        return (var1, var2) if var1 < var2 else (var2, var1)

    estimator = ExpertInLoop(
        orientation_fn=simple_orient,
        effect_size_threshold=0.0001,
        show_progress=False,
    )
    estimator.fit(data)

    # Check adjacency matrix structure
    adj_matrix = estimator.adjacency_matrix_
    assert adj_matrix.shape == (3, 3)
    assert set(adj_matrix.index) == {"A", "B", "C"}
    assert set(adj_matrix.columns) == {"A", "B", "C"}

    # Check consistency between adjacency matrix and causal graph
    for u, v in estimator.causal_graph_.edges():
        assert adj_matrix.loc[u, v] == 1


def test_empty_graph():
    """Test behavior when no edges are added."""
    # Create independent data
    np.random.seed(42)
    data = pd.DataFrame(
        {"A": np.random.choice([0, 1], 100), "B": np.random.choice([0, 1], 100)},
        dtype="category",
    )

    # Use very high thresholds to ensure no edges are added
    estimator = ExpertInLoop(
        orientation_fn=simple_orient,
        effect_size_threshold=1.0,  # Very high threshold
        pval_threshold=0.0,  # Very low p-value threshold
        show_progress=False,
    )
    estimator.fit(data)

    # Should have nodes but no edges
    assert set(estimator.causal_graph_.nodes()) == {"A", "B"}


# --- _break_cycle unit tests ---


@pytest.fixture
def fake_ci_estimator():
    data = pd.DataFrame(
        {
            "A": [1, 2, 3, 4, 5],
            "B": [2, 3, 4, 5, 6],
            "C": [3, 4, 5, 6, 7],
            "D": [4, 5, 6, 7, 8],
        }
    )
    return ExpertInLoop(orientation_fn=simple_orient, show_progress=False), data


@pytest.fixture
def simple_dag():
    dag = DAG()
    dag.add_nodes_from(["A", "B", "C"])
    dag.add_edges_from([("A", "B"), ("B", "C")])
    return dag


class WeakCI(_BaseCITest):
    def __init__(self, data):
        self.data = data
        super().__init__()

    def run_test(self, X, Y, Z):
        self.statistic_ = 0.01
        self.p_value_ = 0.9
        return (0.01, 0.9)


class StrongCI(_BaseCITest):
    def __init__(self, data):
        self.data = data
        super().__init__()

    def run_test(self, X, Y, Z):
        self.statistic_ = 0.5
        self.p_value_ = 0.001
        return (0.5, 0.001)


class MockCI(_BaseCITest):
    def __init__(self, data):
        self.data = data
        super().__init__()

    def run_test(self, X, Y, Z):
        if {X, Y} == {"A", "B"}:
            self.statistic_ = 0.01
            self.p_value_ = 0.9
            return (self.statistic_, self.p_value_)
        else:
            self.statistic_ = 0.5
            self.p_value_ = 0.001
            return (self.statistic_, self.p_value_)


class TestBreakCycle:
    def test_all_weak_edges_removed(self, fake_ci_estimator, simple_dag):
        estimator, data = fake_ci_estimator
        result = estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=WeakCI(data),
            data=data,
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert len(result) > 0
        for edge in result:
            assert edge in [("A", "B"), ("B", "C")]

    def test_all_strong_edges_kept(self, fake_ci_estimator, simple_dag):
        estimator, data = fake_ci_estimator
        result = estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=StrongCI(data),
            data=data,
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert result == []

    def test_selective_removal(self, fake_ci_estimator, simple_dag):
        estimator, data = fake_ci_estimator

        # def mock_ci_test(X, Y, Z, data, boolean):
        #     # A->B is weak, everything else is strong
        #     if set([X, Y]) == {"A", "B"}:
        #         return (0.01, 0.9)
        #     return (0.5, 0.001)

        result = estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=MockCI(data),
            data=data,
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert ("B", "C") not in result
        assert ("C", "A") not in result

    def test_new_edge_never_in_result(self, fake_ci_estimator, simple_dag):
        estimator, data = fake_ci_estimator
        result = estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=WeakCI(data),
            data=data,
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert ("C", "A") not in result

    def test_original_dag_not_modified(self, fake_ci_estimator, simple_dag):
        estimator, data = fake_ci_estimator
        original_edges = set(simple_dag.edges())
        original_nodes = set(simple_dag.nodes())

        estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=WeakCI(data),
            data=data,
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert set(simple_dag.edges()) == original_edges
        assert set(simple_dag.nodes()) == original_nodes

    def test_longer_cycle(self, fake_ci_estimator):
        """Test with a 4-node cycle: A -> B -> C -> D, adding D -> A."""
        estimator, data = fake_ci_estimator
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C", "D"])
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

        result = estimator._break_cycle(
            dag,
            "D",
            "A",
            ci_test=WeakCI(data),
            data=data,
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert len(result) > 0
        for edge in result:
            assert edge in [("A", "B"), ("B", "C"), ("C", "D")]
        assert ("D", "A") not in result

    def test_multiple_cycles(self, fake_ci_estimator):
        """A -> B -> D and A -> C -> D; adding D -> A creates two cycles."""
        estimator, data = fake_ci_estimator
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C", "D"])
        dag.add_edges_from([("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")])

        result = estimator._break_cycle(
            dag,
            "D",
            "A",
            ci_test=WeakCI(data),
            data=data,
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert len(result) > 0
        assert ("D", "A") not in result
        existing_edges = {("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")}
        for edge in result:
            assert edge in existing_edges


# --- Coverage gap tests ---


def test_no_orientation_fn_raises():
    np.random.seed(0)
    # Strongly correlated data so at least one candidate edge is found
    x = np.arange(50, dtype=float)
    data = pd.DataFrame({"A": x, "B": x + 1.0})

    estimator = ExpertInLoop(
        orientation_fn=None,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        show_progress=False,
    )
    with pytest.raises(ValueError, match="No orientation function is available"):
        estimator.fit(data)


def test_orientation_fn_returns_none_blacklists_edge():
    np.random.seed(0)
    x = np.arange(50, dtype=float)
    data = pd.DataFrame({"A": x, "B": x + 1.0})

    def none_orient(var1, var2, **kwargs):
        return None

    estimator = ExpertInLoop(
        orientation_fn=none_orient,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        show_progress=False,
    )
    estimator.fit(data)

    # Edge should have been blacklisted and NOT added to the graph
    assert ("A", "B") not in estimator.causal_graph_.edges()
    assert ("B", "A") not in estimator.causal_graph_.edges()


def test_cycle_rejected_when_no_removable_edge():
    """New edge is blacklisted when _break_cycle finds no weak edge to remove."""
    np.random.seed(0)
    n = 30
    # Three perfectly correlated variables so all pairs have strong association
    x = np.arange(n, dtype=float)
    data = pd.DataFrame({"A": x, "B": x + 1.0, "C": x + 2.0})

    # Orientation function that always returns the cycle-completing direction:
    # A->B, B->C, then C->A (which would create A->B->C->A cycle)
    orient_order = [("A", "B"), ("B", "C"), ("C", "A")]
    call_count = {"n": 0}

    def cyclic_orient(var1, var2, **kwargs):
        idx = call_count["n"] % len(orient_order)
        call_count["n"] += 1
        return orient_order[idx]

    estimator = ExpertInLoop(
        orientation_fn=cyclic_orient,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        show_progress=False,
    )

    # Patch get_ci_test to return StrongCI so _break_cycle sees no weak edge
    eiul_module = sys.modules["pgmpy.causal_discovery.ExpertInLoop"]
    with patch.object(eiul_module, "get_ci_test", return_value=StrongCI(data)):
        estimator.fit(data)

    # The result must still be a valid DAG (the cycle-completing edge was rejected)
    assert nx.is_directed_acyclic_graph(estimator.causal_graph_)


def test_show_progress_logs_orientation(caplog):
    """logger.info is called when show_progress=True and an edge is oriented."""
    np.random.seed(0)
    x = np.arange(50, dtype=float)
    data = pd.DataFrame({"A": x, "B": x + 1.0})

    estimator = ExpertInLoop(
        orientation_fn=simple_orient,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        show_progress=True,
    )

    eiul_module = sys.modules["pgmpy.causal_discovery.ExpertInLoop"]
    with patch.object(eiul_module, "config") as mock_cfg:
        mock_cfg.SHOW_PROGRESS = True
        with caplog.at_level(logging.INFO, logger="pgmpy"):
            estimator.fit(data)

    orientation_logs = [r for r in caplog.records if "Queried for edge orientation" in r.message]
    assert len(orientation_logs) >= 1


def test_get_edge_orientation_expert_knowledge_orientations():
    ek = ExpertKnowledge(orientations=[("A", "B")])
    estimator = ExpertInLoop(expert_knowledge=ek)
    assert estimator._get_edge_orientation("A", "B") == ("A", "B")
    assert estimator._get_edge_orientation("B", "A") == ("A", "B")


def test_get_edge_orientation_temporal_ordering_both_directions():
    ek = ExpertKnowledge(temporal_order=[["A"], ["B"]])
    estimator = ExpertInLoop(expert_knowledge=ek)
    assert estimator._get_edge_orientation("A", "B") == ("A", "B")
    assert estimator._get_edge_orientation("B", "A") == ("A", "B")


def test_get_edge_orientation_expert_knowledge_fn():
    def ek_orient(u, v, **kwargs):
        return (v, u)

    ek = ExpertKnowledge(orientation_fn=ek_orient)
    estimator = ExpertInLoop(expert_knowledge=ek)
    assert estimator._get_edge_orientation("A", "B") == ("B", "A")


def test_fit_nonedge_empty_breaks():
    n = 20
    data = pd.DataFrame({"A": np.random.normal(size=n)})
    estimator = ExpertInLoop(orientation_fn=simple_orient)
    estimator.fit(data)
    assert estimator.causal_graph_.number_of_nodes() == 1
    assert estimator.causal_graph_.number_of_edges() == 0


def test_fit_cycle_broken_successfully():
    np.random.seed(0)
    data = pd.DataFrame({"A": [1, 2, 3, 4, 5], "B": [1, 2, 3, 4, 5], "C": [1, 2, 3, 4, 5]})
    estimator = ExpertInLoop(effect_size_threshold=0.0, pval_threshold=1.0)

    def det_orient(u, v):
        edges = {
            ("A", "B"): ("A", "B"),
            ("B", "A"): ("A", "B"),
            ("B", "C"): ("B", "C"),
            ("C", "B"): ("B", "C"),
            ("C", "A"): ("C", "A"),
            ("A", "C"): ("C", "A"),
        }
        return edges.get((u, v))

    with patch.object(estimator, "_get_edge_orientation", side_effect=det_orient):
        with patch.object(ExpertInLoop, "_break_cycle", return_value=[("A", "B")]):
            estimator.fit(data)

    assert ("A", "B") not in estimator.causal_graph_.edges()
    assert ("B", "C") in estimator.causal_graph_.edges()
    assert ("C", "A") in estimator.causal_graph_.edges()
    assert nx.is_directed_acyclic_graph(estimator.causal_graph_)


def test_required_edges_not_removed_even_if_weak():
    np.random.seed(42)
    n = 100
    data = pd.DataFrame({"A": np.random.normal(size=n), "B": np.random.normal(size=n)})

    ek = ExpertKnowledge(required_edges=[("A", "B")])
    estimator = ExpertInLoop(expert_knowledge=ek, effect_size_threshold=0.8, pval_threshold=0.0, show_progress=False)
    estimator.fit(data)

    assert ("A", "B") in estimator.causal_graph_.edges()


def test_use_cache_false_ignores_and_does_not_populate_cache():

    call_count = {"n": 0}

    def count_orient(u, v):
        call_count["n"] += 1
        return (u, v)

    ek = ExpertKnowledge(orientation_fn=count_orient)
    estimator = ExpertInLoop(expert_knowledge=ek, use_cache=False)

    # Pre-populate cache with something that would normally be returned
    estimator.orientation_cache_ = {("X", "Y")}

    # 1. Call for something IN cache - should still call orientation_fn
    res1 = estimator._get_edge_orientation("X", "Y")
    assert res1 == ("X", "Y")
    assert call_count["n"] == 1

    # 2. Call for something NOT in cache - should call orientation_fn and NOT add to cache
    res2 = estimator._get_edge_orientation("A", "B")
    assert res2 == ("A", "B")
    assert call_count["n"] == 2
    assert ("A", "B") not in estimator.orientation_cache_


def test_orientation_from_expertknowledge_orientations_with_temporal_override():
    ek = ExpertKnowledge(orientations=[("B", "A")], temporal_order=[["A"], ["B"]])
    est = ExpertInLoop(expert_knowledge=ek)
    # Explicit says B->A, temporal says A before B.
    assert est._get_edge_orientation("A", "B") == ("B", "A")
    assert est._get_edge_orientation("B", "A") == ("B", "A")


def test_orientation_from_ctor_orientations_with_temporal_override():
    ek = ExpertKnowledge(temporal_order=[["X"], ["Y"]])
    est = ExpertInLoop(expert_knowledge=ek, orientations={("Y", "X")})
    # Explicit says Y->X, temporal says X before Y.
    assert est._get_edge_orientation("X", "Y") == ("Y", "X")


def test_orientation_fn_llm_requires_descriptions():
    # Local import to avoid altering top-level imports
    from pgmpy.utils import llm_pairwise_orient

    ek = ExpertKnowledge(temporal_order=[["A"], ["B"]])
    est = ExpertInLoop(expert_knowledge=ek, orientation_fn=llm_pairwise_orient)
    with pytest.raises(ValueError, match="LLM orientation requires variable descriptions"):
        est._get_edge_orientation("A", "B")


def test_orientation_fn_with_cache_and_use_cache_true():
    calls = {"n": 0}

    def count_orient(u, v):
        calls["n"] += 1
        return (u, v)

    ek = ExpertKnowledge(orientation_fn=count_orient)
    est = ExpertInLoop(expert_knowledge=ek, use_cache=True)
    r1 = est._get_edge_orientation("A", "B")
    r2 = est._get_edge_orientation("A", "B")
    assert r1 == ("A", "B") and r2 == ("A", "B")
    # first call hits fn, second should be served from cache
    assert calls["n"] == 1


def test_temporal_fallback_when_no_other_orientation_available():
    ek = ExpertKnowledge(temporal_order=[["A"], ["B"]])
    est = ExpertInLoop(expert_knowledge=ek)  # no fn, no orientations
    assert est._get_edge_orientation("A", "B") == ("A", "B")
    assert est._get_edge_orientation("B", "A") == ("A", "B")


def test_required_edges_protection_and_blacklist_skip_in_fit():
    np.random.seed(0)
    n = 200
    # Make A->B correlated, C independent
    A = np.random.randn(n)
    B = A + 0.05 * np.random.randn(n)
    C = np.random.randn(n)
    data = pd.DataFrame({"A": A, "B": B, "C": C})

    # Required A->B must persist; forbid B->C to test blacklist skip
    ek = ExpertKnowledge(required_edges=[("A", "B")], forbidden_edges=[("B", "C")])

    def orient(u, v):
        return (u, v)

    estimator = ExpertInLoop(
        expert_knowledge=ek,
        orientation_fn=orient,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        show_progress=False,
        max_iter=10,
    )
    estimator.fit(data)

    # Required edge must remain
    assert ("A", "B") in estimator.causal_graph_.edges()
    # Forbidden should not appear in either direction
    assert ("B", "C") not in estimator.causal_graph_.edges()
    assert ("C", "B") not in estimator.causal_graph_.edges()

    # adjacency_matrix_ shape and dtype sanity (via to_numpy_array)
    assert estimator.adjacency_matrix_.shape == (3, 3)
    assert estimator.adjacency_matrix_.values.dtype.kind in ("i", "b")


def test_blacklist_filter_no_false_positive_edges():
    """Blacklisted pairs should be matched by exact tuples only, not per-endpoint membership."""
    import numpy as np
    import pandas as pd

    from pgmpy.causal_discovery.ExpertInLoop import ExpertInLoop
    from pgmpy.estimators import ExpertKnowledge

    np.random.seed(0)
    n = 200
    # Construct data with correlation only between A and D
    A = np.random.randn(n)
    D = A + 0.05 * np.random.randn(n)
    B = np.random.randn(n)
    C = np.random.randn(n)
    data = pd.DataFrame({"A": A, "B": B, "C": C, "D": D})

    # Blacklist (A,B) and (C,D) only
    ek = ExpertKnowledge(forbidden_edges=[("A", "B"), ("C", "D")])

    def orient(u, v):
        return (u, v)

    est = ExpertInLoop(
        expert_knowledge=ek,
        orientation_fn=orient,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        show_progress=False,
        max_iter=10,
    )
    est.fit(data)
    # Edge A->D should be learnable and present (not blocked by blacklist of (A,B) or (C,D))
    assert ("A", "D") in est.causal_graph_.edges() or ("D", "A") in est.causal_graph_.edges()


def test_handles_empty_nonedge_effects_safely():
    """When all significant non-edges are blacklisted, selection should not crash."""
    import numpy as np
    import pandas as pd

    from pgmpy.causal_discovery.ExpertInLoop import ExpertInLoop
    from pgmpy.estimators import ExpertKnowledge

    np.random.seed(0)
    n = 100
    A = np.random.randn(n)
    B = A + 0.01 * np.random.randn(n)
    data = pd.DataFrame({"A": A, "B": B})

    # Blacklist both directions for the only correlated pair
    ek = ExpertKnowledge(forbidden_edges=[("A", "B"), ("B", "A")])

    def orient(u, v):
        return (u, v)

    est = ExpertInLoop(
        expert_knowledge=ek,
        orientation_fn=orient,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        show_progress=False,
        max_iter=5,
    )
    # Should not raise; graph may end up with 0 edges due to blacklist
    est.fit(data)
    assert ("A", "B") not in est.causal_graph_.edges()
    assert ("B", "A") not in est.causal_graph_.edges()


def test_orientation_fn_partial_llm():
    """Test that orientation_fn as a partial(mock_llm, ...) works correctly and merges keywords."""
    from unittest.mock import MagicMock

    np.random.seed(42)
    # Use highly correlated data to ensure an edge is found
    A = np.random.normal(size=100)
    B = A + 0.1 * np.random.normal(size=100)
    data = pd.DataFrame({"A": A, "B": B})
    descriptions = {"A": "Var A", "B": "Var B"}

    # Create a mock that simulates the llm_pairwise_orient function
    mock_llm = MagicMock()
    mock_llm.__name__ = "llm_pairwise_orient"
    mock_llm.return_value = ("A", "B")

    # Create a partial wrapping the mock
    partial_orient = partial(mock_llm, some_arg="test")

    estimator = ExpertInLoop(
        orientation_fn=partial_orient,
        effect_size_threshold=0.01,
        pval_threshold=0.1,
        ci_test="pearsonr",
        show_progress=False,
    )
    estimator.descriptions = descriptions
    estimator.fit(data)

    # Verify that mock_llm was correctly identified by name and called with merged keywords
    assert mock_llm.called
    _, kwargs = mock_llm.call_args
    assert kwargs["descriptions"] == descriptions
    assert kwargs["some_arg"] == "test"


def test_get_edge_orientation_orientations_ctor():
    est = ExpertInLoop(orientations={("A", "B")})
    assert est._get_edge_orientation("A", "B") == ("A", "B")
    assert est._get_edge_orientation("B", "A") == ("A", "B")
    ek = ExpertKnowledge(temporal_order=[["B"], ["A"]])
    est = ExpertInLoop(expert_knowledge=ek, orientations={("A", "B")})
    # orientations take precedence over ek temporal_order
    assert est._get_edge_orientation("A", "B") == ("A", "B")


def test_get_edge_orientation_temporal_tie():
    ek = ExpertKnowledge(temporal_order=[["A", "B"]])
    est = ExpertInLoop(expert_knowledge=ek)
    assert est._get_edge_orientation("A", "B") is None


def test_fit_merge_orientations_list():
    ek = ExpertKnowledge(orientations=[("A", "B")])
    est = ExpertInLoop(expert_knowledge=ek, orientations={("C", "D")})
    data = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [1, 2], "D": [1, 2]})
    est.fit(data)
    assert set(est.expert_knowledge_.orientations) == {("A", "B"), ("C", "D")}


def test_fit_edge_removal():
    np.random.seed(42)
    data = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)})

    est = ExpertInLoop(orientation_fn=simple_orient, pval_threshold=0.05, effect_size_threshold=0.05)
    est.fit(data)
    mock_effects = pd.DataFrame(
        [["A", "B", [], True, 0.01, 0.9]],  # Edge present but weak
        columns=["u", "v", "z", "edge_present", "effect", "p_val"],
    )

    with patch.object(est, "_test_all", return_value=mock_effects):

        def side_effect(*args, **kwargs):
            if est.n_iter_ == 1:
                return pd.DataFrame(
                    [["A", "B", [], False, 0.8, 0.001]],
                    columns=["u", "v", "z", "edge_present", "effect", "p_val"],
                )
            else:
                return pd.DataFrame(
                    [["A", "B", [], True, 0.01, 0.9]],
                    columns=["u", "v", "z", "edge_present", "effect", "p_val"],
                )

        est = ExpertInLoop(orientation_fn=simple_orient, max_iter=2)
        with patch.object(est, "_test_all", side_effect=side_effect):
            est.fit(data)
    assert ("A", "B") not in est.causal_graph_.edges()


def test_fit_only_removals_iteration():
    np.random.seed(42)
    data = pd.DataFrame({"A": [1, 2], "B": [1, 2]})

    def side_effect(*args, **kwargs):
        if est.n_iter_ == 1:
            return pd.DataFrame(
                [["A", "B", [], False, 0.8, 0.001]],
                columns=["u", "v", "z", "edge_present", "effect", "p_val"],
            )
        elif est.n_iter_ == 2:
            return pd.DataFrame(
                [["A", "B", [], True, 0.01, 0.9]],
                columns=["u", "v", "z", "edge_present", "effect", "p_val"],
            )
        else:
            return pd.DataFrame([], columns=["u", "v", "z", "edge_present", "effect", "p_val"])

    est = ExpertInLoop(orientation_fn=simple_orient, max_iter=3)
    with patch.object(est, "_test_all", side_effect=side_effect):
        est.fit(data)
    assert est.n_iter_ == 3


def test_max_iter_warning(caplog):
    np.random.seed(0)
    n = 100
    data = pd.DataFrame(
        {
            "A": np.random.randn(n),
            "B": np.random.randn(n) + 0.9 * np.random.randn(n),
            "C": np.random.randn(n) + 0.9 * np.random.randn(n),
            "D": np.random.randn(n) + 0.9 * np.random.randn(n),
        }
    )

    def always_orient(u, v):
        return (u, v)

    estimator = ExpertInLoop(
        orientation_fn=always_orient,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        max_iter=2,
        show_progress=True,
    )

    with caplog.at_level(logging.WARNING, logger="pgmpy"):
        estimator.fit(data)

    assert any("stopped after reaching max_iter" in record.message for record in caplog.records)


def test_cache_cleared_between_fits():
    np.random.seed(0)
    data1 = pd.DataFrame({"A": np.random.randn(100), "B": np.random.randn(100)})
    data2 = pd.DataFrame({"X": np.random.randn(100), "Y": np.random.randn(100)})

    call_count = {"n": 0}

    def counting_orient(u, v):
        call_count["n"] += 1
        return (u, v)

    estimator = ExpertInLoop(
        orientation_fn=counting_orient,
        use_cache=True,
        effect_size_threshold=0.0,
        pval_threshold=1.0,
        show_progress=False,
    )

    class AlwaysSignificantCI:
        def run_test(self, X, Y, Z):
            return (0.5, 0.001)

    eiul_module = sys.modules["pgmpy.causal_discovery.ExpertInLoop"]
    with patch.object(eiul_module, "get_ci_test", return_value=AlwaysSignificantCI()):
        estimator.fit(data1)
        first_call_count = call_count["n"]
        assert first_call_count > 0

        estimator.fit(data2)
        second_call_count = call_count["n"]

        assert second_call_count > first_call_count
