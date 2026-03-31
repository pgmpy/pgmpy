"""
Tests for the sklearn-compatible ExpertInLoop class in pgmpy.causal_discovery
"""

import logging
import sys
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
    """Subset of adult data for faster tests (equivalent to self.estimator_small.data in legacy)."""
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


# --- Tests ---


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
    # Create estimator and set the orientation cache
    estimator = ExpertInLoop(
        orientation_fn=simple_orient,
        use_cache=True,
        pval_threshold=0.1,
        effect_size_threshold=0.1,
        show_progress=False,
    )
    # Pre-populate the orientation cache
    estimator.orientation_cache_ = orientations_small

    estimator.fit(adult_data_small)

    assert orientations_small == set(estimator.causal_graph_.edges())
    # Cache should still contain the orientations
    assert estimator.orientation_cache_ == orientations_small


@pytest.mark.skipif(
    not _check_soft_dependencies("xgboost", severity="none"),
    reason="execute only if required dependency present",
)
def test_estimate_with_custom_orient_fn(adult_data_small):
    """Test estimation with custom orientation function."""

    def custom_orient(var1, var2, **kwargs):
        # Always orient edges from alphabetically first to second
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
    """ValueError is raised during fitting when no orientation function is configured.

    Covers ExpertInLoop._get_edge_orientation lines 319-323: the ``orient_fn is None``
    branch that now raises instead of silently returning None.
    """
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
    """When orientation_fn returns None the candidate edge is blacklisted, not added.

    Covers ExpertInLoop._fit lines 456-461: the ``edge_direction is None`` branch.
    """
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
    """New edge is blacklisted when _break_cycle finds no weak edge to remove.

    Covers ExpertInLoop._fit lines 478-480: the ``len(edges_to_remove) == 0`` branch.
    Uses a cyclic orientation function + StrongCI (all edges look strong so
    _break_cycle returns []) to trigger the rejection path naturally.
    """
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
    """logger.info is called when show_progress=True and an edge is oriented.

    Covers ExpertInLoop._get_edge_orientation lines 327-330.
    """
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
    """ExpertKnowledge orientations are correctly used in _get_edge_orientation.
    Covers ExpertInLoop.py line 280, 284-287.
    """
    ek = ExpertKnowledge(orientations=[("A", "B")])
    estimator = ExpertInLoop(expert_knowledge=ek)
    # Forward hit (line 285)
    assert estimator._get_edge_orientation("A", "B") == ("A", "B")
    # Reversed hit (line 287)
    assert estimator._get_edge_orientation("B", "A") == ("A", "B")


def test_get_edge_orientation_temporal_ordering_both_directions():
    """temporal_ordering covers both u < v and v < u directions.
    Covers ExpertInLoop.py lines 290-300.
    """
    ek = ExpertKnowledge(temporal_order=[["A"], ["B"]])
    estimator = ExpertInLoop(expert_knowledge=ek)
    # Forward: A < B (line 298)
    assert estimator._get_edge_orientation("A", "B") == ("A", "B")
    # Reversed: B > A (line 300)
    assert estimator._get_edge_orientation("B", "A") == ("A", "B")


def test_get_edge_orientation_expert_knowledge_fn():
    """ExpertKnowledge orientation_fn is correctly used in _get_edge_orientation.
    Covers ExpertInLoop.py line 315.
    """

    def ek_orient(u, v, **kwargs):
        return (v, u)

    ek = ExpertKnowledge(orientation_fn=ek_orient)
    estimator = ExpertInLoop(expert_knowledge=ek)
    # Orientation from ExpertKnowledge.orientation_fn should be prioritized
    assert estimator._get_edge_orientation("A", "B") == ("B", "A")


def test_fit_nonedge_empty_breaks():
    """_fit breaks when nonedge_effects is empty and no removals occurred.
    Covers ExpertInLoop.py lines 445-446.
    """
    n = 20
    data = pd.DataFrame({"A": np.random.normal(size=n)})
    # ExpertInLoop on single variable will have no candidate edges
    estimator = ExpertInLoop(orientation_fn=simple_orient)
    estimator.fit(data)
    assert estimator.causal_graph_.number_of_nodes() == 1
    assert estimator.causal_graph_.number_of_edges() == 0


def test_fit_cycle_broken_successfully():
    """A cycle that is identified is broken by removing a weak edge.
    Covers ExpertInLoop.py lines 482-484.
    """
    np.random.seed(0)
    data = pd.DataFrame({"A": [1, 2], "B": [1, 2], "C": [1, 2]})
    estimator = ExpertInLoop(effect_size_threshold=0.0, pval_threshold=1.0)
    
    # We must ensure variables_ is set before fit is called if we are mocking parts of fit,
    # but estimator.fit(data) will set it. 
    
    # We patch _get_edge_orientation to return a sequence that creates A->B, B->C, then C->A (cycle)
    # The 4th return is for the re-evaluation if needed.
    with patch.object(estimator, "_get_edge_orientation", side_effect=[("A", "B"), ("B", "C"), ("C", "A"), ("C", "A")]):
        # Mock _break_cycle on the instance to return A->B
        with patch.object(estimator, "_break_cycle", return_value=[("A", "B")]):
            estimator.fit(data)

    # Final graph should have B->C and C->A, but NOT A->B.
    assert ("A", "B") not in estimator.causal_graph_.edges()
    assert ("B", "C") in estimator.causal_graph_.edges()
    assert ("C", "A") in estimator.causal_graph_.edges()
    assert nx.is_directed_acyclic_graph(estimator.causal_graph_)
