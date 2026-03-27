"""
Tests for the sklearn-compatible ExpertInLoop class in pgmpy.causal_discovery
"""

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
        show_progress=False,
    )
    estimator.fit(adult_data)

    # Check forbidden edges
    assert ("Age", "Income") not in estimator.causal_graph_.edges()

    # Check temporal order
    for u, v in estimator.causal_graph_.edges():
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
