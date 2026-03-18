import networkx as nx
import pandas as pd
import pytest
from skbase.utils.dependencies import _check_soft_dependencies

from pgmpy.base import DAG
from pgmpy.estimators import ExpertInLoop, ExpertKnowledge


@pytest.fixture
def adult_df():
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
def estimator(adult_df):
    return ExpertInLoop(data=adult_df)


@pytest.fixture
def descriptions():
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
def estimator_small(adult_df):
    return ExpertInLoop(data=adult_df[["Age", "Education", "Race", "Sex", "Income"]])


@pytest.fixture
def orientations_small():
    return {
        ("Education", "Income"),
        ("Race", "Education"),
        ("Age", "Education"),
    }


class TestExpertInLoop:
    @pytest.mark.skipif(
        not _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_estimate(self, estimator):
        true_edges = [
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

        true_dag = nx.DiGraph(true_edges)
        true_dag.add_nodes_from(estimator.data.columns)

        def oracle_orient(var1, var2, **kwargs):
            """Orientation function that knows the 'true' structure."""
            if true_dag.has_edge(var1, var2):
                return (var1, var2)
            elif true_dag.has_edge(var2, var1):
                return (var2, var1)
            else:
                return None

        # Use the expert estimator with our oracle orientation function
        estimated_dag = estimator.estimate(
            orientation_fn=oracle_orient,
            pval_threshold=0.05,
            effect_size_threshold=0.05,
            show_progress=True,
        )

        for u, v in estimated_dag.edges():
            assert true_dag.has_edge(u, v)

        assert nx.is_directed_acyclic_graph(estimated_dag)

    @pytest.mark.skipif(
        not _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_estimate_with_custom_orient_fn(self, estimator_small):
        def custom_orient(var1, var2, **kwargs):
            # Always orient edges from alphabetically first to second
            if var1 < var2:
                return (var1, var2)
            else:
                return (var2, var1)

        dag = estimator_small.estimate(
            orientation_fn=custom_orient,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )

        # Check that all edges are oriented from alphabetically lower to higher
        for edge in dag.edges():
            assert edge[0] < edge[1]

        # Check that orientations were cached
        assert len(estimator_small.orientation_cache) > 0
        for edge in estimator_small.orientation_cache:
            assert edge[0] < edge[1]

    @pytest.mark.skipif(
        not _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_estimate_with_orient_fn_kwargs(self, estimator_small):
        def orient_with_kwargs(var1, var2, **kwargs):
            # Use a keyword argument to determine orientation
            if kwargs.get("reverse_alphabetical", False):
                if var1 > var2:
                    return (var1, var2)
                else:
                    return (var2, var1)
            else:
                if var1 < var2:
                    return (var1, var2)
                else:
                    return (var2, var1)

        # Test with reverse_alphabetical=True
        dag_reverse = estimator_small.estimate(
            orientation_fn=orient_with_kwargs,
            reverse_alphabetical=True,
            pval_threshold=0.1,
            effect_size_threshold=0.1,
        )

        # Check that all edges are oriented from alphabetically higher to lower
        for edge in dag_reverse.edges():
            assert edge[0] > edge[1]

    @pytest.mark.skipif(
        not _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_combined_expert_knowledge(self, estimator):
        """Test combination of forbidden edges, required edges, and temporal order."""
        expert_knowledge = ExpertKnowledge(
            forbidden_edges=[("Age", "Income")],
            required_edges=[("Education", "Income")],
            temporal_order=[["Age", "Race"], ["Education"], ["Income", "HoursPerWeek"]],
        )

        # Run the algorithm
        dag = estimator.estimate(
            expert_knowledge=expert_knowledge,
            effect_size_threshold=0.0001,
            show_progress=False,
        )

        # Check forbidden edges
        assert ("Age", "Income") not in dag.edges()

        # Check temporal order
        for u, v in dag.edges():
            u_order = expert_knowledge.temporal_ordering[u]
            v_order = expert_knowledge.temporal_ordering[v]
            assert u_order <= v_order, f"Edge {u}->{v} violates temporal order"

    @pytest.mark.skipif(
        not _check_soft_dependencies("xgboost", severity="none"),
        reason="execute only if required dependency present",
    )
    def test_edge_orientation_priority(self, estimator):
        """Test that edge orientation follows the correct priority order."""
        expert_knowledge = ExpertKnowledge(temporal_order=[["Age", "Race"], ["Education"], ["Income", "HoursPerWeek"]])

        # Define orientations that should take precedence over temporal order
        orientations = {("Income", "Education")}  # Opposite of temporal order

        # Run the algorithm
        dag = estimator.estimate(
            expert_knowledge=expert_knowledge,
            orientations=orientations,
            effect_size_threshold=0.0001,
            show_progress=False,
        )

        # Check that specified orientations take precedence
        if ("Income", "Education") in dag.edges():
            assert ("Education", "Income") not in dag.edges()


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
    return ExpertInLoop(data=data)


@pytest.fixture
def simple_dag():
    dag = DAG()
    dag.add_nodes_from(["A", "B", "C"])
    dag.add_edges_from([("A", "B"), ("B", "C")])
    return dag


def make_weak_ci():
    """Return a mock CI test that always reports a weak (non-significant) edge."""

    def ci_test(X, Y, Z, data, boolean):
        return (0.01, 0.9)  # low effect, high p-value

    return ci_test


def make_strong_ci():
    """Return a mock CI test that always reports a strong (significant) edge."""

    def ci_test(X, Y, Z, data, boolean):
        return (0.5, 0.001)  # high effect, low p-value

    return ci_test


class TestFakeCI:
    def test_all_weak_edges_removed(self, fake_ci_estimator, simple_dag):
        result = fake_ci_estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=make_weak_ci(),
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert len(result) > 0
        for edge in result:
            assert edge in [("A", "B"), ("B", "C")]

    def test_all_strong_edges_kept(self, fake_ci_estimator, simple_dag):
        result = fake_ci_estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=make_strong_ci(),
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert result == []

    def test_selective_removal(self, fake_ci_estimator, simple_dag):
        def mock_ci_test(X, Y, Z, data, boolean):
            # A->B is weak, everything else is strong
            if {X, Y} == {"A", "B"}:
                return (0.01, 0.9)
            return (0.5, 0.001)

        result = fake_ci_estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=mock_ci_test,
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert ("B", "C") not in result
        assert ("C", "A") not in result

    def test_new_edge_never_in_result(self, fake_ci_estimator, simple_dag):
        result = fake_ci_estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=make_weak_ci(),
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert ("C", "A") not in result

    def test_original_dag_not_modified(self, fake_ci_estimator, simple_dag):
        original_edges = set(simple_dag.edges())
        original_nodes = set(simple_dag.nodes())

        fake_ci_estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=make_weak_ci(),
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert set(simple_dag.edges()) == original_edges
        assert set(simple_dag.nodes()) == original_nodes

    def test_longer_cycle(self, fake_ci_estimator):
        """Test with a 4-node cycle: A -> B -> C -> D, adding D -> A."""
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C", "D"])
        dag.add_edges_from([("A", "B"), ("B", "C"), ("C", "D")])

        result = fake_ci_estimator._break_cycle(
            dag,
            "D",
            "A",
            ci_test=make_weak_ci(),
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert len(result) > 0
        for edge in result:
            assert edge in [("A", "B"), ("B", "C"), ("C", "D")]
        assert ("D", "A") not in result

    def test_multiple_cycles(self, fake_ci_estimator):
        """A -> B -> D and A -> C -> D; adding D -> A creates two cycles."""
        dag = DAG()
        dag.add_nodes_from(["A", "B", "C", "D"])
        dag.add_edges_from([("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")])

        result = fake_ci_estimator._break_cycle(
            dag,
            "D",
            "A",
            ci_test=make_weak_ci(),
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert len(result) > 0
        assert ("D", "A") not in result
        existing_edges = {("A", "B"), ("B", "D"), ("A", "C"), ("C", "D")}
        for edge in result:
            assert edge in existing_edges

    def test_conditioning_set(self, fake_ci_estimator, simple_dag):
        """The CI test must be called with Z = cycle_nodes - {X, Y}."""
        calls = []

        def recording_ci_test(X, Y, Z, data, boolean):
            calls.append((X, Y, set(Z)))
            return (0.5, 0.001)  # strong – keeps all edges

        fake_ci_estimator._break_cycle(
            simple_dag,
            "C",
            "A",
            ci_test=recording_ci_test,
            effect_size_threshold=0.05,
            pval_threshold=0.05,
        )

        assert len(calls) > 0
        for X, Y, Z in calls:
            assert X not in Z
            assert Y not in Z
            assert Z.issubset({"A", "B", "C"})
