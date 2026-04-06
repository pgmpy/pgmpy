"""
Tests for the sklearn-compatible GES class in pgmpy.causal_discovery.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import PDAG
from pgmpy.causal_discovery import GES
from pgmpy.structure_score import K2


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [GES(return_type="dag")],
    expected_failed_checks=expected_failed_checks,
)
def test_ges_compatibility(estimator, check):
    check(estimator)


@pytest.fixture
def rand_data():
    data = pd.DataFrame(
        np.random.randint(0, 5, size=(int(1e4), 2)),
        columns=list("AB"),
        dtype="category",
    )
    data["C"] = data["B"]
    return data


@pytest.fixture
def titanic_data():
    return pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")


@pytest.fixture
def titanic_data_mixed(titanic_data):
    # dropna needed because Age and Embarked columns have missing values
    return titanic_data[["Survived", "Sex", "Pclass", "Age", "Embarked"]].dropna()


@pytest.fixture
def titanic_data_categorical(titanic_data):
    return titanic_data[["Survived", "Sex", "Pclass"]].astype("category")


# Optional manual parity check against `causal-learn==0.1.4.5`.
# This is intentionally kept as documentation instead of an executable test,
# because `causal-learn` is an optional dependency.
#
# Reproduction:
#   python - <<'PY'
#   import numpy as np
#   import pandas as pd
#   from causallearn.search.ScoreBased.GES import ges
#   from pgmpy.causal_discovery import GES
#   from pgmpy.structure_score import BDeu
#
#   def causallearn_cpdag_edge_sets(graph, columns):
#       directed = set()
#       undirected = set()
#       adjacency = graph.graph
#       for i in range(len(columns)):
#           for j in range(i + 1, len(columns)):
#               a, b = adjacency[i, j], adjacency[j, i]
#               if a == -1 and b == -1:
#                   undirected.add(tuple(sorted((columns[i], columns[j]))))
#               elif a == -1 and b == 1:
#                   directed.add((columns[i], columns[j]))
#               elif a == 1 and b == -1:
#                   directed.add((columns[j], columns[i]))
#       return directed, undirected
#
#   def pgmpy_cpdag_edge_sets(pdag):
#       directed = set(pdag.directed_edges)
#       undirected = {tuple(sorted(edge)) for edge in pdag.undirected_edges}
#       return directed, undirected
#
#   def run_case(name, data, columns):
#       df = pd.DataFrame(
#           {col: pd.Series(data[:, i], dtype="category") for i, col in enumerate(columns)}
#       )
#       causal_graph = ges(data, score_func="local_score_BDeu", node_names=columns)["G"]
#       pgmpy_graph = GES(
#           scoring_method=BDeu(df, equivalent_sample_size=1),
#           return_type="pdag",
#       ).fit(df).causal_graph_
#       print(name, causallearn_cpdag_edge_sets(causal_graph, columns), pgmpy_cpdag_edge_sets(pgmpy_graph))
#
#   columns = ["A", "B", "C"]
#
#   rng = np.random.default_rng(0)
#   n = 5000
#   a = rng.binomial(1, 0.5, size=n)
#   b = np.where(rng.random(n) < 0.95, a, 1 - a)
#   c = np.where(rng.random(n) < 0.95, b, 1 - b)
#   run_case("chain", np.column_stack([a, b, c]).astype(int), columns)
#
#   rng = np.random.default_rng(1)
#   a = rng.binomial(1, 0.5, size=n)
#   b = rng.binomial(1, 0.5, size=n)
#   base = np.logical_or(a, b).astype(int)
#   noise = rng.binomial(1, 0.03, size=n)
#   c = np.logical_xor(base, noise).astype(int)
#   run_case("collider", np.column_stack([a, b, c]).astype(int), columns)
#   PY
#
# Observed results:
#   chain:
#       causal-learn -> (set(), {("A", "B"), ("B", "C")})
#       pgmpy       -> (set(), {("A", "B"), ("B", "C")})
#   collider:
#       causal-learn -> ({("A", "C"), ("B", "C")}, set())
#       pgmpy       -> ({("A", "C"), ("B", "C")}, set())


def test_insert_orients_t_away_from_v():
    est = GES()

    pdag = PDAG(undirected_ebunch=[("B", "C")])
    pdag.add_nodes_from(["A", "B", "C"])

    new_model = est.insert("A", "B", {"C"}, pdag)

    assert new_model.directed_edges == {("A", "B"), ("B", "C")}
    assert new_model.undirected_edges == set()


def test_legal_edge_deletions_include_both_orders_for_undirected_edges():
    est = GES()

    pdag = PDAG(undirected_ebunch=[("A", "B")])
    pdag.add_nodes_from(["A", "B"])

    assert set(est._legal_edge_deletions(pdag)) == {("A", "B"), ("B", "A")}


class TestGESCore:
    """Tests for core GES functionality."""

    def test_estimate_rand(self, rand_data):
        est = GES(scoring_method="k2", return_type="dag")
        est.fit(rand_data)
        assert set(est.causal_graph_.nodes()) == {"A", "B", "C"}
        assert list(est.causal_graph_.edges()) == [("B", "C")] or list(est.causal_graph_.edges()) == [("C", "B")]

    def test_estimate_rand_with_structure_score_instance(self, rand_data):
        est = GES(scoring_method=K2(rand_data), return_type="dag")
        est.fit(rand_data)
        assert set(est.causal_graph_.nodes()) == {"A", "B", "C"}
        assert list(est.causal_graph_.edges()) == [("B", "C")] or list(est.causal_graph_.edges()) == [("C", "B")]

    def test_estimate_titanic(self, titanic_data_categorical):
        est = GES(scoring_method="k2", return_type="dag")
        est.fit(titanic_data_categorical)
        assert len(est.causal_graph_.edges()) > 0

    def test_return_type_pdag(self, rand_data):
        est = GES(scoring_method="k2", return_type="pdag")
        est.fit(rand_data)
        assert est.causal_graph_ is not None
        assert est.adjacency_matrix_ is not None

    def test_return_type_dag(self, rand_data):
        est = GES(scoring_method="k2", return_type="dag")
        est.fit(rand_data)
        assert est.causal_graph_ is not None
        assert est.adjacency_matrix_ is not None

    def test_adjacency_matrix(self, rand_data):
        est = GES(scoring_method="k2", return_type="dag")
        est.fit(rand_data)
        assert est.adjacency_matrix_ is not None
        assert est.adjacency_matrix_.shape[0] == len(rand_data.columns)
        assert est.adjacency_matrix_.shape[1] == len(rand_data.columns)

    def test_feature_names(self, rand_data):
        est = GES(scoring_method="k2", return_type="dag")
        est.fit(rand_data)
        assert hasattr(est, "n_features_in_")
        assert hasattr(est, "feature_names_in_")


def test_expert_knowledge_not_supported():
    with pytest.raises(TypeError, match="expert_knowledge"):
        GES(expert_knowledge=None)
