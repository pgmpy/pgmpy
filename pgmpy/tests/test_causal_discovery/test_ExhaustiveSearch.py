from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery import ExhaustiveSearch
from pgmpy.estimators import ExhaustiveSearch as LegacyExhaustiveSearch
from pgmpy.structure_score import BDeu

# sklearn checks that fit on five or more columns, which an exhaustive search cannot finish in reasonable time.
TOO_WIDE_CHECKS = {"check_estimators_dtypes", "check_dtype_object"}


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [ExhaustiveSearch(return_type="dag", show_progress=False)],
    expected_failed_checks=expected_failed_checks,
)
def test_exhaustive_search_compatibility(estimator, check):
    if check.func.__name__ in TOO_WIDE_CHECKS:
        pytest.skip("Fits on five or more columns.")
    check(estimator)


@pytest.fixture
def rand_data():
    data = pd.DataFrame(
        np.random.default_rng(42).integers(0, 5, size=(5000, 2)),
        columns=list("AB"),
        dtype="category",
    )
    data["C"] = data["B"]
    return data


@pytest.fixture
def titanic_data():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")
    return data[["Survived", "Sex", "Pclass"]].astype("category")


def test_all_dags():
    nodes = list("ABCD")
    assert [len(list(ExhaustiveSearch._all_dags(nodes[:n]))) for n in range(1, 5)] == [1, 3, 25, 543]

    dags = list(ExhaustiveSearch._all_dags(nodes[:3]))
    assert all(nx.is_directed_acyclic_graph(dag) and set(dag.nodes()) == set("ABC") for dag in dags)
    n_edges = [dag.number_of_edges() for dag in dags]
    assert n_edges == sorted(n_edges)

    with pytest.warns(UserWarning, match="7 nodes"):
        next(ExhaustiveSearch._all_dags(list("ABCDEFG")))


def test_fit_titanic(titanic_data):
    est = ExhaustiveSearch(scoring_method="k2", return_type="dag", show_progress=False).fit(titanic_data)
    expected_edges = {("Survived", "Pclass"), ("Sex", "Pclass"), ("Sex", "Survived")}
    assert isinstance(est.causal_graph_, DAG)
    assert set(est.causal_graph_.nodes()) == {"Survived", "Sex", "Pclass"}
    assert set(est.causal_graph_.edges()) == expected_edges
    assert est.n_features_in_ == 3
    adjacency = est.adjacency_matrix_
    assert {(u, v) for u in adjacency.index for v in adjacency.columns if adjacency.loc[u, v]} == expected_edges

    # With the default score, the scores of the complete DAGs differ only by round-off, so check their PDAG.
    pdag = ExhaustiveSearch(show_progress=False).fit(titanic_data).causal_graph_
    assert pdag.directed_edges == set()
    assert {frozenset(edge) for edge in pdag.undirected_edges} == {
        frozenset(pair) for pair in combinations(titanic_data.columns, 2)
    }


@pytest.mark.parametrize("scoring_method", [None, "k2", "bdeu", "bic-d"])
def test_fit_rand(rand_data, scoring_method):
    dag = ExhaustiveSearch(scoring_method=scoring_method, return_type="dag", show_progress=False).fit(rand_data)
    assert set(dag.causal_graph_.nodes()) == {"A", "B", "C"}
    assert set(dag.causal_graph_.edges()) == {("B", "C")}

    pdag = ExhaustiveSearch(scoring_method=scoring_method, show_progress=False).fit(rand_data).causal_graph_
    assert isinstance(pdag, PDAG)
    assert set(pdag.nodes()) == {"A", "B", "C"}
    assert pdag.directed_edges == set()
    assert {frozenset(edge) for edge in pdag.undirected_edges} == {frozenset({"B", "C"})}


def test_dag_scores(titanic_data):
    est = ExhaustiveSearch(scoring_method="k2", return_type="dag", show_progress=False).fit(titanic_data)
    scores = [score for score, _ in est.dag_scores_]
    assert len(est.dag_scores_) == 25
    assert scores == sorted(scores, reverse=True)
    assert scores[0] == pytest.approx(-1885.884573316297)
    assert est.dag_scores_[0][1] == (("Sex", "Pclass"), ("Sex", "Survived"), ("Survived", "Pclass"))
    assert est.dag_scores_[0][1] == tuple(sorted(est.causal_graph_.edges()))
    assert est.dag_scores_[-1] == (pytest.approx(-2072.9132364404695), ())
    for score, edges in est.dag_scores_:
        dag = DAG()
        dag.add_nodes_from(titanic_data.columns)
        dag.add_edges_from(edges)
        assert score == pytest.approx(est.scoring_method_.score(dag))

    est.fit(titanic_data[["Survived", "Sex"]])
    assert len(est.dag_scores_) == 3
    assert set(est.causal_graph_.nodes()) == {"Survived", "Sex"}


def test_fit_continuous():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv", index_col=0)[["A", "B", "C"]]
    est = ExhaustiveSearch(show_progress=False).fit(data)
    assert est.get_tag("data_types") == ["continuous"]
    assert len(est.dag_scores_) == 25

    # Reference graph from causal-learn v0.1.4.5, which returns A -> C <- B:
    # bic_exact_search((data - data.mean()).to_numpy(), search_method="dp")
    assert est.causal_graph_.directed_edges == {("A", "C"), ("B", "C")}
    assert est.causal_graph_.undirected_edges == set()


def test_deprecated_estimator_matches(titanic_data):
    scoring_method = BDeu(titanic_data, equivalent_sample_size=200)
    with pytest.warns(FutureWarning, match=r"deprecated since v1\.2\.0 .* `pgmpy\.causal_discovery\.ExhaustiveSearch`"):
        legacy = LegacyExhaustiveSearch(titanic_data, scoring_method=scoring_method)
    est = ExhaustiveSearch(scoring_method=scoring_method, return_type="dag", show_progress=False).fit(titanic_data)
    assert est.scoring_method_ is scoring_method

    assert list(legacy.estimate().edges()) == list(est.causal_graph_.edges())
    legacy_scores = [(score, tuple(sorted(dag.edges()))) for score, dag in legacy.all_scores()]
    assert legacy_scores == sorted(est.dag_scores_, key=lambda dag_score: dag_score[0])


def test_invalid_return_type(rand_data):
    with pytest.raises(ValueError, match="return_type must be one of: dag, pdag"):
        ExhaustiveSearch(return_type="cpdag", show_progress=False).fit(rand_data)
