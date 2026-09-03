"""
Tests for the sklearn-compatible ExhaustiveSearch class in pgmpy.causal_discovery.
"""

import networkx as nx
import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.exceptions import NotFittedError

from pgmpy.causal_discovery import ExhaustiveSearch
from pgmpy.structure_score import K2, BDeu

# Unlike HillClimbSearch, ExhaustiveSearch cannot go through sklearn's full
# `parametrize_with_checks` battery: several generic checks (e.g.
# `check_dtype_object`) fit on synthetic data with 10 features, and searching
# `2**(10*9)` DAGs is computationally infeasible -- exhaustive search is only
# usable for ~6 variables or fewer by design. We instead exercise the handful
# of sklearn-estimator conventions that don't depend on dataset size.


def test_sklearn_estimator_conventions():
    est = ExhaustiveSearch(scoring_method="k2", return_type="dag")

    params = est.get_params()
    est_clone = clone(est)
    assert est_clone.get_params() == params
    assert not hasattr(est_clone, "causal_graph_")

    est.set_params(**params)
    assert est.get_params() == params

    with pytest.raises(NotFittedError):
        est.all_scores()
    with pytest.raises(NotFittedError):
        est.score(X=pd.DataFrame({"A": [1, 2], "B": [1, 2]}))


@pytest.fixture
def rand_data():
    data = pd.DataFrame(
        np.random.randint(0, 5, size=(5000, 2)),
        columns=list("AB"),
        dtype="category",
    )
    data["C"] = data["B"]
    return data


@pytest.fixture
def titanic_data():
    return pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")


@pytest.fixture
def titanic_data2(titanic_data):
    return titanic_data[["Survived", "Sex", "Pclass"]].astype("category")


@pytest.fixture
def est_rand():
    est = ExhaustiveSearch()
    est.variables_ = ["A", "B", "C"]
    return est


def test_all_dags(est_rand):
    assert len(list(est_rand.all_dags(["A", "B", "C", "D"]))) == 543

    abc_dags = set(map(tuple, [sorted(dag.edges()) for dag in est_rand.all_dags()]))
    abc_dags_ref = {
        (("A", "B"), ("C", "A"), ("C", "B")),
        (("A", "C"), ("B", "C")),
        (("B", "A"), ("B", "C")),
        (("C", "B"),),
        (("A", "C"), ("B", "A")),
        (("B", "C"), ("C", "A")),
        (("A", "B"), ("B", "C")),
        (("A", "C"), ("B", "A"), ("B", "C")),
        (("A", "B"),),
        (("A", "B"), ("C", "A")),
        (("B", "A"), ("C", "A"), ("C", "B")),
        (("A", "C"), ("C", "B")),
        (("A", "B"), ("A", "C"), ("C", "B")),
        (("B", "A"), ("C", "B")),
        (("A", "B"), ("A", "C")),
        (("C", "A"), ("C", "B")),
        (("A", "B"), ("A", "C"), ("B", "C")),
        (("C", "A"),),
        (("B", "A"), ("B", "C"), ("C", "A")),
        (("B", "A"),),
        (("A", "B"), ("C", "B")),
        (),
        (("B", "A"), ("C", "A")),
        (("A", "C"),),
        (("B", "C"),),
    }
    assert abc_dags == abc_dags_ref


def test_estimate_rand(rand_data):
    est_k2 = ExhaustiveSearch(scoring_method=K2(rand_data), return_type="dag")
    est_k2.fit(rand_data)
    assert set(est_k2.causal_graph_.nodes()) == {"A", "B", "C"}
    assert set(est_k2.causal_graph_.edges()) == {("B", "C")}

    est_bdeu = ExhaustiveSearch(scoring_method=BDeu(rand_data), return_type="dag")
    est_bdeu.fit(rand_data)
    assert set(est_bdeu.causal_graph_.edges()) == {("B", "C")}


def test_estimate_titanic(titanic_data2):
    est_k2 = ExhaustiveSearch(scoring_method=K2(titanic_data2), return_type="dag")
    est_k2.fit(titanic_data2)
    assert set(est_k2.causal_graph_.edges()) == {
        ("Survived", "Pclass"),
        ("Sex", "Pclass"),
        ("Sex", "Survived"),
    }


def test_all_scores(titanic_data2):
    est_k2 = ExhaustiveSearch(scoring_method=K2(titanic_data2))
    est_k2.fit(titanic_data2)
    scores = est_k2.all_scores()

    # Sorted by score, ascending; the best (highest-scoring) DAG is last.
    assert scores[-1][0] == pytest.approx(max(score for score, _ in scores))
    assert sorted(scores[-1][1].edges()) == sorted(est_k2.causal_graph_.edges())
    assert len(scores) == len(list(est_k2.all_dags()))


def test_estimate_rand_bic_default(rand_data):
    est_bic = ExhaustiveSearch(return_type="dag")
    est_bic.fit(rand_data)
    assert set(est_bic.causal_graph_.nodes()) == {"A", "B", "C"}
    assert set(est_bic.causal_graph_.edges()) == {("B", "C")}
    assert nx.is_directed_acyclic_graph(est_bic.causal_graph_)


def test_return_type_pdag(rand_data):
    est = ExhaustiveSearch(return_type="pdag")
    est.fit(rand_data)
    assert set(est.causal_graph_.nodes()) == {"A", "B", "C"}


def test_return_type_invalid(rand_data):
    est = ExhaustiveSearch(return_type="not-a-type")
    with pytest.raises(ValueError, match="return_type must be one of"):
        est.fit(rand_data)
