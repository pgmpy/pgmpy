"""
Tests for the sklearn-compatible HillClimbSearch class in pgmpy.causal_discovery.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.causal_discovery import HillClimbSearch
from pgmpy.estimators import K2, ExpertKnowledge
from pgmpy.example_models import load_model
from pgmpy.metrics import SHD, CorrelationScore
from pgmpy.models import DiscreteBayesianNetwork


def expected_failed_checks(estimator):
    return {
        "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
        "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
    }


@parametrize_with_checks(
    [HillClimbSearch(return_type="dag", show_progress=False)],
    expected_failed_checks=expected_failed_checks,
)
def test_hillclimb_compatibility(estimator, check):
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
def titanic_data1(titanic_data):
    return titanic_data[["Survived", "Sex", "Pclass", "Age", "Embarked"]].dropna()


@pytest.fixture
def titanic_data2(titanic_data):
    return titanic_data[["Survived", "Sex", "Pclass"]].astype("category")


@pytest.fixture
def est_rand(rand_data):
    est = HillClimbSearch()
    est.variables_ = list(rand_data.columns)
    return est


@pytest.fixture
def est_titanic1(titanic_data1):
    est = HillClimbSearch()
    est.variables_ = list(titanic_data1.columns)
    return est


@pytest.fixture
def score_rand(rand_data):
    k2score = K2(rand_data)
    return k2score.local_score


@pytest.fixture
def score_structure_prior(rand_data):
    k2score = K2(rand_data)
    return k2score.structure_prior_ratio


@pytest.fixture
def score_titanic1(titanic_data1):
    return K2(titanic_data1).local_score


@pytest.fixture
def model1():
    model = DiscreteBayesianNetwork()
    model.add_nodes_from(["A", "B", "C"])
    return model


@pytest.fixture
def model2(model1):
    model = model1.copy()
    model.add_edge("A", "B")
    return model


def test_legal_operations(est_rand, model2, score_rand, score_structure_prior):
    model2_legal_ops = list(
        est_rand._legal_operations_dag(
            model=model2,
            score=score_rand,
            structure_score=score_structure_prior,
            tabu_list=set(),
            max_indegree=float("inf"),
            required_edges=set(),
            forbidden_edges=set(),
        )
    )
    model2_legal_ops_ref = [
        (("+", ("C", "A")), -28.15602208305154),
        (("+", ("A", "C")), -28.155467430966382),
        (("+", ("C", "B")), 7636.947544933631),
        (("+", ("B", "C")), 7937.805375579936),
        (("-", ("A", "B")), 28.155467430966382),
        (("flip", ("A", "B")), -0.0005546520851567038),
    ]
    assert {op for op, score in model2_legal_ops} == {op for op, score in model2_legal_ops_ref}


def test_legal_operations_forbidden_required(est_rand, model2, score_rand, score_structure_prior):
    model2_legal_ops_bl = list(
        est_rand._legal_operations_dag(
            model=model2,
            score=score_rand,
            structure_score=score_structure_prior,
            tabu_list=set(),
            max_indegree=float("inf"),
            forbidden_edges={("A", "B"), ("A", "C"), ("C", "A"), ("C", "B")},
            required_edges=set(),
        )
    )
    model2_legal_ops_bl_ref = [
        ("+", ("B", "C")),
        ("-", ("A", "B")),
        ("flip", ("A", "B")),
    ]
    assert {op for op, score in model2_legal_ops_bl} == set(model2_legal_ops_bl_ref)

    model2_legal_ops_wl = list(
        est_rand._legal_operations_dag(
            model=model2,
            score=score_rand,
            structure_score=score_structure_prior,
            tabu_list=set(),
            max_indegree=float("inf"),
            forbidden_edges={("B", "C"), ("C", "B"), ("B", "A")},
            required_edges=set(),
        )
    )
    model2_legal_ops_wl_ref = [
        ("+", ("A", "C")),
        ("+", ("C", "A")),
        ("-", ("A", "B")),
    ]
    assert {op for op, score in model2_legal_ops_wl} == set(model2_legal_ops_wl_ref)


def test_legal_operations_titanic(est_titanic1, score_titanic1, score_structure_prior):
    start_model = DiscreteBayesianNetwork([("Survived", "Sex"), ("Pclass", "Age"), ("Pclass", "Embarked")])

    legal_ops = est_titanic1._legal_operations_dag(
        model=start_model,
        score=score_titanic1,
        structure_score=score_structure_prior,
        tabu_list=[],
        max_indegree=float("inf"),
        forbidden_edges=set(),
        required_edges=set(),
    )
    assert len(list(legal_ops)) == 20

    tabu_list = [
        ("-", ("Survived", "Sex")),
        ("-", ("Survived", "Pclass")),
        ("flip", ("Age", "Pclass")),
    ]
    legal_ops_tabu = est_titanic1._legal_operations_dag(
        model=start_model,
        score=score_titanic1,
        structure_score=score_structure_prior,
        tabu_list=tabu_list,
        max_indegree=float("inf"),
        forbidden_edges=set(),
        required_edges=set(),
    )
    assert len(list(legal_ops_tabu)) == 18

    legal_ops_indegree = est_titanic1._legal_operations_dag(
        model=start_model,
        score=score_titanic1,
        structure_score=score_structure_prior,
        tabu_list=[],
        max_indegree=1,
        forbidden_edges=set(),
        required_edges=set(),
    )
    assert len(list(legal_ops_indegree)) == 11

    legal_ops_both = est_titanic1._legal_operations_dag(
        model=start_model,
        score=score_titanic1,
        structure_score=score_structure_prior,
        tabu_list=tabu_list,
        max_indegree=1,
        forbidden_edges=set(),
        required_edges=set(),
    )

    legal_ops_both_ref = {
        ("+", ("Embarked", "Survived")): 10.050632580087495,
        ("+", ("Survived", "Pclass")): 41.8886804654893,
        ("+", ("Age", "Survived")): -23.635716036430722,
        ("+", ("Pclass", "Survived")): 41.81314459373152,
        ("+", ("Sex", "Pclass")): 4.772261678791324,
        ("-", ("Pclass", "Age")): 11.546515590730905,
        ("-", ("Pclass", "Embarked")): -32.17148283253266,
        ("flip", ("Pclass", "Embarked")): 3.3563814191275583,
        ("flip", ("Survived", "Sex")): 0.0397370279797542,
    }
    assert {op for op, score in legal_ops_both} == set(legal_ops_both_ref)
    for op, score in legal_ops_both:
        assert score == pytest.approx(legal_ops_both_ref[op])


def test_estimate_rand(rand_data):
    est1 = HillClimbSearch(scoring_method="k2", return_type="dag", show_progress=False)
    est1.fit(rand_data)
    assert set(est1.causal_graph_.nodes()) == {"A", "B", "C"}
    assert list(est1.causal_graph_.edges()) == [("B", "C")] or list(est1.causal_graph_.edges()) == [("C", "B")]

    est2 = HillClimbSearch(
        scoring_method="k2",
        start_dag=DiscreteBayesianNetwork([("A", "B"), ("A", "C")]),
        return_type="dag",
        show_progress=False,
    )
    est2.fit(rand_data)
    assert list(est2.causal_graph_.edges()) == [("B", "C")] or list(est2.causal_graph_.edges()) == [("C", "B")]

    expert_knowledge = ExpertKnowledge(required_edges=[("B", "C")])
    est3 = HillClimbSearch(
        scoring_method="k2",
        expert_knowledge=expert_knowledge,
        return_type="dag",
        show_progress=False,
    )
    est3.fit(rand_data)
    assert [("B", "C")] == list(est3.causal_graph_.edges())


def test_estimate_titanic(titanic_data2):
    est = HillClimbSearch(scoring_method="k2", return_type="dag", show_progress=False)
    est.fit(titanic_data2)
    assert set(est.causal_graph_.edges()) == {("Survived", "Pclass"), ("Sex", "Pclass"), ("Sex", "Survived")}

    expert_knowledge = ExpertKnowledge(required_edges=[("Pclass", "Survived")])
    est2 = HillClimbSearch(
        scoring_method="k2",
        expert_knowledge=expert_knowledge,
        return_type="dag",
        show_progress=False,
    )
    est2.fit(titanic_data2)
    assert ("Pclass", "Survived") in est2.causal_graph_.edges()

    temporal_knowledge = ExpertKnowledge(temporal_order=[["Pclass", "Sex"], ["Survived"]])
    est3 = HillClimbSearch(expert_knowledge=temporal_knowledge, return_type="dag", show_progress=False)
    est3.fit(titanic_data2)
    assert est3.causal_graph_.edges() <= {
        ("Sex", "Survived"),
        ("Sex", "Pclass"),
        ("Pclass", "Sex"),
        ("Pclass", "Survived"),
    }


def test_no_legal_operation():
    data = pd.DataFrame(
        [
            [1, 0, 0, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 0, 1, 0, 1, 0],
            [1, 0, 0, 0, 0, 1, 0, 1, 1],
            [1, 1, 0, 1, 0, 1, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 1, 0, 0],
        ],
        columns=list("ABCDEFGHI"),
        dtype="category",
    )
    expert_knowledge = ExpertKnowledge(
        required_edges=[("A", "B"), ("B", "C")],
        forbidden_edges=[(u, v) for u in data.columns for v in data.columns],
    )
    est = HillClimbSearch(
        scoring_method="k2",
        expert_knowledge=expert_knowledge,
        return_type="dag",
        show_progress=False,
    )
    est.fit(data)


@pytest.mark.parametrize("scoring_method", ["k2", "bdeu", "bds", "bic-d", "aic-d"])
def test_estimate_discrete(rand_data, scoring_method):
    est = HillClimbSearch(scoring_method=scoring_method, return_type="dag", show_progress=False)
    est.fit(rand_data)


@pytest.mark.parametrize("scoring_method", ["ll-cg", "aic-cg", "bic-cg"])
def test_estimate_mixed(titanic_data1, scoring_method):
    est = HillClimbSearch(scoring_method=scoring_method, return_type="dag", show_progress=False)
    est.fit(titanic_data1)


def test_search_space():
    adult_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/adult.csv")

    search_space = [
        ("Age", "Education"),
        ("Education", "HoursPerWeek"),
        ("Education", "Income"),
        ("HoursPerWeek", "Income"),
        ("Age", "Income"),
    ]

    expert_knowledge = ExpertKnowledge(search_space=search_space)

    est = HillClimbSearch(
        scoring_method="k2",
        expert_knowledge=expert_knowledge,
        return_type="dag",
        show_progress=False,
    )
    est.fit(adult_data)
    # assert if dag is a subset of search_space
    for edge in est.causal_graph_.edges():
        assert edge in search_space


@pytest.mark.parametrize("scoring_method", ["aic-g", "bic-g"])
def test_estimate_gaussian(scoring_method):
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv", index_col=0)
    est = HillClimbSearch(scoring_method=scoring_method, return_type="dag", show_progress=False)
    est.fit(data)


def test_estimate_mixed_data():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)
    data["A_cat"] = data.A_cat.astype("category")
    data["B_cat"] = data.B_cat.astype("category")
    data["C_cat"] = data.C_cat.astype("category")
    data["B_int"] = data.B_int.astype("category")

    est = HillClimbSearch(scoring_method="ll-cg", return_type="dag", show_progress=False)
    est.fit(data)


def test_score():
    asia_model = load_model("bnlearn/asia")
    data = asia_model.simulate(n_samples=int(1e4), seed=42)
    est = HillClimbSearch(
        return_type="dag",
        show_progress=False,
    )

    with pytest.raises(NotFittedError):
        corr_score = est.score(X=data)

    est.fit(X=data)
    corr_score = est.score(X=data)
    shd = est.score(true_graph=asia_model)

    assert np.round(corr_score, 4) > 0.5
    assert shd, 2

    corr_score = est.score(X=data, metric=CorrelationScore(significance_level=0.01))
    shd = est.score(true_graph=asia_model, metric=SHD())

    assert np.round(corr_score, 4) > 0.5
    assert shd, 2

    structure_score = est.score(X=data, metric="structure_score")
    shd = est.score(true_graph=asia_model, metric="SHD")
    assert np.round(structure_score, 4) > -3e4
    assert shd, 2
