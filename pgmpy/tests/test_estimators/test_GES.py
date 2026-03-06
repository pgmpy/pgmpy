import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import GES, ExpertKnowledge


@pytest.fixture
def random_data_estimator():
    rand_data = pd.DataFrame(
        np.random.randint(0, 5, size=(int(1e4), 2)),
        columns=list("AB"),
        dtype="category",
    )
    rand_data["C"] = rand_data["B"]
    return GES(rand_data, use_cache=False)


@pytest.fixture
def titanic_estimators():
    titanic_data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")

    titanic_data1 = titanic_data[["Survived", "Sex", "Pclass", "Age", "Embarked"]]
    est1 = GES(titanic_data1, use_cache=False)

    titanic_data2 = titanic_data[["Survived", "Sex", "Pclass"]].astype("category")
    est2 = GES(titanic_data2, use_cache=False)

    return est1, est2


@pytest.fixture
def gaussian_data():
    return pd.read_csv(
        "pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv",
        index_col=0,
    )


@pytest.fixture
def mixed_data():
    data = pd.read_csv(
        "pgmpy/tests/test_estimators/testdata/mixed_testdata.csv",
        index_col=0,
    )
    data["A_cat"] = data.A_cat.astype("category")
    data["B_cat"] = data.B_cat.astype("category")
    data["C_cat"] = data.C_cat.astype("category")
    data["B_int"] = data.B_int.astype("category")
    return data


def test_estimate_discrete(random_data_estimator, titanic_estimators):
    est_rand = random_data_estimator
    est_titanic1, est_titanic2 = titanic_estimators

    est_rand.estimate()
    est_titanic1.estimate()

    temporal_knowledge = ExpertKnowledge(
        temporal_order=[["Pclass", "Sex"], ["Survived"]]
    )

    dag2 = est_titanic2.estimate(
        expert_knowledge=temporal_knowledge,
        scoring_method="k2",
    )

    expected_edges = {
        ("Sex", "Survived"),
        ("Sex", "Pclass"),
        ("Pclass", "Survived"),
    }

    assert set(dag2.edges()) == expected_edges


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
    est = GES(adult_data)

    dag = est.estimate(
        scoring_method="k2",
        expert_knowledge=expert_knowledge,
    )

    for edge in dag.edges():
        assert edge in search_space


def test_estimate_gaussian(gaussian_data):
    est = GES(gaussian_data)

    for score in ["aic-g", "bic-g"]:
        est.estimate(scoring_method=score, debug=True)


def test_estimate_mixed(mixed_data):
    est = GES(mixed_data)
    est.estimate(scoring_method="ll-cg")
