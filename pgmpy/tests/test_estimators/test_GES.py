import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import GES
from pgmpy.example_models import load_model


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


def test_estimate_discrete(random_data_estimator, titanic_estimators):
    est_rand = random_data_estimator
    est_titanic1, _est_titanic2 = titanic_estimators

    est_rand.estimate()
    est_titanic1.estimate()


def test_cancer_model():
    cancer_model = load_model("bnlearn/cancer")
    data = cancer_model.simulate(3000, seed=0)

    est = GES(data)
    dag = est.estimate()

    assert set(cancer_model.edges) <= set(dag.edges)


def test_child_model():
    cancer_model = load_model("bnlearn/child")
    data = cancer_model.simulate(3000, seed=0)

    est = GES(data)
    dag = est.estimate()

    assert set(cancer_model.edges) <= set(dag.edges)


def test_estimate_gaussian(gaussian_data):
    est = GES(gaussian_data)

    for score in ["aic-g", "bic-g"]:
        est.estimate(scoring_method=score, debug=True)
