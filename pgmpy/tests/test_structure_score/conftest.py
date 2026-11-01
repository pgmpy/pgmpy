import pandas as pd
import pytest

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.structure_score import (
    AICCondGauss,
    AICGauss,
    BICCondGauss,
    BICGauss,
    LogLikelihoodCondGauss,
    LogLikelihoodGauss,
)


@pytest.fixture
def small_df():
    return pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]})


@pytest.fixture
def small_df_models():
    m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
    m2 = DiscreteBayesianNetwork([("C", "A"), ("C", "B"), ("A", "D")])
    return m1, m2


@pytest.fixture
def titanic_data():
    # data_link - "https://www.kaggle.com/c/titanic/download/train.csv"
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/titanic_train.csv")
    return data[["Survived", "Sex", "Pclass"]]


@pytest.fixture
def bds_df():
    """Example taken from https://arxiv.org/pdf/1708.00689.pdf"""
    return pd.DataFrame(
        data={
            "X": [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
            "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
            "Z": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
            "W": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        }
    )


@pytest.fixture
def bds_models():
    m1 = DiscreteBayesianNetwork([("W", "X"), ("Z", "X")])
    m1.add_node("Y")
    m2 = DiscreteBayesianNetwork([("W", "X"), ("Z", "X"), ("Y", "X")])
    return m1, m2


@pytest.fixture
def aic_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
    return AICGauss(data)


@pytest.fixture
def bic_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
    return BICGauss(data)


@pytest.fixture
def bic_cond_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)
    return BICCondGauss(data)


@pytest.fixture
def aic_cond_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)
    return AICCondGauss(data)


@pytest.fixture
def loglik_cond_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)
    return LogLikelihoodCondGauss(data)


@pytest.fixture
def loglik_cond_gauss_manual_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/mixed_testdata.csv", index_col=0)
    return LogLikelihoodCondGauss(data.iloc[:2, :])


@pytest.fixture
def loglik_gauss_score():
    data = pd.read_csv("pgmpy/tests/test_estimators/testdata/gaussian_testdata.csv")
    return LogLikelihoodGauss(data)


@pytest.fixture
def gauss_models():
    m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C")])
    m2 = DiscreteBayesianNetwork([("A", "B"), ("B", "C")])
    return m1, m2
