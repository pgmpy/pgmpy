import pytest
from numpy import nan
from pandas import DataFrame

from pgmpy.estimators import ParameterEstimator
from pgmpy.models import DiscreteBayesianNetwork


@pytest.fixture
def m1():
    return DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])


@pytest.fixture
def d1():
    return DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]})


@pytest.fixture
def d2():
    return DataFrame(
        data={
            "A": [0, nan, 1],
            "B": [0, 1, 0],
            "C": [1, 1, nan],
            "D": [nan, "Y", nan],
        }
    )


def test_state_count(m1, d1):
    e = ParameterEstimator(m1, d1)
    assert e.state_counts("A").values.tolist() == [[2], [1]]
    assert e.state_counts("C").values.tolist() == [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_missing_data(m1, d2):
    e = ParameterEstimator(m1, d2, state_names={"C": [0, 1]})
    assert e.state_counts("A").values.tolist() == [[1], [1]]
    assert e.state_counts("C").values.tolist() == [[0, 0, 0, 0], [1, 0, 0, 0]]
