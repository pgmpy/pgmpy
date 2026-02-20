import pytest
from numpy import nan
from pandas import DataFrame

from pgmpy.estimators import ParameterEstimator
from pgmpy.models import DiscreteBayesianNetwork


@pytest.fixture
def estimator_setup():
    """Fixture providing test data for ParameterEstimator tests."""
    m1 = DiscreteBayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
    d1 = DataFrame(
        data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
    )
    d2 = DataFrame(
        data={
            "A": [0, nan, 1],
            "B": [0, 1, 0],
            "C": [1, 1, nan],
            "D": [nan, "Y", nan],
        }
    )
    return m1, d1, d2


def test_state_count(estimator_setup):
    """Test state_counts method of ParameterEstimator."""
    m1, d1, _ = estimator_setup
    e = ParameterEstimator(m1, d1)
    assert e.state_counts("A").values.tolist() == [[2], [1]]
    assert e.state_counts("C").values.tolist() == [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_missing_data(estimator_setup):
    """Test ParameterEstimator with missing data."""
    m1, _, d2 = estimator_setup
    e = ParameterEstimator(m1, d2, state_names={"C": [0, 1]})
    assert e.state_counts("A").values.tolist() == [[1], [1]]
    assert e.state_counts("C").values.tolist() == [[0, 0, 0, 0], [1, 0, 0, 0]]
