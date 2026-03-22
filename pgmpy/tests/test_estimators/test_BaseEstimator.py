import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import BaseEstimator


@pytest.fixture
def dataset1():
    return pd.DataFrame(data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]})


@pytest.fixture
def dataset2():
    return pd.DataFrame(
        data={
            "A": [0, np.nan, 1],
            "B": [0, 1, 0],
            "C": [1, 1, np.nan],
            "D": [np.nan, "Y", np.nan],
        }
    )


def test_state_count(dataset1):
    e = BaseEstimator(dataset1)

    assert e.state_counts("A").values.tolist() == [[2], [1]]
    assert e.state_counts("C", parents=["A", "B"]).values.tolist() == [
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 1.0, 0.0, 0.0],
    ]


def test_missing_data(dataset2):
    e = BaseEstimator(dataset2, state_names={"C": [0, 1]})

    assert e.state_counts("A").values.tolist() == [[1], [1]]
    assert e.state_counts("C", parents=["A", "B"]).values.tolist() == [
        [0, 0, 0, 0],
        [1, 0, 0, 0],
    ]
