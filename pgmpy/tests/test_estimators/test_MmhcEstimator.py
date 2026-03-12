import numpy as np
import pandas as pd
import pytest

from pgmpy.estimators import MmhcEstimator


@pytest.fixture
def mmhc_estimator():
    data = pd.DataFrame(
        np.random.randint(0, 2, size=(int(1e5), 3)), columns=list("XYZ")
    )
    data["sum"] = data.sum(axis=1)
    est = MmhcEstimator(data)
    return est


def test_estimate(mmhc_estimator):
    dag1 = mmhc_estimator.estimate()
    assert len(dag1.edges()) > 1
    assert set(dag1.edges()).issubset(
        {
            ("X", "sum"),
            ("Y", "sum"),
            ("Z", "sum"),
            ("sum", "X"),
            ("sum", "Y"),
            ("sum", "Z"),
            ("X", "Y"),
            ("X", "Z"),
            ("Y", "Z"),
            ("Y", "X"),
            ("Z", "X"),
            ("Z", "Y"),
        }
    )
    dag2 = mmhc_estimator.estimate(significance_level=0.001)
    assert len(dag2.edges()) > 1
    assert set(dag2.edges()).issubset(
        {
            ("X", "sum"),
            ("Y", "sum"),
            ("Z", "sum"),
            ("sum", "X"),
            ("sum", "Y"),
            ("sum", "Z"),
            ("X", "Y"),
            ("X", "Z"),
            ("Y", "Z"),
            ("Y", "X"),
            ("Z", "X"),
            ("Z", "Y"),
        }
    )
