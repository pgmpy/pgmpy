import pytest
import pandas as pd
import numpy as np

from pgmpy.estimators import MmhcEstimator
from pgmpy.factors.discrete import TabularCPD


class TestMmhcEstimator:
    @pytest.fixture
    def setup(self):
        data1 = pd.DataFrame(
            np.random.randint(0, 2, size=(int(1e5), 3)), columns=list("XYZ")
        )
        data1["sum"] = data1.sum(axis=1)
        est1 = MmhcEstimator(data1)
        return data1, est1

    def test_estimate(self, setup):
        data1, est1 = setup
        dag1 = est1.estimate()
        assert len(dag1.edges()) > 1
        assert set(dag1.edges()).issubset(
            set(
                [
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
                ]
            )
        )
        dag2 = est1.estimate(significance_level=0.001)
        assert len(dag2.edges()) > 1
        assert set(dag2.edges()).issubset(
            set(
                [
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
                ]
            )
        )
