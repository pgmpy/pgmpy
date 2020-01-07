import unittest

import pandas as pd
import numpy as np

from pgmpy.estimators import MmhcEstimator, K2Score
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel


class TestMmhcEstimator(unittest.TestCase):
    def setUp(self):
        self.data1 = pd.DataFrame(
            np.random.randint(0, 2, size=(15000, 3)), columns=list("XYZ")
        )
        self.data1["sum"] = self.data1.sum(axis=1)
        self.est1 = MmhcEstimator(self.data1)

    @unittest.skip("currently disabled due to non-determenism")
    def test_estimate(self):
        self.assertTrue(
            set(self.est1.estimate().edges()).issubset(
                set(
                    [
                        ("X", "sum"),
                        ("Y", "sum"),
                        ("Z", "sum"),
                        ("sum", "X"),
                        ("sum", "Y"),
                        ("sum", "Z"),
                    ]
                )
            )
        )
        self.assertTrue(
            set(self.est1.estimate(significance_level=0.001).edges()).issubset(
                set(
                    [
                        ("X", "sum"),
                        ("Y", "sum"),
                        ("Z", "sum"),
                        ("sum", "X"),
                        ("sum", "Y"),
                        ("sum", "Z"),
                    ]
                )
            )
        )

    def tearDown(self):
        del self.data1
        del self.est1
