import unittest

import pandas as pd
import numpy as np

from pgmpy.estimators import MmhcEstimator, K2Score
from pgmpy.factors.discrete import TabularCPD


class TestMmhcEstimator(unittest.TestCase):
    def setUp(self):
        self.data1 = pd.DataFrame(
            np.random.randint(0, 2, size=(int(1e5), 3)), columns=list("XYZ")
        )
        self.data1["sum"] = self.data1.sum(axis=1)
        self.est1 = MmhcEstimator(self.data1)

    def test_estimate(self):
        dag1 = self.est1.estimate()
        self.assertTrue(len(dag1.edges()) > 1)
        self.assertTrue(
            set(dag1.edges()).issubset(
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
        )
        dag2 = self.est1.estimate(significance_level=0.001)
        self.assertTrue(len(dag2.edges()) > 1)
        self.assertTrue(
            set(dag2.edges()).issubset(
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
        )

    def tearDown(self):
        del self.data1
        del self.est1
