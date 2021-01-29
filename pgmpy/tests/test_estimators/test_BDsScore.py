import unittest

import pandas as pd

from pgmpy.models import BayesianModel
from pgmpy.estimators import BDsScore


class TestBDsScore(unittest.TestCase):
    def setUp(self):
        """ Example taken from https://arxiv.org/pdf/1708.00689.pdf"""
        self.d1 = pd.DataFrame(
            data={
                "X": [0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                "Y": [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
                "Z": [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1],
                "W": [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            }
        )
        self.m1 = BayesianModel([("W", "X"), ("Z", "X")])
        self.m1.add_node("Y")
        self.m2 = BayesianModel([("W", "X"), ("Z", "X"), ("Y", "X")])

    def test_score(self):
        self.assertAlmostEqual(
            BDsScore(self.d1, equivalent_sample_size=1).score(self.m1),
            -36.82311976667139,
        )
        self.assertEqual(
            BDsScore(self.d1, equivalent_sample_size=1).score(self.m2),
            -45.788991276221964,
        )

    def tearDown(self):
        del self.d1
        del self.m1
        del self.m2
