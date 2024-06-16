import unittest

from numpy import nan
from pandas import DataFrame

from pgmpy.estimators import ParameterEstimator
from pgmpy.models import BayesianNetwork


class TestParameterEstimator(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianNetwork([("A", "C"), ("B", "C"), ("D", "B")])
        self.d1 = DataFrame(
            data={"A": [0, 0, 1], "B": [0, 1, 0], "C": [1, 1, 0], "D": ["X", "Y", "Z"]}
        )
        self.d2 = DataFrame(
            data={
                "A": [0, nan, 1],
                "B": [0, 1, 0],
                "C": [1, 1, nan],
                "D": [nan, "Y", nan],
            }
        )

    def test_state_count(self):
        e = ParameterEstimator(self.m1, self.d1)
        self.assertEqual(e.state_counts("A").values.tolist(), [[2], [1]])
        self.assertEqual(
            e.state_counts("C").values.tolist(),
            [[0.0, 0.0, 1.0, 0.0], [1.0, 1.0, 0.0, 0.0]],
        )

    def test_missing_data(self):
        e = ParameterEstimator(self.m1, self.d2, state_names={"C": [0, 1]})
        self.assertEqual(e.state_counts("A").values.tolist(), [[1], [1]])
        self.assertEqual(
            e.state_counts("C").values.tolist(), [[0, 0, 0, 0], [1, 0, 0, 0]]
        )

    def tearDown(self):
        del self.m1
        del self.d1
