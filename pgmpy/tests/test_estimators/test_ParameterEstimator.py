import unittest

from pandas import DataFrame
from numpy import NaN

from pgmpy.models import BayesianModel
from pgmpy.estimators import ParameterEstimator


class TestParameterEstimator(unittest.TestCase):
    def setUp(self):
        self.m1 = BayesianModel([('A', 'C'), ('B', 'C'), ('D', 'B')])
        self.d1 = DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0], 'D': ['X', 'Y', 'Z']})
        self.d2 = DataFrame(data={'A': [0, NaN, 1], 'B': [0, 1, 0], 'C': [1, 1, NaN], 'D': [NaN, 'Y', NaN]})

    def test_state_count(self):
        e = ParameterEstimator(self.m1, self.d1)
        self.assertEqual(e.state_counts('A').values.tolist(), [[2], [1]])
        self.assertEqual(e.state_counts('C').values.tolist(),
                         [[0., 0., 1., 0.], [1., 1., 0., 0.]])

    def test_missing_data(self):
        e = ParameterEstimator(self.m1, self.d2, state_names={'C': [0, 1]}, complete_samples_only=False)
        self.assertEqual(e.state_counts('A', complete_samples_only=True).values.tolist(), [[0], [0]])
        self.assertEqual(e.state_counts('A').values.tolist(), [[1], [1]])
        self.assertEqual(e.state_counts('C', complete_samples_only=True).values.tolist(),
                         [[0, 0, 0, 0], [0, 0, 0, 0]])
        self.assertEqual(e.state_counts('C').values.tolist(),
                         [[0, 0, 0, 0], [1, 0, 0, 0]])

    def tearDown(self):
        del self.m1
        del self.d1
