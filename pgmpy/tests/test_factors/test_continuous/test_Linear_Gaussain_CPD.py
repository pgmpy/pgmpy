import unittest

import numpy.testing as np_test

from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.continuous.distributions import GaussianDistribution

class TestInit(unittest.TestCase):
    def test_init(self):
        cpd = LinearGaussianCPD(variable='A', mean=[1, 1, 1], cov=1, evidence=['B', 'C'])
        self.assertEqual(list(cpd.dist.variables), ['A'])
        np_test.assert_almost_equal(cpd.dist.mean, [[1, 1, 1]])
        np_test.assert_almost_equal(cpd.dist.cov, [[1]])
        self.assertEqual(list(cpd.dist.evidence), ['B', 'C'])

    def test_init_wrong_mean_size(self):
        self.assertRaises(ValueError, LinearGaussianCPD, variable='A', mean=[1, 1, 1, 1],
                          cov=1, evidence=['B', 'C'])

    def test_init_single_var_dist(self):
        single_var_dist = GaussianDistribution(variables=['A'], mean=[1, 1, 1], cov=[1],
                                               evidence=['B', 'C'])
        cpd = LinearGaussianCPD(dist=single_var_dist)
        self.assertEqual(list(cpd.dist.variables), ['A'])
        np_test.assert_almost_equal(cpd.dist.mean, [[1, 1, 1]])
        np_test.assert_almost_equal(cpd.dist.cov, [[1]])
        self.assertEqual(list(cpd.dist.evidence), ['B', 'C'])


    def test_init_multi_var_dist(self):
        self.assertRaises(ValueError, GaussianDistribution, variables=['A', 'B'], 
                          mean=[[1, 1, 1], [1, 1, 1]], cov=[[1, 0], [0, 1]],
                          evidence=['C', 'D'])
