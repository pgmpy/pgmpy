import unittest

import numpy.testing as np_test

from pgmpy.factors.continuous import ContinuousFactor
from pgmpy.factors.continuous.distributions import GaussianDistribution


class TestContinuousFactorInit(unittest.TestCase):
    def test_init(self):
        phi_str = ContinuousFactor(variables=['A', 'B', 'C'], dist='gaussian',
                                   mean=[1, 2, 1], cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(list(phi_str.dist.variables), ['A', 'B', 'C'])
        np_test.assert_almost_equal(phi_str.dist.mean, [[1], [2], [1]])
        np_test.assert_almost_equal(phi_str.dist.cov, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(list(phi_str.dist.evidence), [])

        gauss_dist = GaussianDistribution(variables=['A', 'B', 'C'], mean=[1, 2, 1],
                                          cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        phi_dist = ContinuousFactor(dist=gauss_dist)
        self.assertEqual(list(phi_dist.dist.variables), ['A', 'B', 'C'])
        np_test.assert_almost_equal(phi_dist.dist.mean, [[1], [2], [1]])
        np_test.assert_almost_equal(phi_dist.dist.cov, [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.assertEqual(list(phi_dist.dist.evidence), [])

    def test_init_errors(self):
        self.assertRaises(ValueError, ContinuousFactor, dist='gaussian')
        self.assertRaises(ValueError, ContinuousFactor, variables='variable',
                          dist='gaussian', mean=[1], cov=[[1]])
        self.assertRaises(ValueError, ContinuousFactor, variables=['A', 'B', 'C'],
                          dist='random_dist')
        self.assertRaises(ValueError, ContinuousFactor, variables=['A', 'B', 'C'],
                          dist='gaussian')
        self.assertRaises(ValueError, ContinuousFactor, variables=['A', 'B', 'C'],
                          dist='gaussian', mean=[1, 2, 1])
        self.assertRaises(ValueError, ContinuousFactor, variables=['A', 'B', 'C'],
                          dist=(1, 2,))


class TestContinuousFactorMethods(unittest.TestCase):
    def setUp(self):
        self.joint_dist = GaussianDistribution(variables=['X', 'Y', 'Z'], mean=[1, 1, 1],
                                               cov=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self.joint_dist_c = GaussianDistribution(variables=['X', 'Y', 'Z'], mean=[0.2, 0.3, 0.4],
                                                 cov=[[0.4, 0.8, 0.7], [0.8, 0.9, 0.5], [0.7, 0.5, 1]])

        self.phi = ContinuousFactor(dist=self.joint_dist)
        self.phi_c = ContinuousFactor(dist=self.joint_dist_c)

    def test_marginalize_inplace_false(self):
        marginal_phi = self.phi.marginalize(['Y'], inplace=False)
        self.assertEqual(list(marginal_phi.dist.variables), ['X', 'Z'])
        self.assertEqual(list(marginal_phi.dist.evidence), [])
        np_test.assert_almost_equal(marginal_phi.dist.mean, [[1], [1]])
        np_test.assert_almost_equal(marginal_phi.dist.cov, [[1, 0], [0, 1]])
        
        marginal_phi_c = self.phi_c.marginalize(['X', 'Z'], inplace=False)
        self.assertEqual(list(marginal_phi_c.dist.variables), ['Y'])
        self.assertEqual(list(marginal_phi_c.dist.evidence), [])
        np_test.assert_almost_equal(marginal_phi_c.dist.mean, [[0.3]])
        np_test.assert_almost_equal(marginal_phi_c.dist.cov, [[0.9]])

    def test_marginalize_inplace_true(self):
        self.phi.marginalize(['Y'])
        self.assertEqual(list(self.phi.dist.variables), ['X', 'Z'])
        self.assertEqual(list(self.phi.dist.evidence), [])
        np_test.assert_almost_equal(self.phi.dist.mean, [[1], [1]])
        np_test.assert_almost_equal(self.phi.dist.cov, [[1, 0], [0, 1]])

    def test_reduce_inplace_false(self):
        reduced_phi = self.phi.reduce(values=[('Y', 1)], inplace=False)
        self.assertEqual(list(reduced_phi.dist.variables), ['X', 'Z'])
        np_test.assert_almost_equal(reduced_phi.dist.mean, [[1], [1]], decimal=2)
        np_test.assert_almost_equal(reduced_phi.dist.cov, [[1, 0], [0, 1]], decimal=2)
        self.assertEqual(list(reduced_phi.dist.evidence), [])

        reduced_phi_c = self.phi_c.reduce(values=[('Y', 0.5), ('Z', 0.8)], inplace=False)
        self.assertEqual(list(reduced_phi_c.dist.variables), ['X'])
        np_test.assert_almost_equal(reduced_phi_c.dist.mean, [[0.48]], decimal=2)
        np_test.assert_almost_equal(reduced_phi_c.dist.cov, [[-0.4]], decimal=2)
        self.assertEqual(list(reduced_phi_c.dist.evidence), [])

    def test_reduce_inplace_true(self):
        self.phi.reduce(values=[('Y', 1)], inplace=True)
        self.assertEqual(list(self.phi.dist.variables), ['X', 'Z'])
        np_test.assert_almost_equal(self.phi.dist.mean, [[1], [1]], decimal=2)
        np_test.assert_almost_equal(self.phi.dist.cov, [[1, 0], [0, 1]], decimal=2)
        self.assertEqual(list(self.phi.dist.evidence), [])

    def tearDown(self):
        del self.joint_dist
        del self.joint_dist_c
        del self.phi
        del self.phi_c
