import unittest

import numpy as np

from pgmpy.inference import (HamiltonianMCda as HMC, ModifiedEuler as Euler,
                             JointGaussianDistribution as JGD, GradientLogPDFGaussian as grad_gaussian)


class TestHMCInference(unittest.TestCase):

    def setUp(self):
        mean_vec = np.array([1, 1])
        cov_matrix = np.matrix([[1, 0.7], [0.7, 4]])
        self.test_model = JGD(mean_vec, cov_matrix)
        self.sampler_leapfrog = HMC(model=self.test_model, Lambda=2, grad_log_pdf=grad_gaussian)
        self.sampler_euler = HMC(model=self.test_model, Lambda=2, grad_log_pdf=grad_gaussian,
                                 discretize_time=Euler)

    def test_sampler(self):
        theta0 = np.random.randn(1, 2)
        samples = self.sampler_leapfrog.sample(theta0, num_adapt=10000, num_samples=10000)
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.cov_matrix) < 0.3)

        theta0 = np.random.randn(1, 2)
        samples = self.sampler_euler.sample(theta0, num_adapt=10000, num_samples=10000)
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.cov_matrix) < 1.5)

    def tearDown(self):
        del self.sampler_euler
        del self.sampler_leapfrog
        del self.test_model
