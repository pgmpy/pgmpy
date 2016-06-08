import unittest

import numpy as np

from pgmpy.inference import (HamiltonianMCda as HMCda, ModifiedEuler as Euler,
                             JointGaussianDistribution as JGD, GradLogPDFGaussian as grad_gaussian)


class TestHMCInference(unittest.TestCase):

    def setUp(self):
        mean_vec = np.array([1, 1])
        cov_matrix = np.matrix([[1, 0.7], [0.7, 4]])
        self.test_model = JGD(mean_vec, cov_matrix)
        self.sampler_leapfrog = HMCda(model=self.test_model, grad_log_pdf=grad_gaussian)
        self.sampler_euler = HMCda(model=self.test_model, grad_log_pdf=grad_gaussian,
                                   discretize_time=Euler)
        self.sampler_hmc = HMCda(model=self.test_model, grad_log_pdf=grad_gaussian)
        self.sampler_hmc2 = HMCda(model=self.test_model, grad_log_pdf=grad_gaussian)

    def test_sampler(self):
        theta0 = np.random.randn(1, 2)
        samples = self.sampler_leapfrog.sample(theta0, num_adapt=9998, num_samples=10000, Lambda=3)
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.cov_matrix) < 0.3)

        theta0 = np.random.randn(1, 2)
        gen_samples = self.sampler_euler.generate_sample(theta0, num_adapt=9998, num_samples=10000, Lambda=3)
        samples = [sample for sample in gen_samples]
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.cov_matrix) < 1.5)

        theta0 = np.random.randn(1, 2)
        samples = self.sampler_hmc.sample(theta0, num_adapt=0, num_samples=10000, Lambda=3)
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.cov_matrix) < 1.0)

        theta0 = np.random.randn(1, 2)
        gen_samples = self.sampler_hmc2.generate_sample(theta0, num_adapt=0, num_samples=10000, Lambda=3)
        samples = [sample for sample in gen_samples]
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.cov_matrix) < 1.0)

    def tearDown(self):
        del self.sampler_euler
        del self.sampler_leapfrog
        del self.sampler_hmc
        del self.sampler_hmc2
        del self.test_model
