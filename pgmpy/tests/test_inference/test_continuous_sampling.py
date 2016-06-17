import unittest

import numpy as np

from pgmpy.inference.continuous import (HamiltonianMC as HMC, HamiltonianMCda as HMCda, ModifiedEuler as Euler,
                                        GradLogPDFGaussian as GradGaussian)
from pgmpy.models import JointGaussianDistribution as JGD


class TestHMCInference(unittest.TestCase):

    def setUp(self):
        mean = np.array([1, 1])
        covariance = np.matrix([[1, 0.7], [0.7, 4]])
        self.test_model = JGD(['x', 'y'], mean, covariance)
        self.sampler_leapfrog = HMCda(model=self.test_model, grad_log_pdf=GradGaussian)
        self.sampler_euler = HMCda(model=self.test_model, grad_log_pdf=GradGaussian,
                                   simulate_dynamics=Euler)
        self.sampler_hmc = HMCda(model=self.test_model, grad_log_pdf=GradGaussian)
        self.sampler_hmc2 = HMCda(model=self.test_model)
        self.sampler_hmc3 = HMCda(model=self.test_model)

    def test_errors(self):
        with self.assertRaises(TypeError):
            HMCda(model=self.test_model, grad_log_pdf=Euler)
        with self.assertRaises(TypeError):
            HMCda(model=self.test_model, grad_log_pdf=GradGaussian, simulate_dynamics=GradGaussian)
        with self.assertRaises(TypeError):
            self.sampler_leapfrog.sample(position0=1, num_adapt=1, num_samples=1, trajectory_length=1)
        with self.assertRaises(TypeError):
            self.sampler_leapfrog.generate_sample(1, 1, 1, 1).send(None)
        with self.assertRaises(TypeError):
            self.sampler_hmc3.sample(position0=1, num_samples=1, trajectory_length=1)
        with self.assertRaises(TypeError):
            self.sampler_hmc3.generate_sample(1, 1, 1).next().send(None)
        with self.assertRaises(TypeError):
            JGD(['x', 'y'], mean=1, covariance=1)
        with self.assertRaises(TypeError):
            JGD(['x', 'y'], mean=np.ones(2), covariance=1)
        with self.assertRaises(TypeError):
            GradGaussian(1, self.test_model)
        with self.assertRaises(TypeError):
            Euler(grad_log_pdf=1, model=1, position=1, momentum=1, stepsize=1)
        with self.assertRaises(TypeError):
            Euler(grad_log_pdf=1, model=1, position=[1], momentum=1, stepsize=1)
        with self.assertRaises(TypeError):
            Euler(grad_log_pdf=1, model=1, position=[1], momentum=[1], stepsize=1)
        with self.assertRaises(ValueError):
            JGD(['x', 'y'], mean=np.ones(3), covariance=np.eye(2))
        with self.assertRaises(ValueError):
            JGD(['x', 'y'], mean=np.ones(3), covariance=[[1, 2, 3], [1, 2, 3]])
        with self.assertRaises(ValueError):
            Euler(grad_log_pdf=GradGaussian, model=self.test_model, position=[1, 2], momentum=[1], stepsize=1)

    def test_sampler(self):
        theta0 = np.random.randn(2)
        samples = self.sampler_leapfrog.sample(theta0, num_adapt=9998, num_samples=10000, trajectory_length=3)
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.covariance) < 0.3)

        gen_samples = self.sampler_euler.generate_sample(theta0, num_adapt=9998,
                                                         num_samples=10000, trajectory_length=3)
        samples = [sample for sample in gen_samples]
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.covariance) < 1.5)
        # High norm taken because of poor performance

        samples = self.sampler_hmc.sample(theta0, num_adapt=0, num_samples=10000, trajectory_length=3, stepsize=0.3)
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.covariance) < 5.0)
        # High norm taken because of inconsistent performance (0.4 to 5.0)

        gen_samples = self.sampler_hmc2.generate_sample(theta0, num_adapt=0, num_samples=10000,
                                                        trajectory_length=3, stepsize=0.3)
        samples = [sample for sample in gen_samples]
        concatenated_samples = np.concatenate(samples, axis=1)
        samples_cov = np.cov(concatenated_samples)
        self.assertTrue(np.linalg.norm(samples_cov - self.test_model.covariance) < 5.0)
        # High norm taken because of inconsistent performance (0.4 to 5.0)

    def tearDown(self):
        del self.sampler_euler
        del self.sampler_leapfrog
        del self.sampler_hmc
        del self.sampler_hmc2
        del self.test_model
