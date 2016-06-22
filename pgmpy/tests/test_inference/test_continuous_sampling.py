import unittest

import numpy as np

from pgmpy.inference.continuous import (HamiltonianMC as HMC, HamiltonianMCda as HMCda, ModifiedEuler,
                                        GradLogPDFGaussian as GradGaussian)
from pgmpy.models import JointGaussianDistribution as JGD


class TestHMCInference(unittest.TestCase):

    def setUp(self):
        mean = [-1, 1, -1]
        covariance = np.array([[3, 0.8, 0.2], [0.8, 2, 0.3], [0.2, 0.3, 1]])
        self.test_model = JGD(['x', 'y', 'z'], mean, covariance)
        self.hmc_sampler = HMCda(model=self.test_model)

    def test_errors(self):
        with self.assertRaises(TypeError):
            HMCda(model=self.test_model, grad_log_pdf=1)
        with self.assertRaises(TypeError):
            HMCda(model=self.test_model, grad_log_pdf=GradGaussian, simulate_dynamics=1)
        with self.assertRaises(TypeError):
            self.hmc_sampler.sample(initial_pos=1, num_adapt=1, num_samples=1, trajectory_length=1)
        with self.assertRaises(TypeError):
            self.hmc_sampler.generate_sample(1, 1, 1, 1).send(None)
        with self.assertRaises(TypeError):
            HMC(model=self.test_model).sample(position0=1, num_samples=1, trajectory_length=1)
        with self.assertRaises(TypeError):
            HMC(model=self.test_model).generate_sample(1, 1, 1).send(None)
        # TODO: Remove them after implementation of JointGaussianDistribution
        with self.assertRaises(TypeError):
            JGD(['x', 'y'], mean=1, covariance=1)
        with self.assertRaises(TypeError):
            JGD(['x', 'y'], mean=np.ones(2), covariance=1)
        with self.assertRaises(ValueError):
            JGD(['x', 'y'], mean=np.ones(3), covariance=np.eye(2))
        with self.assertRaises(ValueError):
            JGD(['x', 'y'], mean=np.ones(3), covariance=[[1, 2, 3], [1, 2, 3]])

    def test_acceptance_prob(self):
        acceptance_probability = self.hmc_sampler._acceptance_prob(np.array([1, 2, 3]), np.array([2, 3, 4]),
                                                                   np.array([1, -1, 1]), np.array([0, 0, 0]))
        np.testing.assert_almost_equal(acceptance_probability, 0.0347363)

    def test_find_resonable_stepsize(self):
        np.random.seed(987654321)
        stepsize = self.hmc_sampler._find_reasonable_stepsize(np.array([-1, 1, -1]))
        np.testing.assert_almost_equal(stepsize, 2.0)

    def test_adapt_params(self):
        stepsize, stepsize_bar, h_bar = self.hmc_sampler._adapt_params(0.0025, 1, 1, np.log(0.025), 2, 1)
        np.testing.assert_almost_equal(stepsize, 3.13439452e-13)
        np.testing.assert_almost_equal(stepsize_bar, 3.6742481e-08)
        np.testing.assert_almost_equal(h_bar, 0.8875)

    def test_sample(self):
        # Seeding is done for _find_reasonable_stepsize method
        # Testing sample method simple HMC
        np.random.seed(3124141)
        samples = self.hmc_sampler.sample(initial_pos=[0.3, 0.4, 0.2], num_adapt=0,
                                          num_samples=10000, trajectory_length=4)
        covariance = np.cov(samples.values.T)
        self.assertTrue(np.linalg.norm(covariance - self.test_model.covariance) < 3)

        # Testing sample of method of HMCda
        np.random.seed(3124141)
        samples = self.hmc_sampler.sample(initial_pos=[0.6, 0.2, 0.8], num_adapt=10000,
                                          num_samples=10000, trajectory_length=4)
        covariance = np.cov(samples.values.T)
        self.assertTrue(np.linalg.norm(covariance - self.test_model.covariance) < 0.3)

        # Testing generate_sample method of simple HMC
        np.random.seed(3124141)
        gen_samples = self.hmc_sampler.generate_sample(initial_pos=[0.3, 0.4, 0.2], num_adapt=0,
                                                       num_samples=10000, trajectory_length=4)
        samples = np.array([sample for sample in gen_samples])
        covariance = np.cov(samples.T)
        self.assertTrue(np.linalg.norm(covariance - self.test_model.covariance) < 3)

        # Testing sample of method of HMCda
        np.random.seed(3124141)
        gen_samples = self.hmc_sampler.generate_sample(initial_pos=[0.6, 0.2, 0.8], num_adapt=10000,
                                                       num_samples=10000, trajectory_length=4)
        samples = np.array([sample for sample in gen_samples])
        covariance = np.cov(samples.T)
        self.assertTrue(np.linalg.norm(covariance - self.test_model.covariance) < 0.3)

    def tearDown(self):
        del self.hmc_sampler
        del self.test_model
