import unittest

import numpy as np

from pgmpy.factors.distributions import GaussianDistribution as JGD
from pgmpy.sampling import (HamiltonianMC as HMC, HamiltonianMCDA as HMCda, GradLogPDFGaussian, NoUTurnSampler as NUTS,
                            NoUTurnSamplerDA as NUTSda)


class TestHMCInference(unittest.TestCase):

    def setUp(self):
        mean = [-1, 1, -1]
        covariance = np.array([[3, 0.8, 0.2], [0.8, 2, 0.3], [0.2, 0.3, 1]])
        self.test_model = JGD(['x', 'y', 'z'], mean, covariance)
        self.hmc_sampler = HMCda(model=self.test_model, grad_log_pdf=GradLogPDFGaussian)

    def test_errors(self):
        with self.assertRaises(TypeError):
            HMCda(model=self.test_model, grad_log_pdf=1)
        with self.assertRaises(TypeError):
            HMCda(model=self.test_model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=1)
        with self.assertRaises(ValueError):
            HMCda(model=self.test_model, delta=-1)
        with self.assertRaises(TypeError):
            self.hmc_sampler.sample(initial_pos=1, num_adapt=1, num_samples=1, trajectory_length=1)
        with self.assertRaises(TypeError):
            self.hmc_sampler.generate_sample(1, 1, 1, 1).send(None)
        with self.assertRaises(TypeError):
            HMC(model=self.test_model).sample(initial_pos=1, num_samples=1, trajectory_length=1)
        with self.assertRaises(TypeError):
            HMC(model=self.test_model).generate_sample(1, 1, 1).send(None)

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


class TestNUTSInference(unittest.TestCase):

    def setUp(self):
        mean = np.array([-1, 1, 0])
        covariance = np.array([[6, 0.7, 0.2], [0.7, 3, 0.9], [0.2, 0.9, 1]])
        self.test_model = JGD(['x', 'y', 'z'], mean, covariance)
        self.nuts_sampler = NUTSda(model=self.test_model, grad_log_pdf=GradLogPDFGaussian)

    def test_errors(self):
        with self.assertRaises(TypeError):
            NUTS(model=self.test_model, grad_log_pdf=JGD)
        with self.assertRaises(TypeError):
            NUTS(model=self.test_model, grad_log_pdf=None, simulate_dynamics=GradLogPDFGaussian)
        with self.assertRaises(ValueError):
            NUTSda(model=self.test_model, delta=-0.2, grad_log_pdf=None)
        with self.assertRaises(ValueError):
            NUTSda(model=self.test_model, delta=1.1, grad_log_pdf=GradLogPDFGaussian)
        with self.assertRaises(TypeError):
            NUTS(self.test_model, GradLogPDFGaussian).sample(initial_pos={1, 1, 1}, num_samples=1)
        with self.assertRaises(ValueError):
            NUTS(self.test_model, GradLogPDFGaussian).sample(initial_pos=[1, 1], num_samples=1)
        with self.assertRaises(TypeError):
            NUTSda(self.test_model, GradLogPDFGaussian).sample(initial_pos=1, num_samples=1, num_adapt=1)
        with self.assertRaises(ValueError):
            NUTSda(self.test_model, GradLogPDFGaussian).sample(initial_pos=[1, 1, 1, 1], num_samples=1, num_adapt=1)
        with self.assertRaises(TypeError):
            NUTS(self.test_model, GradLogPDFGaussian).generate_sample(initial_pos=0.1, num_samples=1).send(None)
        with self.assertRaises(ValueError):
            NUTS(self.test_model, GradLogPDFGaussian).generate_sample(initial_pos=(0, 1, 1, 1),
                                                                      num_samples=1).send(None)
        with self.assertRaises(TypeError):
            NUTSda(self.test_model, GradLogPDFGaussian).generate_sample(initial_pos=[[1, 2, 3]], num_samples=1,
                                                                        num_adapt=1).send(None)
        with self.assertRaises(ValueError):
            NUTSda(self.test_model, GradLogPDFGaussian).generate_sample(initial_pos=[1], num_samples=1,
                                                                        num_adapt=1).send(None)

    def test_sampling(self):
        np.random.seed(1010101)
        samples = self.nuts_sampler.sample(initial_pos=[-0.4, 1, 3.6], num_adapt=0, num_samples=10000,
                                           return_type='recarray')
        sample_array = np.array([samples[var_name] for var_name in self.test_model.variables])
        sample_covariance = np.cov(sample_array)
        self.assertTrue(np.linalg.norm(sample_covariance - self.test_model.covariance) < 3)

        np.random.seed(1210161)
        samples = self.nuts_sampler.generate_sample(initial_pos=[-0.4, 1, 3.6], num_adapt=0, num_samples=10000)
        samples_array = np.array([sample for sample in samples])
        sample_covariance = np.cov(samples_array.T)
        self.assertTrue(np.linalg.norm(sample_covariance - self.test_model.covariance) < 3)

        np.random.seed(12313131)
        samples = self.nuts_sampler.sample(initial_pos=[0.2, 0.4, 2.2], num_adapt=10000, num_samples=10000)
        sample_covariance = np.cov(samples.values.T)
        self.assertTrue(np.linalg.norm(sample_covariance - self.test_model.covariance) < 0.4)

        np.random.seed(921312312)
        samples = self.nuts_sampler.generate_sample(initial_pos=[0.2, 0.4, 2.2], num_adapt=10000, num_samples=10000)
        samples_array = np.array([sample for sample in samples])
        sample_covariance = np.cov(samples_array.T)
        self.assertTrue(np.linalg.norm(sample_covariance - self.test_model.covariance) < 0.4)

    def tearDown(self):
        del self.test_model
        del self.nuts_sampler
