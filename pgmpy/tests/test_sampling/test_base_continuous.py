import unittest

import numpy as np

from pgmpy.factors.distributions import GaussianDistribution as JGD
from pgmpy.sampling import LeapFrog, ModifiedEuler, GradLogPDFGaussian


class TestGradLogPDFGaussian(unittest.TestCase):

    def setUp(self):
        mean = np.array([1, 2, 3, 4])
        covariance = np.array([[1, 0.2, 0.4, 0.7], [0.2, 2, 0.5, 0.8], [0.4, 0.5, 3, 0.6], [0.7, 0.8, 0.6, 4]])
        self.test_model = JGD(['x', 'y', 'z', 't'], mean, covariance)
        self.test_gradient = GradLogPDFGaussian([0, 0, 0, 0], self.test_model)

    def test_error(self):
        with self.assertRaises(TypeError):
            GradLogPDFGaussian(1, self.test_model)
        with self.assertRaises(ValueError):
            GradLogPDFGaussian([1, 1], self.test_model)

    def test_gradient(self):
        grad, log = self.test_gradient.get_gradient_log_pdf()
        np.testing.assert_almost_equal(grad, np.array([0.05436475, 0.49454937, 0.75465073, 0.77837868]))
        np.testing.assert_almost_equal(log, -3.21046521505)


class TestLeapFrog(unittest.TestCase):

    def setUp(self):
        mean = np.array([-1, 1, -1])
        covariance = np.array([[1, 0.6, 0.5], [0.6, 2, 0.3], [0.5, 0.3, 1]])
        self.test_model = JGD(['x', 'y', 'z'], mean, covariance)
        position = [0, 0, 0]
        momentum = [-1, -1, -1]
        self.test_with_grad_log = LeapFrog(model=self.test_model, position=position, momentum=momentum,
                                           stepsize=0.3, grad_log_pdf=GradLogPDFGaussian, grad_log_position=None)
        grad_log_position, _ = GradLogPDFGaussian(position, self.test_model).get_gradient_log_pdf()
        self.test_without_grad_log = LeapFrog(model=self.test_model, position=position, momentum=momentum,
                                              stepsize=0.4, grad_log_pdf=GradLogPDFGaussian,
                                              grad_log_position=grad_log_position)

    def test_errors(self):
        with self.assertRaises(TypeError):
            LeapFrog(model=self.test_model, position=1, momentum=[1, 1], stepsize=0.1,
                     grad_log_pdf=GradLogPDFGaussian)
        with self.assertRaises(TypeError):
            LeapFrog(model=self.test_model, position=[1, 1], momentum=1, stepsize=0.1,
                     grad_log_pdf=GradLogPDFGaussian)
        with self.assertRaises(ValueError):
            LeapFrog(model=self.test_model, position=[1, 1], momentum=[1], stepsize=0.1,
                     grad_log_pdf=GradLogPDFGaussian)
        with self.assertRaises(TypeError):
            LeapFrog(model=self.test_model, position=[1], momentum=[1], stepsize=0.1, grad_log_pdf=1)
        with self.assertRaises(ValueError):
            LeapFrog(model=self.test_model, position=[1, 1], momentum=[1, 1], stepsize=0.1,
                     grad_log_pdf=GradLogPDFGaussian)
        with self.assertRaises(TypeError):
            LeapFrog(model=self.test_model, position=[1, 1, 1], momentum=[1, 1, 1], stepsize=0.1,
                     grad_log_pdf=GradLogPDFGaussian, grad_log_position=1)
        with self.assertRaises(ValueError):
            LeapFrog(model=self.test_model, position=[1, 1, 1], momentum=[1, 1, 1], stepsize=0.1,
                     grad_log_pdf=GradLogPDFGaussian, grad_log_position=[1, 1])

    def test_leapfrog_methods(self):
        new_pos, new_momentum, new_grad = self.test_with_grad_log.get_proposed_values()
        np.testing.assert_almost_equal(new_pos, np.array([-0.35634146, -0.25609756, -0.33]))
        np.testing.assert_almost_equal(new_momentum, np.array([-1.3396624, -0.70344884, -1.16963415]))
        np.testing.assert_almost_equal(new_grad, np.array([-1.0123835, 1.00139798, -0.46422764]))
        new_pos, new_momentum, new_grad = self.test_without_grad_log.get_proposed_values()
        np.testing.assert_almost_equal(new_pos, np.array([-0.5001626, -0.32195122, -0.45333333]))
        np.testing.assert_almost_equal(new_momentum, np.array([-1.42947981, -0.60709102, -1.21246612]))
        np.testing.assert_almost_equal(new_grad, np.array([-0.89536651, 0.98893516, -0.39566396]))

    def tearDown(self):
        del self.test_model
        del self.test_with_grad_log
        del self.test_without_grad_log


class TestModifiedEuler(unittest.TestCase):

    def setUp(self):
        mean = np.array([0, 0])
        covariance = np.array([[-1, 0.8], [0.8, 3]])
        self.test_model = JGD(['x', 'y'], mean, covariance)
        position = [0, 0]
        momentum = [-2, 1]
        self.test_with_grad_log = ModifiedEuler(model=self.test_model, position=position, momentum=momentum,
                                                stepsize=0.5, grad_log_pdf=GradLogPDFGaussian, grad_log_position=None)
        grad_log_position, _ = GradLogPDFGaussian(position, self.test_model).get_gradient_log_pdf()
        self.test_without_grad_log = ModifiedEuler(model=self.test_model, position=position, momentum=momentum,
                                                   stepsize=0.3, grad_log_pdf=GradLogPDFGaussian,
                                                   grad_log_position=grad_log_position)

    def test_modified_euler_methods(self):
        new_pos, new_momentum, new_grad = self.test_with_grad_log.get_proposed_values()
        np.testing.assert_almost_equal(new_pos, np.array([-1.0, 0.5]))
        np.testing.assert_almost_equal(new_momentum, np.array([-2.0, 1.0]))
        np.testing.assert_almost_equal(new_grad, np.array([-0.93406593, 0.08241758]))
        new_pos, new_momentum, new_grad = self.test_without_grad_log.get_proposed_values()
        np.testing.assert_almost_equal(new_pos, np.array([-0.6, 0.3]))
        np.testing.assert_almost_equal(new_momentum, np.array([-2.0, 1.0]))
        np.testing.assert_almost_equal(new_grad, np.array([-0.56043956, 0.04945055]))

    def tearDown(self):
        del self.test_model
        del self.test_with_grad_log
        del self.test_without_grad_log
