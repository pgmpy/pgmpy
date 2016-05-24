"""
    A collection of base class objects for continuous model
"""

import numpy as np


class AbstractGaussian(object):
    """
    A container class for gaussian. This class is to avoid cubersome code
    caused by mean_vec and cov_matrix.
    Will be removed when Multivariate Distributions will be implemented

    Paramters
    ---------
    mean_vec: A vector (row matrix or 1d array like object)
              Represents the mean of the distribution

    cov_matrix: A numpy.matrix of size len(mean_vec) x len(mean_vec),
                Covariance matrix for the distribution.

    """

    def __init__(self, mean_vec, cov_matrix):

        if not isinstance(mean_vec, np.matrix):
            if isinstance(mean_vec, (np.ndarray, list, tuple, set, frozenset)):
                length = len(mean_vec)
                mean_vec = np.matrix(mean_vec)
                mean_vec = np.reshape(mean_vec, (1, length))
            else:
                raise TypeError("mean_vec should be a 1d array type object")

        if not isinstance(cov_matrix, np.matrix):
            raise TypeError(
                "cov_matrix must be numpy.matrix type object")

        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError(
                "cov_matrix must be a square matrix")

        if mean.shape[1] != cov_matrix.shape[0]:
            raise ValueError("shape of mean vector should be 1 X d and" +
                             " shape of covariance matrix should be d X d")

        self.mean_vec = mean_vec
        self.cov_matrix = cov_matrix
        self.precision_matrix = np.linalg.inv(cov_matrix)


class GradientLogPDF(object):
    """
    Base class for gradient log of probability density function/ distribution

    Classes inheriting this base class can be passed as an argument for
    finding gradient log of probability distribution in inference algorithms

    Parameters
    ----------
    theta : A 1d array type object or a row matrix of shape 1 X d
            Vector representing values of parameter theta( or X)

    model : A AbstractGuassian type model

    Examples
    --------
    """

    def __init__(self, theta, model):
        # TODO: Take model as parameter instead of precision_matrix

        if not isinstance(theta, np.matrix):
            if isinstance(theta, (np.ndarray, list, tuple, set, frozenset)):
                length = len(theta)
                theta = np.matrix(theta)
                theta = np.reshape(theta, (1, length))
            else:
                raise TypeError("theta should be a 1d array type object")

        if theta.shape[1] != model.precision_matrix.shape[0]:
            raise ValueError("shape of theta vector should be 1 X d if shape" +
                             " of precision matrix of model is d X d")
        self.theta = theta
        self.model = model
        # The gradient of probability distribution at theta
        self.grad_theta = None
        # The gradient log of probability distribution at theta
        self.log_grad_theta = None

    def get_gradient_log_pdf(self):
        return self.grad_theta, self.log_grad_theta


class GradLogPDFGaussian(GradientLogPDF):
    """
    Class for finding gradient and gradient log of distribution
    Inherits pgmpy.inference.base_continuous.GradientLogPDF
    """

    def __init__(self, theta, model):
        GradientLogPDF.__init__(self, theta, model)
        self.grad_theta, self.log_grad_theta = self._get_gradient_log_pdf()

    def _get_gradient_log_pdf(self):
        """
        Method that finds gradient and its log at theta
        """
        grad = - self.theta * self.model.precision_matrix
        log_grad = 0.5 * grad * self.theta.transpose()

        return grad, log_grad


class DiscretizeTime(object):
    """
    Base class for Discretizing continuous time in sumilatin dynamics

    Classes inheriting this base class can be passed as an argument for
    discretizing time in inference algorithms

    Parameters
    ----------
    theta: A vector (row matrix or 1d array like object) of shape 1 X d
           Vector representing the proposed value for the
           distribution parameter theta (position X)

    model : An instance of AbstractGaussian

    grad_log_pdf : A callable object, an instance of
                   pgmpy.inference.base_continuous.GradientLogPDF

    momentum: A vector (row matrix or 1d array like object) of shape 1 X d
              Vector representing the proposed value for momentum
              (velocity)

    epsilon: Float
             step size for the time splitting

    Examples
    --------
    """

    def __init__(self, grad_log_pdf, model, theta, momentum, epsilon):
        # TODO: Take model instead of precision matrix
        if not isinstance(theta, np.matrix):
            if isinstance(theta, (np.ndarray, list, tuple, set, frozenset)):
                length = len(theta)
                theta = np.matrix(theta)
                theta = np.reshape(theta, (1, length))
            else:
                raise TypeError("theta should be a 1d array type object")

        if not isinstance(momentum, np.matrix):
            if isinstance(momentum, (np.ndarray, list, tuple, set, frozenset)):
                length = len(momentum)
                momentum = np.matrix(momentum)
                momentum = np.reshape(momentum, (1, length))
            else:
                raise TypeError("momentum should be a 1d array type object")

        if not isinstance(grad_log_pdf, GradientLogPDF):
            raise TypeError("grad_log_pdf must be an instance" +
                            " of pgmpy.inference.base_continuous.GradientLogPDF")

        if theta.shape != momentum.shape:
            raise ValueError("Shape of theta and momentum must be same")

        if theta.shape[1] != model.precision_matrix.shape[0]:
            raise ValueError("shape of theta vector should be 1 X d if shape" +
                             " of precision matrix of model is d X d")

        self.theta = theta
        self.momentum = momentum
        self.epsilon = epsilon
        self.model = model
        self.grad_log_pdf = grad_log_pdf
        # The new proposed value of theta after taking
        # epsilon step size in time
        self.theta_bar = None
        # The new proposed value of momentum after taking
        # epsilon step size in time
        self.momentum_bar = None

    def discretize_time(self):
        return self.theta_bar, self.momentum_bar


class LeapFrog(SplitTime):
    """
    Class for performing discretization(in time) using leapfrog method
    Inherits pgmpy.inference.base_continuous.DiscretizeTime
    """

    def __init__(self, grad_log_pdf, model, theta, momentum, epsilon):

        SplitTime.__init__(grad_log_pdf, model, theta, momentum, epsilon)

        self.theta_bar, self.momentum_bar, self.grad_theta,\
            self.log_grad_theta = self._split_time()

    def _discretize_time(self):
        """
        Method to perform time splitting using leapfrog
        """
        grad_theta, log_grad_theta = self.grad_log_pdf(self.theta,
                                                       self.model).get_gradient_log_pdf()
        # Take half step in time for updating momentum
        momentum_bar = self.momentum + 0.5 * self.epsilon * log_grad_theta
        # Take full step in time for updating position theta
        theta_bar = self.theta + self.epsilon * momentum_bar

        grad_theta, log_grad_theta = self.grad_log_pdf(theta_bar,
                                                       self.model).get_gradient_log_pdf()
        # Take remaining half step in time for updating momentum
        momentum_bar = momentum_bar + 0.5 * self.epsilon * log_grad_theta

        return theta_bar, momentum_bar, grad_theta, log_grad_theta


class EulerMethod(SplitTime):
    """
    Class for performing splitting in time using Modified euler method
    Inherits pgmpy.inference.base_continuous.DiscretizeTime
    """

    def __init__(self, grad_log_pdf, model, theta, momentum, epsilon):

        SplitTime.__init__(grad_log_pdf, model, theta, momentum, epsilon)

        self.theta_bar, self.momentum_bar, self.grad_theta,\
            self.log_grad_theta = self._discretize_time()

    def _discretize_time(self):
        """
        Method to perform time splitting using Modified euler method
        """
        grad_theta, log_grad_theta = self.grad_log_pdf(self.theta,
                                                       self.model).get_gradient_log_pdf()
        # Take full step in time and update momentum
        momentum_bar = self.momentum + self.epsilon * log_grad_theta
        # Take full step in time and update position
        theta_bar = self.theta + self.epsilon * momentum_bar

        return theta_bar, momentum_bar, grad_theta, log_grad_theta


class BaseHMC(object):
    """
    Base Class for HMC type inference

    Parameters:
    -----------
    model: An instance of AbstractGaussian

    grad_log_pdf: A instance of pgmpy.inference.base_continuous.GradientLogPDF

    discretize_time: A instance of pgmpy.inference.base_continuous.DiscretizeTime
                     Defaults to LeapFrog

    delta: float (in between 0 and 1), defaults to 0.65
           The target HMC acceptance probability
    """

    def __init__(self, model, grad_log_pdf,
                 split_time=LeapFrog, delta=0.65):
        # TODO: Use model instead of mean_vec and cov_matrix

        if not isinstance(grad_log_pdf, GradientLogPDF):
            raise TypeError("grad_log_pdf must be an instance of" +
                            "pgmpy.inference.base_continuous.GradientLogPDF")

        if not isinstance(split_time, SplitTime):
            raise TypeError("split_time must be an instance of" +
                            "pgmpy.inference.base_continuous.SplitTime")

        if not isinstance(delta, float) or delta > 1.0 or delta < 0.0:
            raise AttributeError(
                "delta should be a floating value in between 0 and 1")

        self.model = model
        self.delta = delta
        self.grad_log_pdf = grad_log_pdf
        self.split_time = split_time

        def prob_distribution(self, )
