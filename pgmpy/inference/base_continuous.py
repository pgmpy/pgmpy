"""
    A collection of base class objects for continuous model
"""

import numpy as np


class GradientLogPDF(object):
    """
    Base class for gradient log of probability density function/ distribution

    Classes inheriting this base class can be passed as an argument for
    finding gradient log of probability distribution in inference algorithms

    Parameters
    ----------
    theta : A 1d array type object or a row matrix of shape 1 X d
            Vector representing values of parameter theta( or X)

    precision_matrix : numpy.matrix type object of shape d X d
                       Inverese of covariance matrix of the distribution

    Examples
    --------
    """

    def __init__(self, theta, precision_matrix):
        # TODO: Take model as parameter instead of precision_matrix
        if not isinstance(theta, np.matrix):
            if isinstance(theta, (np.ndarray, list, tuple, set, frozenset)):
                theta = np.matrix(theta)
            else:
                raise TypeError("theta should be a 1d array type object")

        if not isinstance(precision_matrix, np.matrix):
            raise TypeError(
                "precision_matrix must be numpy.matrix type object")

        if precision_matrix.shape[0] != precision_matrix.shape[1]:
            raise ValueError(
                "precision_matrix must be a square matrix")

        if theta.shape[1] != precision_matrix.shape[0]:
            raise ValueError("shape of theta vector should be 1 X d and" +
                             " shape of precision_matrix should be d X d")
        self.theta = theta
        self.precision_matrix = precision_matrix
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

    def __init__(self, theta, precision_matrix):
        GradientLogPDF.__init__(self, theta, precision_matrix)
        self.grad_theta, self.log_grad_theta = self._get_gradient_log_pdf()

    def _get_gradient_log_pdf(self):
        """
        Method that finds gradient and its log at theta
        """
        grad = - self.theta * self.precision_matrix
        log_grad = 0.5 * grad * self.theta.transpose()

        return grad, log_grad


class SplitTime(object):
    """
    Base class for time splitting

    Classes inheriting this base class can be passed as an argument for
    splitting time in inference algorithms

    Parameters
    ----------
    grad_log_pdf : A callable object, an instance of
                   pgmpy.inference.base_continuous.GradientLogPDF

    precision_matrix : numpy.matrix type object of shape d X d
                       Inverse of covariance matrix of the distribution

    theta: A vector (row matrix or 1d array like object) of shape 1 X d
           Vector representing the proposed value for the
           distribution parameter theta (position X)

    momentum: A vector (row matrix or 1d array like object) of shape 1 X d
              Vector representing the proposed value for momentum
              (velocity)

    epsilon: Float
             step size for the time splitting

    Examples
    --------
    """

    def __init__(self, grad_log_pdf, precision_matrix, theta, momentum, epsilon):
        # TODO: Take model instead of precision matrix
        if not isinstance(theta, np.matrix):
            if isinstance(theta, (np.ndarray, list, tuple, set, frozenset)):
                theta = np.matrix(theta)
            else:
                raise TypeError("theta should be a 1d array type object")

        if not isinstance(momentum, np.matrix):
            if isinstance(momentum, (np.ndarray, list, tuple, set, frozenset)):
                momentum = np.matrix(momentum)
            else:
                raise TypeError("momentum should be a 1d array type object")

        if not isinstance(grad_log_pdf, GradientLogPDF):
            raise TypeError("grad_log_pdf must be an instance" +
                            " of pgmpy.inference.base_continuous.GradientLogPDF")

        if not isinstance(precision_matrix, np.matrix):
            raise TypeError(
                "precision_matrix must be numpy.matrix type object")

        if theta.shape != momentum.shape:
            raise ValueError("Shape of theta and momentum must be same")

        if theta.shape[1] != precision_matrix.shape[0]:
            raise ValueError("shape of theta vector should be 1 X d and" +
                             " shape of precision_matrix should be d X d")

        self.theta = theta
        self.momentum = momentum
        self.epsilon = epsilon
        self.precision_matrix = precision_matrix
        self.grad_log_pdf = grad_log_pdf
        # The new proposed value of theta after taking
        # epsilon step size in time
        self.theta_bar = None
        # The new proposed value of momentum after taking
        # epsilon step size in time
        self.momentum_bar = None

    def split_time(self):
        return self.theta_bar, self.momentum_bar


class LeapFrog(SplitTime):
    """
    Class for performing splitting in time using leapfrog method
    Inherits pgmpy.inference.base_continuous.SplitTime
    """

    def __init__(self, grad_log_pdf, precision_matrix, theta, momentum, epsilon):

        SplitTime.__init__(grad_log_pdf,
                           precision_matrix, theta, momentum, epsilon)

        self.theta_bar, self.momentum_bar, self.grad_theta,\
            self.log_grad_theta = self._split_time()

    def _split_time(self):
        """
        Method to perform time splitting using leapfrog
        """
        grad_theta, log_grad_theta = self.grad_log_pdf(
            self.theta, self.precision_matrix).get_gradient_log_pdf()
        # Take half step in time for updating momentum
        momentum_bar = self.momentum + 0.5 * self.epsilon * log_grad_theta
        # Take full step in time for updating position theta
        theta_bar = self.theta + self.epsilon * momentum_bar

        grad_theta, log_grad_theta = self.grad_log_pdf(
            theta_bar, self.precision_matrix).get_gradient_log_pdf()
        # Take remaining half step in time for updating momentum
        momentum_bar = momentum_bar + 0.5 * self.epsilon * log_grad_theta

        return theta_bar, momentum_bar, grad_theta, log_grad_theta


class EulerMethod(SplitTime):
    """
    Class for performing splitting in time using Modified euler method
    Inherits pgmpy.inference.base_continuous.SplitTime
    """

    def __init__(self, grad_log_pdf, precision_matrix, theta, momentum, epsilon):

        SplitTime.__init__(grad_log_pdf,
                           precision_matrix, theta, momentum, epsilon)

        self.theta_bar, self.momentum_bar, self.grad_theta,\
            self.log_grad_theta = self._split_time()

    def _split_time(self):
        """
        Method to perform time splitting using Modified euler method
        """
        grad_theta, log_grad_theta = self.grad_log_pdf(
            self.theta, self.precision_matrix).get_gradient_log_pdf()
        # Take full step in time and update momentum
        momentum_bar = self.momentum + self.epsilon * log_grad_theta
        # Take full step in time and update position
        theta_bar = self.theta + self.epsilon * momentum_bar

        return theta_bar, momentum_bar, grad_theta, log_grad_theta
