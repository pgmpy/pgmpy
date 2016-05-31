"""
    A collection of base class objects for continuous model
"""

import numpy as np


class JointGaussianDistribution(object):
    """
    A naive gaussian container class.(Depricated)
    Will be removed when Multivariate Distributions will be implemented

    Paramters
    ---------
    mean_vec: A vector (row matrix or a column matrix or a 1d array type structure)
              Represents the mean of the distribution
              Will be converted into a column matrix of appropriate shape

    cov_matrix: A numpy.matrix of size len(mean_vec) x len(mean_vec),
                Covariance matrix for the distribution.

    """

    def __init__(self, mean_vec, cov_matrix):

        if isinstance(mean_vec, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            mean_vec = np.array(mean_vec).flatten()
        else:
            raise TypeError("mean_vec should be a 1d array type object")
        mean_vec = np.matrix(np.reshape(mean_vec, (len(mean_vec), 1)))

        if not isinstance(cov_matrix, np.matrix):
            raise TypeError(
                "cov_matrix must be numpy.matrix type object")

        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError(
                "cov_matrix must be a square matrix")

        if mean_vec.shape[0] != cov_matrix.shape[0]:
            raise ValueError("shape of mean vector should be d X 1 and" +
                             " shape of covariance matrix should be d X d")

        self.mean_vec = mean_vec
        self.cov_matrix = cov_matrix
        self.precision_matrix = np.linalg.inv(cov_matrix)


class GradientLogPDF(object):
    """
    Base class for gradient log of probability density function/ distribution

    Classes inheriting this base class can be passed as an argument for
    finding gradient log of probability distribution in inference algorithms

    The class should initialize the gradient of log P(X) and log P(X)
    (Constant terms can be ignored)

    Parameters
    ----------
    theta : A 1d array type object or a row matrix of shape 1 X d
            or d X 1.(Will be converted to d X 1)
            Vector representing values of parameter theta( or X)

    model : An instance of pgmpy.models.Continuous

    Examples
    --------
    >>> # Example shows how to use the container class
    >>> from pgmpy.inference import JointGuassianDistribution as JGD
    >>> from pgmpy.inference import GradientLogPDF as GLP
    >>> import numpy as np
    >>> class GradientLogGaussian(GLP):
    ...     def __init__(self, X, model):
    ...         GLP.__init__(self, X, model)
    ...         self.grad_log_theta, self.log_pdf_theta = self._get_gradient_log_pdf()
    ...     def _get_gradient_log_pdf(self):
    ...         sub_vec = self.theta - self.model.mean_vec
    ...         grad = - self.model.precision_matrix * sub_vec
    ...         log_pdf = 0.5 * sub_vec.transpose * grad
    ...         return grad, log_pdf
    """

    def __init__(self, theta, model):
        # TODO: Take model as parameter instead of precision_matrix

        if isinstance(theta, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            theta = np.array(theta).flatten()
            theta = np.matrix(np.reshape(theta, (len(theta), 1)))
        else:
            raise TypeError("theta should be a 1d array type object")

        if theta.shape[0] != model.precision_matrix.shape[0]:
            raise ValueError("shape of theta vector should be 1 X d if shape" +
                             " of precision matrix of model is d X d")
        self.theta = theta
        self.model = model
        # The gradient of probability distribution at theta
        self.grad_log = None
        # The gradient log of probability distribution at theta
        self.log_pdf = None

    def get_gradient_log_pdf(self):
        return self.grad_log, self.log_pdf


class GradientLogPDFGaussian(GradientLogPDF):
    """
    Class for finding gradient and gradient log of distribution
    Inherits pgmpy.inference.base_continuous.GradientLogPDF
    Here model must be an instance of JointGaussianDistribution
    """

    def __init__(self, theta, model):
        GradientLogPDF.__init__(self, theta, model)
        self.grad_log, self.log_pdf = self._get_gradient_log_pdf()

    def _get_gradient_log_pdf(self):
        """
        Method that finds gradient and its log at theta
        """
        sub_vec = self.theta - self.model.mean_vec
        grad = - np.dot(self.model.precision_matrix, sub_vec)
        log_pdf = 0.5 * np.float(np.dot(sub_vec.T, grad))

        return grad, log_pdf


class DiscretizeTime(object):
    """
    Base class for Discretizing continuous time in sumilatin dynamics

    Classes inheriting this base class can be passed as an argument for
    discretizing time in inference algorithms

    Parameters
    ----------
    theta : A 1d array type object or a row matrix of shape 1 X d
            or d X 1.(Will be converted to d X 1)
            Vector representing values of parameter theta( or X)

    model : An instance of pgmpy.models.Continuous

    grad_log_pdf : A subclass of pgmpy.inference.base_continuous.GradientLogPDF

    momentum: A vector (row matrix or 1d array like object) of shape 1 X d
              Vector representing the proposed value for momentum
              (velocity)

    epsilon: Float
             step size for the time splitting

    Examples
    --------
    >>> # Example shows how to use the container class
    >>> from pgmpy.inference import DiscretizeTime
    >>> from pgmpy.inference import GradLogPDFGaussian
    >>> class ModifiedEuler(DiscretizeTime):
    ...     def __init__(self, theta, model, grad_log_pdf, momentum, epsilon):
    ...         DiscretizeTime.__init__(theta, model, grad_log_pdf, momentum, epsilon)
    ...         self.theta_bar, self.momentum_bar, self.grad_log_theta,\
    ...                 self.log_pdf = self._discretize_time()
    ...     def _discretize_time(self):
    ...         grad_log_theta, log_pdf_theta =\
    ...                 self.grad_log_pdf(self.theta,
    ...                                   self.model).get_gradient_log_pdf()
    ...         momentum_bar = self.momentum + self.epsilon * grad_log_theta
    ...         theta_bar = self.theta + self.epsilon * momentum_bar
    ...        return theta_bar, momentum_bar, grad_log_theta, log_pdf_theta
    """

    def __init__(self, grad_log_pdf, model, theta, momentum, epsilon):
        # TODO: Take model instead of precision matrix
        if isinstance(theta, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            theta = np.array(theta).flatten()
            theta = np.matrix(np.reshape(theta, (len(theta), 1)))
        else:
            raise TypeError("theta should be a 1d array type object")

        if isinstance(momentum, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            momentum = np.array(momentum).flatten()
            momentum = np.matrix(np.reshape(momentum, (len(momentum), 1)))
        else:
            raise TypeError("theta should be a 1d array type object")

        if not issubclass(grad_log_pdf, GradientLogPDF):
            raise TypeError("grad_log_pdf must be an instance" +
                            " of pgmpy.inference.base_continuous.GradientLogPDF")

        if theta.shape != momentum.shape:
            raise ValueError("Shape of theta and momentum must be same")

        if theta.shape[0] != model.precision_matrix.shape[0]:
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


class LeapFrog(DiscretizeTime):
    """
    Class for performing discretization(in time) using leapfrog method
    Inherits pgmpy.inference.base_continuous.DiscretizeTime
    """

    def __init__(self, grad_log_pdf, model, theta, momentum, epsilon):

        DiscretizeTime.__init__(self, grad_log_pdf, model, theta, momentum, epsilon)

        self.theta_bar, self.momentum_bar, self.grad_log,\
            self.log_pdf = self._discretize_time()

    def _discretize_time(self):
        """
        Method to perform time splitting using leapfrog
        """
        grad_log, log_pdf = self.grad_log_pdf(self.theta,
                                              self.model).get_gradient_log_pdf()
        # Take half step in time for updating momentum
        momentum_bar = self.momentum + 0.5 * self.epsilon * grad_log
        # Take full step in time for updating position theta
        theta_bar = self.theta + self.epsilon * momentum_bar

        grad_log, log_pdf = self.grad_log_pdf(theta_bar,
                                              self.model).get_gradient_log_pdf()
        # Take remaining half step in time for updating momentum
        momentum_bar = momentum_bar + 0.5 * self.epsilon * grad_log

        return theta_bar, momentum_bar, grad_log, log_pdf


class ModifiedEuler(DiscretizeTime):
    """
    Class for performing splitting in time using Modified euler method
    Inherits pgmpy.inference.base_continuous.DiscretizeTime
    """

    def __init__(self, grad_log_pdf, model, theta, momentum, epsilon):

        DiscretizeTime.__init__(self, grad_log_pdf, model, theta, momentum, epsilon)

        self.theta_bar, self.momentum_bar, self.grad_log,\
            self.log_pdf = self._discretize_time()

    def _discretize_time(self):
        """
        Method to perform time splitting using Modified euler method
        """
        grad_log, log_pdf = self.grad_log_pdf(self.theta,
                                              self.model).get_gradient_log_pdf()
        # Take full step in time and update momentum
        momentum_bar = self.momentum + self.epsilon * grad_log
        # Take full step in time and update position
        theta_bar = self.theta + self.epsilon * momentum_bar

        return theta_bar, momentum_bar, grad_log, log_pdf


class BaseHMC(object):
    """
    Base Class for HMC type inference

    Parameters:
    -----------
    model: An instance of pgmpy.models.Continuous

    grad_log_pdf: A instance of pgmpy.inference.base_continuous.GradientLogPDF

    discretize_time: A instance of pgmpy.inference.base_continuous.DiscretizeTime
                     Defaults to LeapFrog

    delta: float (in between 0 and 1), defaults to 0.65
           The target HMC acceptance probability
    """

    def __init__(self, model, grad_log_pdf,
                 discretize_time=LeapFrog, delta=0.65):
        # TODO: Use model instead of mean_vec and cov_matrix

        if not issubclass(grad_log_pdf, GradientLogPDF):
            raise TypeError("grad_log_pdf must be an instance of" +
                            "pgmpy.inference.base_continuous.GradientLogPDF")

        if not issubclass(discretize_time, DiscretizeTime):
            raise TypeError("split_time must be an instance of" +
                            "pgmpy.inference.base_continuous.DiscretizeTime")

        if not isinstance(delta, float) or delta > 1.0 or delta < 0.0:
            raise AttributeError(
                "delta should be a floating value in between 0 and 1")

        self.model = model
        self.delta = delta
        self.grad_log_pdf = grad_log_pdf
        self.discretize_time = discretize_time
