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
    mean_vec: 1d array type of structure
              Represents the mean of the distribution

    cov_matrix: A numpy.array or numpy.matrix of size len(mean_vec) x len(mean_vec),
                Covariance matrix for the distribution.

    """

    def __init__(self, mean_vec, cov_matrix):

        if isinstance(mean_vec, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            mean_vec = np.array(mean_vec).flatten()
        else:
            raise TypeError("mean_vec should be a 1d array type object")
        mean_vec = np.reshape(mean_vec, (len(mean_vec), 1))

        if not isinstance(cov_matrix, (np.matrix, np.ndarray, list)):
            raise TypeError(
                "cov_matrix must be numpy.matrix type object")

        if cov_matrix.shape[0] != cov_matrix.shape[1]:
            raise ValueError(
                "cov_matrix must be a square matrix")

        if mean_vec.shape[0] != cov_matrix.shape[0]:
            raise ValueError("shape of mean vector should be d X 1 and" +
                             " shape of covariance matrix should be d X d")

        self.mean_vec = mean_vec
        self.cov_matrix = np.array(cov_matrix)
        self.precision_matrix = np.linalg.inv(cov_matrix)


class BaseGradLogPDF(object):
    """
    Base class for gradient log of probability density function/ distribution

    Classes inheriting this base class can be passed as an argument for
    finding gradient log of probability distribution in inference algorithms

    The class should initialize the gradient of log P(X) and log P(X)
    (Constant terms can be ignored)

    Parameters
    ----------
    position : A 1d array type object
               Vector representing values of parameter position(theta or X)

    model : An instance of pgmpy.models.Continuous

    Examples
    --------
    >>> # Example shows how to use the containGer class
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference import BaseGradLogPDF
    >>> import numpy as np
    >>> class GradLogGaussian(BaseGradLogPDF):
    ...     def __init__(self, position, model):
    ...         BaseGradLogPDF.__init__(self, position, model)
    ...         self.grad_log, self.log_pdf = self._get_gradient_log_pdf()
    ...     def _get_gradient_log_pdf(self):
    ...         sub_vec = self.position - self.model.mean_vec
    ...         grad = - np.dot(self.model.precision_matrix, sub_vec)
    ...         log_pdf = 0.5 * np.float(np.dot(sub_vec.T, grad))
    ...         return grad, log_pdf
    >>> mean_vec = np.array([[1], [1]])
    >>> cov_matrix = np.array([[1, 0.2], [0.2, 7]])
    >>> model = JointGaussianDistribution(mean_vec, cov_matrix)
    >>> pos = np.array([[0.1], [0.9]])
    >>> grad_logp, logp = GradLogGaussian(pos, model).get_gradient_log_pdf()
    >>> logp
    -0.4054597701149426
    >>> grad_logp
    array([[ 0.90229885],
           [-0.01149425]])
    """

    def __init__(self, position, model):
        # TODO: Take model as parameter instead of precision_matrix

        if isinstance(position, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            position = np.array(position).flatten()
            position = np.reshape(position, (len(position), 1))
        else:
            raise TypeError("position should be a 1d array type object")

        if position.shape[0] != model.precision_matrix.shape[0]:
            raise ValueError("shape of position vector should be 1 X d if shape" +
                             " of precision matrix of model is d X d")
        self.position = position
        self.model = model
        # The gradient of probability distribution at position
        self.grad_log = None
        # The gradient log of probability distribution at position
        self.log_pdf = None

    def get_gradient_log_pdf(self):
        """
        Returns the gradient log and log of model at given position

        Returns
        -------
        numpy.array: Representing gradient log of model at given position
        float: Representing log of model at given position

        Example
        --------
        >>> # Using implementation of GradLogPDFGaussian
        >>> from pgmpy.inference import GradLogPDFGaussian
        >>> from pgmpy.models import JointGaussianDistribution
        >>> import numpy as np
        >>> mean_vec = np.array([[1], [1]])
        >>> cov_matrix = np.array([[1, 0.2], [0.2, 7]])
        >>> model = JointGaussianDistribution(mean_vec, cov_matrix)
        >>> pos = np.array([[0.1], [0.9]])
        >>> grad_logp, logp = GradLogPDFGaussian(pos, model).get_gradient_log_pdf()
        >>> logp
        -0.4054597701149426
        >>> grad_logp
        array([[ 0.90229885],
               [-0.01149425]])
        """
        return self.grad_log, self.log_pdf


class GradLogPDFGaussian(BaseGradLogPDF):
    """
    Class for finding gradient and gradient log of distribution
    Inherits pgmpy.inference.base_continuous.BaseGradLogPDF
    Here model must be an instance of JointGaussianDistribution

    Example
    -------
    >>> from pgmpy.inference import GradLogPDFGaussian
    >>> from pgmpy.models import JointGaussianDistribution
    >>> import numpy as np
    >>> mean_vec = np.array([[1], [1]])
    >>> cov_matrix = np.array([[1, 0.2], [0.2, 7]])
    >>> model = JointGaussianDistribution(mean_vec, cov_matrix)
    >>> pos = np.array([[0.1], [0.9]])
    >>> grad_logp, logp = GradLogPDFGaussian(pos, model).get_gradient_log_pdf()
    >>> logp
    -0.4054597701149426
    >>> grad_logp
    array([[ 0.90229885],
           [-0.01149425]])
    """

    def __init__(self, position, model):
        BaseGradLogPDF.__init__(self, position, model)
        self.grad_log, self.log_pdf = self._get_gradient_log_pdf()

    def _get_gradient_log_pdf(self):
        """
        Method that finds gradient and its log at position
        """
        sub_vec = self.position - self.model.mean_vec
        grad = - np.dot(self.model.precision_matrix, sub_vec)
        log_pdf = 0.5 * np.float(np.dot(sub_vec.T, grad))

        return grad, log_pdf


class BaseDiscretizeTime(object):
    """
    Base class for Discretizing continuous time in sumilatin dynamics

    Classes inheriting this base class can be passed as an argument for
    discretizing time in inference algorithms

    Parameters
    ----------
    grad_log_pdf : A subclass of pgmpy.inference.base_continuous.BaseGradLogPDF

    model : An instance of pgmpy.models.Continuous
            Model for which DiscretizeTime object is initialized

    position : A 1d array like object
               Vector representing values of parameter position( or X)

    momentum: A 1d array like object
              Vector representing the proposed value for momentum
              (velocity)

    epsilon: Float
             step size for the time splitting

    Examples
    --------
    >>> # Example shows how to use the container class
    >>> from pgmpy.inference import BaseDiscretizeTime
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference import GradLogPDFGaussian
    >>> class ModifiedEuler(BaseDiscretizeTime):
    ...     def __init__(self, grad_log_pdf, model, position, momentum, epsilon):
    ...         BaseDiscretizeTime.__init__(self, grad_log_pdf, model, position, momentum, epsilon)
    ...         self.position_bar, self.momentum_bar, self.grad_log_position,\
    ...                 self.log_pdf = self._discretize_time()
    ...     def _discretize_time(self):
    ...         grad_log_position, log_pdf_position =\
    ...                 self.grad_log_pdf(self.position,
    ...                                   self.model).get_gradient_log_pdf()
    ...         momentum_bar = self.momentum + self.epsilon * grad_log_position
    ...         position_bar = self.position + self.epsilon * momentum_bar
    ...         return position_bar, momentum_bar, grad_log_position, log_pdf_position
    >>> import numpy as np
    >>> pos = np.array([[1], [2]])
    >>> momentum = np.array([[0], [0]])
    >>> mean_vec = np.array([[0], [0]])
    >>> cov_matrix = np.eye(2)
    >>> model = JointGaussianDistribution(mean_vec, cov_matrix)
    >>> new_pos, new_momentum = ModifiedEuler(GradLogPDFGaussian, model, pos, momentum, 0.25).discretize_time()
    >>> new_pos
    array([[ 0.9375],
           [ 1.875 ]])
    >>> new_momentum
    array([[-0.25],
           [-0.5 ]])
    """

    def __init__(self, grad_log_pdf, model, position, momentum, epsilon):
        # TODO: Take model instead of precision matrix
        if isinstance(position, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            position = np.array(position).flatten()
            position = np.reshape(position, (len(position), 1))
        else:
            raise TypeError("position should be a 1d array type object")

        if isinstance(momentum, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            momentum = np.array(momentum).flatten()
            momentum = np.reshape(momentum, (len(momentum), 1))
        else:
            raise TypeError("position should be a 1d array type object")

        if not issubclass(grad_log_pdf, BaseGradLogPDF):
            raise TypeError("grad_log_pdf must be an instance" +
                            " of pgmpy.inference.base_continuous.BaseGradLogPDF")

        if position.shape != momentum.shape:
            raise ValueError("Shape of position and momentum must be same")

        if position.shape[0] != model.precision_matrix.shape[0]:
            raise ValueError("shape of position vector should be 1 X d if shape" +
                             " of precision matrix of model is d X d")

        self.position = position
        self.momentum = momentum
        self.epsilon = epsilon
        self.model = model
        self.grad_log_pdf = grad_log_pdf
        # The new proposed value of position after taking
        # epsilon step size in time
        self.position_bar = None
        # The new proposed value of momentum after taking
        # epsilon step size in time
        self.momentum_bar = None

    def discretize_time(self):
        """
        Returns the new proposed values of position and momentum

        Returns
        -------
        numpy.array: New proposed value of position
        numpy.array: New proposed value of momentum

        Example
        -------
        >>> # Using implementation of ModifiedEuler
        >>> from pgmpy.inference import ModifiedEuler
        >>> from pgmpy.models import JointGaussianDistribution
        >>> from pgmpy.inference import GradLogPDFGaussian
        >>> import numpy as np
        >>> pos = np.array([[1], [2]])
        >>> momentum = np.array([[0], [0]])
        >>> mean_vec = np.array([[0], [0]])
        >>> cov_matrix = np.eye(2)
        >>> model = JointGaussianDistribution(mean_vec, cov_matrix)
        >>> new_pos, new_momentum = ModifiedEuler(GradLogPDFGaussian, model, pos, momentum, 0.25).discretize_time()
        >>> new_pos
        array([[ 0.9375],
               [ 1.875 ]])
        >>> new_momentum
        array([[-0.25],
               [-0.5 ]])
        """
        return self.position_bar, self.momentum_bar


class LeapFrog(BaseDiscretizeTime):
    """
    Class for performing discretization(in time) using leapfrog method
    Inherits pgmpy.inference.base_continuous.BaseDiscretizeTime

    Example
    --------
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference import GradLogPDFGaussian, LeapFrog
    >>> import numpy as np
    >>> pos = np.array([[2], [1]])
    >>> momentum = np.array([[1], [1]])
    >>> mean_vec = np.array([[0], [0]])
    >>> cov_matrix = np.eye(2)
    >>> model = JointGaussianDistribution(mean_vec, cov_matrix)
    >>> new_pos, new_momentum = LeapFrog(GradLogPDFGaussian, model, pos, momentum, 0.25).discretize_time()
    >>> new_pos
    array([[ 2.1875 ],
           [ 1.21875]])
    >>> new_momentum
    array([[ 0.4765625 ],
           [ 0.72265625]])
    """

    def __init__(self, grad_log_pdf, model, position, momentum, epsilon):

        BaseDiscretizeTime.__init__(self, grad_log_pdf, model, position, momentum, epsilon)

        self.position_bar, self.momentum_bar, self.grad_log,\
            self.log_pdf = self._discretize_time()

    def _discretize_time(self):
        """
        Method to perform time splitting using leapfrog
        """
        grad_log, log_pdf = self.grad_log_pdf(self.position,
                                              self.model).get_gradient_log_pdf()
        # Take half step in time for updating momentum
        momentum_bar = self.momentum + 0.5 * self.epsilon * grad_log
        # Take full step in time for updating position position
        position_bar = self.position + self.epsilon * momentum_bar

        grad_log, log_pdf = self.grad_log_pdf(position_bar,
                                              self.model).get_gradient_log_pdf()
        # Take remaining half step in time for updating momentum
        momentum_bar = momentum_bar + 0.5 * self.epsilon * grad_log

        return position_bar, momentum_bar, grad_log, log_pdf


class ModifiedEuler(BaseDiscretizeTime):
    """
    Class for performing splitting in time using Modified euler method
    Inherits pgmpy.inference.base_continuous.BaseDiscretizeTime

    Example
    --------
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference import GradLogPDFGaussian, ModifiedEuler
    >>> import numpy as np
    >>> pos = np.array([[2], [1]])
    >>> momentum = np.array([[1], [1]])
    >>> mean_vec = np.array([[0], [0]])
    >>> cov_matrix = np.eye(2)
    >>> model = JointGaussianDistribution(mean_vec, cov_matrix)
    >>> new_pos, new_momentum = ModifiedEuler(GradLogPDFGaussian, model, pos, momentum, 0.25).discretize_time()
    >>> new_pos
    array([[ 2.125 ],
           [ 1.1875]])
    >>> new_momentum
    array([[ 0.5 ],
           [ 0.75]])
    """

    def __init__(self, grad_log_pdf, model, position, momentum, epsilon):

        BaseDiscretizeTime.__init__(self, grad_log_pdf, model, position, momentum, epsilon)

        self.position_bar, self.momentum_bar, self.grad_log,\
            self.log_pdf = self._discretize_time()

    def _discretize_time(self):
        """
        Method to perform time splitting using Modified euler method
        """
        grad_log, log_pdf = self.grad_log_pdf(self.position,
                                              self.model).get_gradient_log_pdf()
        # Take full step in time and update momentum
        momentum_bar = self.momentum + self.epsilon * grad_log
        # Take full step in time and update position
        position_bar = self.position + self.epsilon * momentum_bar

        return position_bar, momentum_bar, grad_log, log_pdf
