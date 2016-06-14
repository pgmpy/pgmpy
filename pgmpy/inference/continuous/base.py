import numpy as np


class JointGaussianDistribution(object):
    """
    A naive gaussian container class.
    Will be removed when Multivariate Distributions will be implemented

    Paramters
    ---------
    mean: 1d array type of structure
        Represents the mean of the distribution

    covariance: A 2d array type object of size len(mean) x len(mean),
        Covariance of the distribution.

    """

    def __init__(self, mean, covariance):

        if isinstance(mean, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            mean = np.array(mean).flatten()
        else:
            raise TypeError("mean should be a 1d array type object")
        mean = np.reshape(mean, (len(mean), 1))

        if not isinstance(covariance, (np.matrix, np.ndarray, list)):
            raise TypeError(
                "covariance must be a 2d array type object")
        covariance = np.array(covariance)
        if covariance.shape[0] != covariance.shape[1]:
            raise ValueError(
                "covariance must be a square in shape")

        if mean.shape[0] != covariance.shape[0]:
            raise ValueError("shape of mean vector should be d X 1 and" +
                             " shape of covariance matrix should be d X d")

        self.mean = mean
        self.covariance = covariance
        self.precision_matrix = np.linalg.inv(covariance)


class BaseGradLogPDF(object):
    """
    Base class for evaluating gradient log of probability density function/ distribution

    Classes inheriting this base class can be passed as an argument for
    finding gradient log of probability distribution in inference algorithms

    The class should initialize  self.grad_log and self.log_pdf

    Parameters
    ----------
    distribution_param : A 1d array type object
        Vector representing values of distribution parameter at which we want to find gradient and log

    model : An instance of pgmpy.models.Continuous

    Examples
    --------
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference import BaseGradLogPDF
    >>> import numpy as np
    >>> class GradLogGaussian(BaseGradLogPDF):
    ...     def __init__(self, position, model):
    ...         BaseGradLogPDF.__init__(self, position, model)
    ...         self.grad_log, self.log_pdf = self._get_gradient_log_pdf()
    ...     def _get_gradient_log_pdf(self):
    ...         sub_vec = self.position - self.model.mean
    ...         grad = - np.dot(self.model.precision_matrix, sub_vec)
    ...         log_pdf = 0.5 * np.float(np.dot(sub_vec.T, grad))
    ...         return grad, log_pdf
    >>> mean = np.array([[1], [1]])
    >>> covariance = np.array([[1, 0.2], [0.2, 7]])
    >>> model = JointGaussianDistribution(mean, covariance)
    >>> dist_param = np.array([[0.1], [0.9]])
    >>> grad_logp, logp = GradLogGaussian(dist_param, model).get_gradient_log_pdf()
    >>> logp
    -0.4054597701149426
    >>> grad_logp
    array([[ 0.90229885],
           [-0.01149425]])
    """

    def __init__(self, distribution_param, model):

        if isinstance(distribution_param, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            distribution_param = np.array(distribution_param).flatten()
            distribution_param = np.reshape(distribution_param, (len(distribution_param), 1))
        else:
            raise TypeError("distribution_param should be a 1d array type object")

        self.distribution_param = distribution_param
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
        >>> mean = np.array([[1], [1]])
        >>> covariance = np.array([[1, 0.2], [0.2, 7]])
        >>> model = JointGaussianDistribution(mean, covariance)
        >>> dist_param = np.array([[0.1], [0.9]])
        >>> grad_logp, logp = GradLogPDFGaussian(dist_param, model).get_gradient_log_pdf()
        >>> logp
        -0.4054597701149426
        >>> grad_logp
        array([[ 0.90229885],
               [-0.01149425]])
        """
        return self.grad_log, self.log_pdf


class GradLogPDFGaussian(BaseGradLogPDF):
    """
    Class for finding gradient and gradient log of Joint Gaussian Distribution
    Inherits pgmpy.inference.base_continuous.BaseGradLogPDF
    Here model must be an instance of JointGaussianDistribution

    Example
    -------
    >>> from pgmpy.inference import GradLogPDFGaussian
    >>> from pgmpy.models import JointGaussianDistribution
    >>> import numpy as np
    >>> mean = np.array([[1], [1]])
    >>> covariance = np.array([[1, 0.2], [0.2, 7]])
    >>> model = JointGaussianDistribution(mean, covariance)
    >>> dist_param = np.array([[0.1], [0.9]])
    >>> grad_logp, logp = GradLogPDFGaussian(dist_param, model).get_gradient_log_pdf()
    >>> logp
    -0.4054597701149426
    >>> grad_logp
    array([[ 0.90229885],
           [-0.01149425]])
    """

    def __init__(self, distribution_param, model):
        BaseGradLogPDF.__init__(self, distribution_param, model)
        self.grad_log, self.log_pdf = self._get_gradient_log_pdf()

    def _get_gradient_log_pdf(self):
        """
        Method that finds gradient and its log at position
        """
        sub_vec = self.distribution_param - self.model.mean
        grad = - np.dot(self.model.precision_matrix, sub_vec)
        log_pdf = 0.5 * np.float(np.dot(sub_vec.T, grad))

        return grad, log_pdf


class BaseSimulateDynamics(object):
    """
    Base class for proposing new values of position and momentum by simulating Hamiltonian Dynamics.

    Classes inheriting this base class can be passed as an argument for
    simulate_dynamics in inference algorithms.

    Parameters
    ----------
    grad_log_pdf : A subclass of pgmpy.inference.base_continuous.BaseGradLogPDF

    model : An instance of pgmpy.models.Continuous
        Model for which DiscretizeTime object is initialized

    position : A 1d array like object
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    Examples
    --------
    >>> from pgmpy.inference import BaseSimulateDynamics
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference import GradLogPDFGaussian
    >>> # Class should initalize self.position_bar and self.momentum_bar
    >>> class ModifiedEuler(BaseSimulateDynamics):
    ...     def __init__(self, grad_log_pdf, model, position, momentum, stepsize):
    ...         BaseSimulateDynamics.__init__(self, grad_log_pdf, model, position, momentum, stepsize)
    ...         self.position_bar, self.momentum_bar, self.grad_logp, self.logp = self._get_proposed_values()
    ...     def _get_proposed_values(self):
    ...         grad_log_position, log_pdf_position = self.grad_log_pdf(self.position,
    ...                                                                 self.model).get_gradient_log_pdf()
    ...         momentum_bar = self.momentum + self.stepsize * grad_log_position
    ...         position_bar = self.position + self.stepsize * momentum_bar
    ...         return position_bar, momentum_bar, grad_log_position, log_pdf_position
    >>> import numpy as np
    >>> pos = np.array([[1], [2]])
    >>> momentum = np.array([[0], [0]])
    >>> mean = np.array([[0], [0]])
    >>> covariance = np.eye(2)
    >>> model = JointGaussianDistribution(mean, covariance)
    >>> new_pos, new_momentum = ModifiedEuler(GradLogPDFGaussian, model, pos, momentum, 0.25).get_proposed_values()
    >>> new_pos
    array([[ 0.9375],
           [ 1.875 ]])
    >>> new_momentum
    array([[-0.25],
           [-0.5 ]])
    """

    def __init__(self, grad_log_pdf, model, position, momentum, stepsize):

        if isinstance(position, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            position = np.array(position).flatten()
            position = np.reshape(position, (len(position), 1))
        else:
            raise TypeError("position should be a 1d array type object")

        if isinstance(momentum, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            momentum = np.array(momentum).flatten()
            momentum = np.reshape(momentum, (len(momentum), 1))
        else:
            raise TypeError("momentum should be a 1d array type object")

        if not issubclass(grad_log_pdf, BaseGradLogPDF):
            raise TypeError("grad_log_pdf must be an instance" +
                            " of pgmpy.inference.continuous.base.BaseGradLogPDF")

        if position.shape != momentum.shape:
            raise ValueError("Shape of position and momentum must be same")

        self.position = position
        self.momentum = momentum
        self.stepsize = stepsize
        self.model = model
        self.grad_log_pdf = grad_log_pdf
        # The new proposed value of position after taking
        # stepsize step size in time
        self.position_bar = None
        # The new proposed value of momentum after taking
        # stepsize step size in time
        self.momentum_bar = None

    def get_proposed_values(self):
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
        >>> mean = np.array([[0], [0]])
        >>> covariance = np.eye(2)
        >>> model = JointGaussianDistribution(mean, covariance)
        >>> new_pos, new_momentum = ModifiedEuler(GradLogPDFGaussian, model, pos, momentum, 0.25).get_proposed_values()
        >>> new_pos
        array([[ 0.9375],
               [ 1.875 ]])
        >>> new_momentum
        array([[-0.25],
               [-0.5 ]])
        """
        return self.position_bar, self.momentum_bar


class LeapFrog(BaseSimulateDynamics):
    """
    Class for simulating hamiltonian dynamics using leapfrog method
    Inherits pgmpy.inference.base_continuous.BaseSimulateDynamics

    Example
    --------
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference import GradLogPDFGaussian, LeapFrog
    >>> import numpy as np
    >>> pos = np.array([[2], [1]])
    >>> momentum = np.array([[1], [1]])
    >>> mean = np.array([[0], [0]])
    >>> covariance = np.eye(2)
    >>> model = JointGaussianDistribution(mean, covariance)
    >>> new_pos, new_momentum = LeapFrog(GradLogPDFGaussian, model, pos, momentum, 0.25).get_proposed_values()
    >>> new_pos
    array([[ 2.1875 ],
           [ 1.21875]])
    >>> new_momentum
    array([[ 0.4765625 ],
           [ 0.72265625]])
    """

    def __init__(self, grad_log_pdf, model, position, momentum, stepsize):

        BaseSimulateDynamics.__init__(self, grad_log_pdf, model, position, momentum, stepsize)

        self.position_bar, self.momentum_bar, self.grad_logp, self.logp = self._get_proposed_values()

    def _get_proposed_values(self):
        """
        Method to perform time splitting using leapfrog
        """
        grad_log, log_pdf = self.grad_log_pdf(self.position,
                                              self.model).get_gradient_log_pdf()
        # Take half step in time for updating momentum
        momentum_bar = self.momentum + 0.5 * self.stepsize * grad_log
        # Take full step in time for updating position position
        position_bar = self.position + self.stepsize * momentum_bar

        grad_log, log_pdf = self.grad_log_pdf(position_bar,
                                              self.model).get_gradient_log_pdf()
        # Take remaining half step in time for updating momentum
        momentum_bar = momentum_bar + 0.5 * self.stepsize * grad_log

        return position_bar, momentum_bar, grad_log, log_pdf


class ModifiedEuler(BaseSimulateDynamics):
    """
    Class for simulating Hamiltonian Dynamics using Modified euler method
    Inherits pgmpy.inference.base_continuous.BaseSimulateDynamics

    Example
    --------
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference.base_continuous import GradLogPDFGaussian, ModifiedEuler
    >>> import numpy as np
    >>> pos = np.array([[2], [1]])
    >>> momentum = np.array([[1], [1]])
    >>> mean = np.array([[0], [0]])
    >>> covariance = np.eye(2)
    >>> model = JointGaussianDistribution(mean, covariance)
    >>> new_pos, new_momentum = ModifiedEuler(GradLogPDFGaussian, model, pos, momentum, 0.25).get_proposed_values()
    >>> new_pos
    array([[ 2.125 ],
           [ 1.1875]])
    >>> new_momentum
    array([[ 0.5 ],
           [ 0.75]])
    """

    def __init__(self, grad_log_pdf, model, position, momentum, stepsize):

        BaseSimulateDynamics.__init__(self, grad_log_pdf, model, position, momentum, stepsize)

        self.position_bar, self.momentum_bar, self.grad_log, self.log_pdf = self._get_proposed_values()

    def _get_proposed_values(self):
        """
        Method to perform time splitting using Modified euler method
        """
        grad_log, log_pdf = self.grad_log_pdf(self.position,
                                              self.model).get_gradient_log_pdf()
        # Take full step in time and update momentum
        momentum_bar = self.momentum + self.stepsize * grad_log
        # Take full step in time and update position
        position_bar = self.position + self.stepsize * momentum_bar

        return position_bar, momentum_bar, grad_log, log_pdf
