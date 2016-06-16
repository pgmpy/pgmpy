import numpy as np


def check_1d_array_object(parameter, name_param):
    if isinstance(parameter, (np.ndarray, list)):
        parameter = np.array(parameter)
        if parameter.ndim != 1:
            raise TypeError("{} should be a 1d array type object".format(name_param))
    else:
        raise TypeError("{} should be a 1d array type object".format(name_param))

    return parameter


class BaseGradLogPDF(object):
    """
    Base class for evaluating gradient log of probability density function/ distribution

    Classes inheriting this base class can be passed as an argument for
    finding gradient log of probability distribution in inference algorithms

    The class should initialize  self.grad_log and self.log_pdf

    Parameters
    ----------
    distribution_param : A 1d array like object (numpy.ndarray or list)
        Vector representing values of distribution parameter at which we want to find gradient and log

    model : An instance of pgmpy.models

    Examples
    --------
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference.continuous import BaseGradLogPDF
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
    >>> mean = np.array([1, 1])
    >>> covariance = np.array([[1, 0.2], [0.2, 7]])
    >>> model = JointGaussianDistribution(['x', 'y'], mean, covariance)
    >>> dist_param = np.array([0.1, 0.9])
    >>> grad_logp, logp = GradLogGaussian(dist_param, model).get_gradient_log_pdf()
    >>> logp
    -0.4054597701149426
    >>> grad_logp
    array([ 0.90229885, -0.01149425])
    """

    def __init__(self, distribution_param, model):

        self.distribution_param = check_1d_array_object(distribution_param, 'distribution_param')
        self.model = model

        # The gradient log of probability distribution at position
        self.grad_log = None

        # The gradient log of probability distribution at position
        self.log_pdf = None

    def get_gradient_log_pdf(self):
        """
        Returns the gradient log and log of model at given position

        Returns
        -------
        Returns a tuple of following types

        numpy.array: Representing gradient log of model at given position

        float: Representing log of model at given position

        Example
        --------
        >>> # Using implementation of GradLogPDFGaussian
        >>> from pgmpy.inference.continuous import GradLogPDFGaussian
        >>> from pgmpy.models import JointGaussianDistribution
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, -5], [-5, 2]])
        >>> model = JointGaussianDistribution(['x', 'y'], mean, covariance)
        >>> dist_param = np.array([0.6, 0.8])
        >>> grad_logp, logp = GradLogPDFGaussian(dist_param, model).get_gradient_log_pdf()
        >>> logp
        0.025217391304347823
        >>> grad_logp
        array([-0.07826087, -0.09565217])
        """
        return self.grad_log, self.log_pdf


class GradLogPDFGaussian(BaseGradLogPDF):
    """
    Class for finding gradient and gradient log of Joint Gaussian Distribution
    Inherits pgmpy.inference.base_continuous.BaseGradLogPDF
    Here model must be an instance of JointGaussianDistribution

    Parameters
    ----------
    distribution_param : A 1d array like object (numpy.ndarray or list)
        Vector representing values of distribution parameter at which we want to find gradient and log

    model : An instance of pgmpy.models.JointGaussianDistribution

    Example
    -------
    >>> from pgmpy.inference.continuous import GradLogPDFGaussian
    >>> from pgmpy.models import JointGaussianDistribution
    >>> import numpy as np
    >>> mean = np.array([3, 4])
    >>> covariance = np.array([[5, 4], [4, 5]])
    >>> model = JointGaussianDistribution(['x', 'y'], mean, covariance)
    >>> dist_param = np.array([12, 21])
    >>> grad_logp, logp = GradLogPDFGaussian(dist_param, model).get_gradient_log_pdf()
    >>> logp
    -34.777777777777771
    >>> grad_logp
    array([ 2.55555556, -5.44444444])
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
        log_pdf = 0.5 * np.dot(sub_vec, grad)

        return grad, log_pdf


class BaseSimulateHamiltonianDynamics(object):
    """
    Base class for proposing new values of position and momentum by simulating Hamiltonian Dynamics.

    Classes inheriting this base class can be passed as an argument for
    simulate_dynamics in inference algorithms.

    Parameters
    ----------
    grad_log_pdf : A subclass of pgmpy.inference.continuous.BaseGradLogPDF

    model : An instance of pgmpy.models
        Model for which DiscretizeTime object is initialized

    position : A 1d array like object (numpy.ndarray or list)
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object (numpy.ndarray or list)
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    grad_log_position: A 1d array like object, defaults to None
        Vector representing gradient log at given position
        If None, then will be calculated

    Examples
    --------
    >>> from pgmpy.inference.continuous import BaseSimulateHamiltonianDynamics
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference.continuous import GradLogPDFGaussian
    >>> # Class should initalize self.new_position, self.new_momentum and self.new_grad_logp
    >>> # self.new_grad_logp represents gradient log at new proposed value of position
    >>> class ModifiedEuler(BaseSimulateHamiltonianDynamics):
    ...     def __init__(self, grad_log_pdf, model, position, momentum, stepsize, grad_log_position=None):
    ...         BaseSimulateHamiltonianDynamics.__init__(self, grad_log_pdf, model, position,
    ...                                                  momentum, stepsize, grad_log_position)
    ...         self.new_position, self.new_momentum, self.new_grad_logp = self._get_proposed_values()
    ...     def _get_proposed_values(self):
    ...         momentum_bar = self.momentum + self.stepsize * self.grad_log_position
    ...         position_bar = self.position + self.stepsize * momentum_bar
    ...         return position_bar, momentum_bar, grad_log_position
    >>> import numpy as np
    >>> pos = np.array([1, 2])
    >>> momentum = np.array([0, 0])
    >>> mean = np.array([0, 0])
    >>> covariance = np.eye(2)
    >>> model = JointGaussianDistribution(['x', 'y'], mean, covariance)
    >>> new_pos, new_momentum, new_grad = ModifiedEuler(GradLogPDFGaussian, model,
    ...                                                 pos, momentum, 0.25).get_proposed_values()
    >>> new_pos
    array([0.9375, 1.875])
    >>> new_momentum
    array([-0.25, -0.5])
    >>> new_grad
    array([-0.9375, -1.875])
    """

    def __init__(self, grad_log_pdf, model, position, momentum, stepsize, grad_log_position=None):

        position = check_1d_array_object(position, 'position')

        momentum = check_1d_array_object(momentum, 'momentum')

        if not issubclass(grad_log_pdf, BaseGradLogPDF):
            raise TypeError("grad_log_pdf must be an instance" +
                            " of pgmpy.inference.continuous.base.BaseGradLogPDF")

        if position.shape != momentum.shape:
            raise ValueError("Size of position and momentum must be same")

        if grad_log_position is None:
            grad_log_position, _ = grad_log_pdf(position, model).get_gradient_log_pdf()

        self.position = position
        self.momentum = momentum
        self.stepsize = stepsize
        self.model = model
        self.grad_log_pdf = grad_log_pdf
        self.grad_log_position = grad_log_position

        # new_position is the new proposed position, new_momentum is the new proposed momentum, new_grad_lop
        # is the value of grad log at new_position
        self.new_position = self.new_momentum = self.new_grad_logp = None

    def get_proposed_values(self):
        """
        Returns the new proposed values of position and momentum

        Returns
        -------
        Returns a tuple of following type

        numpy.array: New proposed value of position

        numpy.array: New proposed value of momentum

        numpy.array: Gradient of log distribution at new proposed value of position

        Example
        -------
        >>> # Using implementation of ModifiedEuler
        >>> from pgmpy.inference.continuous import ModifiedEuler
        >>> from pgmpy.models import JointGaussianDistribution
        >>> from pgmpy.inference.continuous import GradLogPDFGaussian
        >>> import numpy as np
        >>> pos = np.array([3, 4])
        >>> momentum = np.array([1, 1])
        >>> mean = np.array([-1, 1])
        >>> covariance = 3*np.eye(2)
        >>> model = JointGaussianDistribution(['x', 'y'], mean, covariance)
        >>> new_pos, new_momentum, new_grad = ModifiedEuler(GradLogPDFGaussian, model, pos,
        ...                                                 momentum, 0.70).get_proposed_values()
        >>> new_pos
        array([ 3.04666667,  4.21      ])
        >>> new_momentum
        array([ 0.06666667,  0.3       ])
        >>> new_grad
        array([-1.34888889, -1.07      ])
        """
        return self.new_position, self.new_momentum, self.new_grad_logp


class LeapFrog(BaseSimulateHamiltonianDynamics):
    """
    Class for simulating hamiltonian dynamics using leapfrog method
    Inherits pgmpy.inference.base_continuous.BaseSimulateHamiltonianDynamics

    Parameters
    ----------
    grad_log_pdf : A subclass of pgmpy.inference.continuous.BaseGradLogPDF

    model : An instance of pgmpy.models
        Model for which DiscretizeTime object is initialized

    position : A 1d array like object (numpy.ndarray or list)
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object (numpy.ndarray or list)
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    grad_log_position: A 1d array like object, defaults to None
        Vector representing gradient log at given position
        If None, then will be calculated

    Example
    --------
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference.continuous import GradLogPDFGaussian, LeapFrog
    >>> import numpy as np
    >>> pos = np.array([2, 1])
    >>> momentum = np.array([7, 7])
    >>> mean = np.array([-5, 5])
    >>> covariance = np.array([[1, 2], [2, 1]])
    >>> model = JointGaussianDistribution(['x', 'y'], mean, covariance)
    >>> new_pos, new_momentum, new_grad = LeapFrog(GradLogPDFGaussian, model, pos,
    ...                                            momentum, 4.0).get_proposed_values()
    >>> new_pos
    array([ 70., -19.])
    >>> new_momentum
    array([  99., -121.])
    >>> new_grad
    array([ 41., -58.])
    """

    def __init__(self, grad_log_pdf, model, position, momentum, stepsize, grad_log_position=None):

        BaseSimulateHamiltonianDynamics.__init__(self, grad_log_pdf, model, position,
                                                 momentum, stepsize, grad_log_position)

        self.new_position, self.new_momentum, self.new_grad_logp = self._get_proposed_values()

    def _get_proposed_values(self):
        """
        Method to perform time splitting using leapfrog
        """
        # Take half step in time for updating momentum
        momentum_bar = self.momentum + 0.5 * self.stepsize * self.grad_log_position

        # Take full step in time for updating position position
        position_bar = self.position + self.stepsize * momentum_bar

        grad_log, _ = self.grad_log_pdf(position_bar,
                                        self.model).get_gradient_log_pdf()

        # Take remaining half step in time for updating momentum
        momentum_bar = momentum_bar + 0.5 * self.stepsize * grad_log

        return position_bar, momentum_bar, grad_log


class ModifiedEuler(BaseSimulateHamiltonianDynamics):
    """
    Class for simulating Hamiltonian Dynamics using Modified euler method
    Inherits pgmpy.inference.base_continuous.BaseSimulateHamiltonianDynamics

    Parameters
    ----------
    grad_log_pdf : A subclass of pgmpy.inference.continuous.BaseGradLogPDF

    model : An instance of pgmpy.models
        Model for which DiscretizeTime object is initialized

    position : A 1d array like object (numpy.ndarray or list)
        Vector representing values of parameter position( or X)

    momentum: A 1d array like object (numpy.ndarray or list)
        Vector representing the proposed value for momentum (velocity)

    stepsize: Float
        stepsize for the simulating dynamics

    grad_log_position: A 1d array like object, defaults to None
        Vector representing gradient log at given position
        If None, then will be calculated

    Example
    --------
    >>> from pgmpy.models import JointGaussianDistribution
    >>> from pgmpy.inference.continuous import GradLogPDFGaussian, ModifiedEuler
    >>> import numpy as np
    >>> pos = np.array([2, 1])
    >>> momentum = np.array([1, 1])
    >>> mean = np.array([0, 0])
    >>> covariance = np.eye(2)
    >>> model = JointGaussianDistribution(['x', 'y'], mean, covariance)
    >>> new_pos, new_momentum, new_grad = ModifiedEuler(GradLogPDFGaussian, model, pos,
    ...                                                 momentum, 0.25).get_proposed_values()
    >>> new_pos
    array([2.125, 1.1875])
    >>> new_momentum
    array([0.5, 0.75])
    >>> new_grad
    array([-2.125, -1.1875])
    """

    def __init__(self, grad_log_pdf, model, position, momentum, stepsize, grad_log_position=None):

        BaseSimulateHamiltonianDynamics.__init__(self, grad_log_pdf, model, position,
                                                 momentum, stepsize, grad_log_position)

        self.new_position, self.new_momentum, self.new_grad_logp = self._get_proposed_values()

    def _get_proposed_values(self):
        """
        Method to perform time splitting using Modified euler method
        """
        # Take full step in time and update momentum
        momentum_bar = self.momentum + self.stepsize * self.grad_log_position

        # Take full step in time and update position
        position_bar = self.position + self.stepsize * momentum_bar

        grad_log, _ = self.grad_log_pdf(position_bar, self.model).get_gradient_log_pdf()

        return position_bar, momentum_bar, grad_log
