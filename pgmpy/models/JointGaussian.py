import numpy as np

from pgmpy.utils import _check_1d_array_object, _check_length_equal
from pgmpy.inference.continuous import BaseGradLogPDF


class JointGaussianDistribution(object):
    """
    A naive gaussian container class.
    Will be removed when Multivariate Distributions will be implemented

    Paramters
    ---------
    variables: iterable of any hashable python object
        The variables for which the distribution is defined.

    mean: 1d array type of structure
        Represents the mean of the distribution

    covariance: A 2d array type object of size len(mean) x len(mean),
        Covariance of the distribution.

    """

    def __init__(self, variables, mean, covariance):

        mean = _check_1d_array_object(mean, 'mean')
        _check_length_equal(mean, variables, 'mean', 'variables')

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
        self.variables = variables
        self.mean = mean
        self.covariance = covariance
        self.precision_matrix = np.linalg.inv(covariance)

    def get_gradient_log_pdf(self, distribution_param, grad_log_pdf=None):
        """
        A method for finding gradient log and log of distribution for given assignment

        Parameters
        ----------
        distribution_param: A 1d array like object
            Assignment of variables at which gradient is to be computed

        grad_log_pdf: A subclass of pgmpy.inference.continuous.BaseGradLogPDF, defaults to None
            A coustom class for finding gradient log and log for given assignment
            If None, the will be computed

        Example
        ---------

        Returns
        --------
        A tuple of following types (in order)

        numpy.array: A 1d numpy.array representing value of gradient log of JointGaussianDistribution

        float: A floating value representin log of JointGaussianDistribution
        """
        if grad_log_pdf is not None:
            if not issubclass(grad_log_pdf, BaseGradLogPDF):
                raise TypeError("grad_log_pdf must be an instance" +
                                " of pgmpy.inference.continuous.base.BaseGradLogPDF")
            return grad_log_pdf(distribution_param, self).get_gradient_log_pdf()

        sub_vec = distribution_param - self.mean
        grad = - np.dot(self.precision_matrix, sub_vec)
        log_pdf = 0.5 * np.dot(sub_vec, grad)

        return grad, log_pdf
