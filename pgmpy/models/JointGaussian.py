import numpy as np

from pgmpy.inference.continuous import check_1d_array_object

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
        
        mean = check_1d_array_object(mean, 'mean')

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
