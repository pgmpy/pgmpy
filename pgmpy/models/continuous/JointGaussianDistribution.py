import six
import numpy as np


class JointGaussianDistribution(object):
    """
    In its most common representation, a multivariate Gaussian distribution
    over X1...........Xn is characterized by an n-dimensional mean vector μ,
    and a symmetric nXn covariance matrix Σ.
    This is the base class for its representation.
    """
    def __init__(self, variables, mean_vector, covariance_matrix):
        """
        Parameters
        ----------
        variables: iterable of any hashable python object
            The variables for which the distribution is defined.
        mean_vector: nX1, array like 
            n-dimensional vector where n is the number of variables.
        covariance_matrix: nXn, matrix or 2-d array like
            nXn dimensional matrix where n is the number of variables.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> dis = JGD(['x1', 'x2', 'x3'], np.array([[1], [-3], [4]])),
                        np.array([[4, 2, -2], [2, 5, -5], [-2, -5, 8]]))
        >>> dis.variables
        ['x1', 'x2', 'x3']
        >>> dis.mean_vector
        np.matrix([[ 1],
                   [-3],
                   [4]]))
        >>> dis.covariance_matrix
        np.matrix([[4, 2, -2],
                   [2, 5, -5],
                   [-2, -5, 8]])
        """
        self.variables = variables
        # dimension of the mean vector and covariance matrix
        n = len(self.variables)

        if len(mean_vector) != n:
            raise ValueError("Length of mean_vector must be equal to the\
                                 number of variables.")

        self.mean_vector = np.matrix(np.reshape(mean_vector, (n, 1)))

        self.covariance_matrix = np.matrix(covariance_matrix)

        if self.covariance_matrix.shape != (n, n):
            raise ValueError("Each dimension of the covariance matrix must\
                                 be equal to the number of variables.")

        self._precision_matrix = None

    @property
    def precision_matrix(self):
        if self._precision_matrix is None:
            self._precision_matrix = np.linalg.inv(self.covariance_matrix)
        return self._precision_matrix
