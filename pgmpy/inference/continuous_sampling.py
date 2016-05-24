"""
    A collection of methods for sampling from continuous models in pgmpy
"""
import numpy as np
from pgmpy.inference.base_continuous import LeapFrog, SplitTime, GradientLogPDF


class HamiltonianMCda(object):
    """
    Class for performing sampling in Continuous model
    using Hamiltonian Monte Carlo with dual averaging

    Parameters:
    -----------
    mean_vec: A vector (row matrix or 1d array like object)
              Represents the mean of the distribution

    cov_matrix: A matrix of size len(mean_vec) x len(mean_vec) or 2d list,
                Covariance matrix for the distribution.

    Lambda: float
            Target trajectory length, epsilon * number of steps(L),
            where L is the number of steps taken per HMC iteration,
            and epsilon is step size for splitting time method.

    delta: float (in between 0 and 1), defaults to 0.65
           The target HMC acceptance probability

    grad_log_pdf: A instance of pgmpy.inference.base_continuous.GradientLogPDF

    split_time: A instance of pgmpy.inference.base_continuous.SplitTime

    Public Methods:
    ---------------
    sample()
    generate_sample()

    Example:
    --------
    #

    References
    -----------
    Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
    Machine Learning Research 15 (2014) 1351-1381
    """

    def __init__(self, mean_vec=None, cov_matrix=None, Lamda, delta=0.65,
                 grad_log_pdf, split_time=LeapFrog):
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

        self.mu = mean_vec
        self.sgima = cov_matrix
        self.Lambda = Lamda
        self.precision_matrix = np.linalg.inv(cov_matrix)
        self.delta = delta
        self.grad_log_pdf = grad_log_pdf

    def _find_resonable_epsilon(self, theta):
        """
        Method for choosing initial value of epsilon

        References
        -----------
        Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
        Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
        Machine Learning Research 15 (2014) 1351-1381
        Algorithm 4 : Heuristic for choosing an initial value of epsilon
        """
