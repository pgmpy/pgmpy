"""
    A collection of methods for sampling from continuous models in pgmpy
"""
import numpy as np


class HamiltonianMC_da(object):
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

    log_grad_pdf: Function or a callable object
                  It should take inputs a vector theta and precision matrix
                  (inverse of cov_matrix) and should return
                  the log of distribution and gradient log of distribution.

    split_time: A string, defaults to 'leapfrog'
                Name of algorithm to use for simulating time.
                Valid inputs are '[E]euler' or '[L]leapfrog'.

    Public Methods:
    ---------------
    leapfrog()
    euler_method()
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
                 log_grad_pdf, split_time='leapfrog'):
        split_time = split_time.lower()

        if split_time == 'leapfrog':
            self.split_time = self.leapfrog

        elif (split_time == 'euler' or split_time == 'euler method' or
                 split_time == 'euler_method'):
            self.split_time = self.euler_method

        else:
            raise AttributeError(
                "split_time must be either leapfrog or euler method")

        self.mu = mean_vec
        self.sgima = cov_matrix
        self.Lambda = Lamda
        self.precision_matrix = np.linalg.inv(cov_matrix)

        if not isinstance(delta, float) or delta > 1.0 or delta < 0.0:
            raise AttributeError(
                "delta should be a floating value in between 0 and 1")

        self.delta = delta

        if not hasattr(log_grad_pdf, '__call__'):
            raise TypeError("log_grad_pdf should be callable type object")

        self.log_grad_pdf = log_grad_pdf

    def leapfrog(self, theta, momentum, epsilon):
        """
        Leap frog method for splitting time

        Parameters
        ------------
        theta: A vector (row matrix or 1d array like object)
               Vector representing the proposed value for the
               distribution parameter theta

        momentum: A vector (row matrix or 1d array like object)
                  Vector representing the proposed value for momentum

        epsilon: Float
                 step size for the Leapfrog.

        Returns
        --------
        numpy.matrix : The new proposed value for random variable theta.
        numpy.matrix : The new proposed value for momentum variable.
        numpy.matrix : The gradient of probability density function
        float : The log of probability density function.
        """
        if not isinstance(theta, np.matrix):
            if isinstance(theta, (np.array, list, tuple, set, frozenset)):
                theta = np.matrix(theta)
            else:
                raise TypeError("theta should be a 1d array type object")

        if not isinstance(momentum, np.matrix):
            if isinstance(momentum, (np.array, list, tuple, set, frozenset)):
                momentum = np.matrix(momentum)
            else:
                raise TypeError("momentum should be a 1d array type object")

        grad_theta, log_grad_theta = self.log_grad_pdf(
            theta, self.precision_matrix)
        momentum_bar = momentum + 0.5 * epsilon * log_grad_theta
        theta_bar = theta + epsilon * momentum_bar
        grad_theta, log_grad_theta = self.log_grad_pdf(
            theta_bar, self.precision_matrix)
        momentum_bar = momentum_bar + 0.5 * epsilon * log_grad_theta

        return theta_bar, momentum_bar, grad_theta, log_grad_theta

    def euler_method(self, theta, momentum, epsilon):
        """
        Modified Euler method for splitting time

        Parameters
        ------------
        theta: A vector (row matrix or 1d array like object)
               Vector representing the proposed value for the
               distribution parameter theta

        momentum: A vector (row matrix or 1d array like object)
                  Vector representing the proposed value for momentum

        epsilon: Float
                 step size for the Leapfrog.

        Returns
        --------
        numpy.matrix : The new proposed value for random variable theta.
        numpy.matrix : The new proposed value for momentum variable.
        numpy.matrix : The gradient of probability density function
        float : The log of probability density function.
        """
        if not isinstance(theta, np.matrix):
            if isinstance(theta, (np.array, list, tuple, set, frozenset)):
                theta = np.matrix(theta)
            else:
                raise TypeError("theta should be a 1d array type object")
        if not isinstance(momentum, np.matrix):
            if isinstance(momentum, (np.array, list, tuple, set, frozenset)):
                momentum = np.matrix(momentum)
            else:
                raise TypeError("momentum should be a 1d array type object")

        grad_theta, log_grad_theta = self.log_grad_pdf(
            theta, self.precision_matrix)
        momentum_bar = momentum + epsilon * log_grad_theta
        theta_bar = theta + epsilon * momentum_bar

        return theta_bar, momentum_bar, grad_theta, log_grad_theta

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
