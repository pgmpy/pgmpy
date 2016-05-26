"""
    A collection of methods for sampling from continuous models in pgmpy
"""
import numpy as np
from pgmpy.inference.base_continuous import (LeapFrog,
                                             DiscretizeTime, GradientLogPDF, BaseHMC, AbstractGaussian)


class HamiltonianMCda(BaseHMC):
    """
    Class for performing sampling in Continuous model
    using Hamiltonian Monte Carlo with dual averaging

    Parameters:
    -----------
    model: An instance AbstractGaussian

    Lambda: float
            Target trajectory length, epsilon * number of steps(L),
            where L is the number of steps taken per HMC iteration,
            and epsilon is step size for splitting time method.

    grad_log_pdf: A instance of pgmpy.inference.base_continuous.GradientLogPDF

    discretize_time: A instance of pgmpy.inference.base_continuous.SplitTime

    delta: float (in between 0 and 1), defaults to 0.65
           The target HMC acceptance probability

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

    def __init__(self, model, Lamda, grad_log_pdf,
                 discretize_time=LeapFrog, delta=0.65):
        # TODO: Use model instead of mean_vec and cov_matrix
        BaseHMC.__init__(model, grad_log_pdf, discretize_time, delta)

        self.Lambda = Lamda

    def _find_resonable_epsilon(self, theta, epsilon_app=1):
        """
        Method for choosing initial value of epsilon

        References
        -----------
        Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
        Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
        Machine Learning Research 15 (2014) 1351-1381
        Algorithm 4 : Heuristic for choosing an initial value of epsilon
        """
        # momentum = N(0, I)
        momentum = np.matrix(np.random.randn(len(theta), 1))
        # Take a single step in time
        theta_bar, _ = self.discretize_time(self.grad_log_pdf, self.model,
                                            theta, momentum, epsilon_app).discretize_time()
        # Parameters to help in evaluating P(theta, momentum)
        _, logp = self.grad_log_pdf(theta, self.model).get_gradient_log_pdf()
        grad_bar, logp_bar = self.grad_log_pdf(
            theta_bar, self.model).get_gradient_log_pdf()

        # acceptance_prob = P(theta_bar, momentum_bar)/ P(theta, momentum)
        acceptance_prob = np.exp(logp_bar - logp - 0.5 * (
            grad_bar.transpose() * grad_bar - momentum.transpose() * momentum)).sum()

        # a = 2I[acceptance_prob] -1
        a = 2 * (acceptance_prob > 0.5) - 1

        condition = acceptance_prob ** a > 2 ** (-a)

        while condition:
            epsilon_app = (2 ** a) * epsilon_app

            theta_bar, _ = self.discretize_time(self.grad_log_pdf, self.model,
                                                theta, momentum, epsilon_app).discretize_time()

            _, logp = self.grad_log_pdf(theta, self.model).get_gradient_log_pdf()
            grad_bar, logp_bar = self.grad_log_pdf(theta_bar, self.model).get_gradient_log_pdf()

            acceptance_prob = np.exp(logp_bar - logp - 0.5 * (
                                     grad_bar.transpose() * grad_bar - momentum.transpose() * momentum)).sum()

            condition = acceptance_prob ** a > 2 ** (-a)

        return epsilon_app
