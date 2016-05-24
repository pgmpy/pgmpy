"""
    A collection of methods for sampling from continuous models in pgmpy
"""
import numpy as np
from pgmpy.inference.base_continuous import LeapFrog, DiscretizeTime, GradientLogPDF, BaseHMC


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
        HMC.__init__(model, grad_log_pdf, discretize_time, delta)

        self.Lambda = Lamda

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
        epsilon = 1
        momentum = np.matrix(np.random.randn(1, len(theta)))
        theta_bar, momentum_bar = self.discretize_time(self.grad_log_pdf, self.model,
                                                  theta, momentum, epsilon).discretize_time()

        grad = - self.theta * self.model.precision_matrix
        log_grad = 0.5 * grad * self.theta.transpose()
        grad_bar = - self.theta_bar * self.model.precision_matrix
        log_grad_bar = 0.5 * grad_bar * self.theta.transpose()

        accept_prob = np.exp(log_grad_bar - log_grad - 0.5 * (grad_bar *
                             grad_bar.transpose - momentum * momentum.transpose()))
        accept_prob = float(accept_prob)
        a = 2 * (accept_prob) > 0.5) - 1

        while(accept_prob ** a > 2 ** (- a)):
            epsilon = (2 ** a) * epsilon

            theta_bar, momentum_bar = self.discretize_time(self.grad_log_pdf, self.model,
                                                    theta, momentum, epsilon).discretize_time()
            grad = - self.theta * self.model.precision_matrix
            log_grad = 0.5 * grad * self.theta.transpose()
            grad_bar = - self.theta_bar * self.model.precision_matrix
            log_grad_bar = 0.5 * grad_bar * self.theta.transpose()
            accept_prob = np.exp(log_grad_bar - log_grad - 0.5 * (grad_bar *
                                grad_bar.transpose - momentum * momentum.transpose()))
            accept_prob=float(accept_prob)
        return epsilon
