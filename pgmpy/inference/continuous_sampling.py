"""
    A collection of methods for sampling from continuous models in pgmpy
"""
import numpy as np
from base_continuous import (LeapFrog,
                             DiscretizeTime, GradientLogPDF, BaseHMC, JointGaussianDistribution)


class HamiltonianMCda(BaseHMC):
    """
    Class for performing sampling in Continuous model
    using Hamiltonian Monte Carlo with dual averaging

    Parameters:
    -----------
    model: An instance pgmpy.models.Continuous

    Lambda: float
            Target trajectory length, epsilon * number of steps(L),
            where L is the number of steps taken per HMC iteration,
            and epsilon is step size for splitting time method.

    grad_log_pdf: A subclass of pgmpy.inference.base_continuous.GradientLogPDF

    discretize_time: A instance of pgmpy.inference.base_continuous.DiscretizeTime

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

    def __init__(self, model, Lambda, grad_log_pdf,
                 discretize_time=LeapFrog, delta=0.65):
        # TODO: Use model instead of mean_vec and cov_matrix
        BaseHMC.__init__(self, model=model, grad_log_pdf=grad_log_pdf,
                         discretize_time=discretize_time, delta=delta)

        self.Lambda = Lambda

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
        momentum = np.matrix(np.reshape(np.random.normal(0, 1, len(theta)), (len(theta), 1)))

        # Take a single step in time
        theta_bar, momentum_bar = self.discretize_time(self.grad_log_pdf, self.model,
                                                       theta, momentum, epsilon_app).discretize_time()
        # Parameters to help in evaluating P(theta, momentum)
        _, logp = self.grad_log_pdf(theta, self.model).get_gradient_log_pdf()
        _, logp_bar = self.grad_log_pdf(theta_bar, self.model).get_gradient_log_pdf()

        # acceptance_prob = P(theta_bar, momentum_bar)/ P(theta, momentum)
        potential_change = logp_bar - logp  # Negative change
        kinetic_change = 0.5 * np.float(np.dot(momentum_bar.T, momentum_bar) - np.dot(momentum.T, momentum))

        acceptance_prob = np.exp(potential_change - kinetic_change)

        # a = 2I[acceptance_prob] -1
        a = 2 * (acceptance_prob > 0.5) - 1

        condition = (acceptance_prob ** a) > (2 ** (-a))

        while condition:
            epsilon_app = (2 ** a) * epsilon_app

            theta_bar, momentum_bar = self.discretize_time(self.grad_log_pdf, self.model,
                                                           theta, momentum, epsilon_app).discretize_time()

            _, logp = self.grad_log_pdf(theta, self.model).get_gradient_log_pdf()
            _, logp_bar = self.grad_log_pdf(theta_bar, self.model).get_gradient_log_pdf()

            potential_change = logp_bar - logp
            kinetic_change = 0.5 * np.float(momentum_bar.T * momentum_bar - momentum.T * momentum)

            acceptance_prob = np.exp(potential_change - kinetic_change)

            condition = (acceptance_prob ** a) > (2 ** (-a))

        return epsilon_app

    def sample(self, theta0, Madapt, num_samples, epsilon=None):
        """
        Method to return samples using Hamiltonian Monte Carlo

        Parameters
        ----------
        theta0: A 1d array type object or a row matrix of shape 1 X d
                or d X 1.(Will be converted to d X 1)
                Vector representing values of parameter theta, the starting
                state in markov chain.

        Madapt: int
                The number of interations to run the adaptation of epsilon,
                after Madapt iterations adaptation will be stopped

        num_samples: int
                     Number of samples to be generated

        epsilon: float , defaults to None
                 The step size for descrete time method
                 If None, then will be choosen suitably

        Returns
        -------

        Examples
        --------
        >>>

        References
        ----------
        Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
        Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
        Machine Learning Research 15 (2014) 1351-1381
        Algorithm 5 : Hamiltonian Monte Carlo with dual averaging
        """
        if isinstance(theta0, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            theta0 = np.array(theta0).flatten()
            theta0 = np.matrix(np.reshape(theta0, (len(theta0), 1)))
        else:
            raise TypeError("theta should be a 1d array type object")

        epsilon_m_1 = epsilon

        if epsilon_m_1 is None:
            epsilon_m_1 = self._find_resonable_epsilon(theta0)

        mu = np.log(10 * epsilon_m_1)  # freely chosen point that iterates xt are shrunk towards
        # log(10 * epsilon) large values to save computation
        theta_m_1 = theta0
        epsilon_bar_m_1 = 1
        hbar_m_1 = 0
        gamma = 0.005  # free parameter that controls the amount of shrinkage towards mu
        t0 = 10  # free parameter that stabilizes the initial iterations of the algorithm
        kappa = 0.75
        # See equation (6) section 3.2.1 for details
        samples = []

        for i in range(0, num_samples):
            # Genrating sample
            # Resampling momentum
            momentum0 = np.matrix(np.reshape(np.random.normal(0, 1, len(theta0)), (len(theta0), 1)))

            theta_m, theta_bar, momentum_bar = theta_m_1.copy(), theta_m_1.copy(), momentum0.copy()
            # Number of steps L to run discretize time algorithm
            L = max(1, round(self.Lambda / epsilon_m_1, 0))
            for i in range(L):
                # Taking L steps in time
                theta_bar, momentum_bar = self.discretize_time(self.grad_log_pdf, self.model,
                                                               self.theta_bar.copy(), self.momentum_bar.copy(), epsilon_m_1)

            _, log_bar = self.grad_log_pdf(theta_bar, self.model)
            _, log_m_1 = self.grad_log_pdf(theta_m_1)
            # Metropolis acceptance probability
            alpha = min(1, np.exp(log_bar - 0.5 * np.float(momentum_bar.T * momentum_bar)) /
                        np.exp(log_m_1 - 0.5 * np.float(momentum0.T * momentum0)))

            if np.random.rand() < alpha:
                theta_m = theta_bar.copy()
            samples.append(theta_m)

            # Adaptation of epsilon till Madapt iterations
            if i <= Madapt:

                hbar_m = (1 - 1 / (i + t0)) * hbar_m_1 + (self.delta - alpha) / (i + t0)

                epsilon_m = np.exp(mu - (hbar_m * i ** 0.5) / gamma)
                i_kappa = i ** (- kappa)
                epsilon_bar_m = np.exp(i_kappa * np.log(epsilon_m) + (1 - i_kappa) * np.log(epsilon_bar_m_1))
                # Update the values for next iteration
                epsilon_m_1 = epsilon_m
                epsilon_bar_m_1 = epsilon_bar_m
                hbar_m_1 = hbar_m

            else:
                epsilon_m_1 = epsilon_bar_m_1
            # Updating values for next iteration
            theta_m_1 = theta_m.copy()

        return samples
