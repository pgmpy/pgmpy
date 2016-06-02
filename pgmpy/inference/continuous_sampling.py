"""
    A collection of methods for sampling from continuous models in pgmpy
"""
from math import sqrt

import numpy as np

from pgmpy.inference import (LeapFrog, BaseHMC)


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

    discretize_time: A subclass of pgmpy.inference.base_continuous.DiscretizeTime

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

    def sample(self, theta0, num_adapt, num_samples, epsilon=None):
        """
        Method to return samples using Hamiltonian Monte Carlo

        Parameters
        ----------
        theta0: A 1d array type object or a row matrix of shape 1 X d
                or d X 1.(Will be converted to d X 1)
                Vector representing values of parameter theta, the starting
                state in markov chain.

        num_adapt: int
                The number of interations to run the adaptation of epsilon,
                after Madapt iterations adaptation will be stopped

        num_samples: int
                     Number of samples to be generated

        epsilon: float , defaults to None
                 The step size for descrete time method
                 If None, then will be choosen suitably

        Returns
        -------
        list: A list of numpy matrix type objects containing samples

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

        if epsilon is None:
            epsilon = self._find_resonable_epsilon(theta0)

        mu = np.log(10.0 * epsilon)  # freely chosen point, after each iteration xt(/theta) is shrunk towards it
        # log(10 * epsilon) large values to save computation
        epsilon_bar = 1.0
        hbar = 0.0
        gamma = 0.05  # free parameter that controls the amount of shrinkage towards mu
        t0 = 10.0  # free parameter that stabilizes the initial iterations of the algorithm
        kappa = 0.75
        # See equation (6) section 3.2.1 for details
        samples = [theta0.copy()]
        theta_m = theta0.copy()
        for i in range(1, num_samples):
            # Genrating sample
            # Resampling momentum
            momentum0 = np.matrix(np.reshape(np.random.normal(0, 1, len(theta0)), (len(theta0), 1)))
            # theta_m here will be the previous sampled value of theta
            theta_bar, momentum_bar = theta_m.copy(), momentum0.copy()
            # Number of steps L to run discretize time algorithm
            lsteps = int(max(1, round(self.Lambda / epsilon, 0)))

            for j in range(lsteps):
                # Taking L steps in time
                theta_bar, momentum_bar = self.discretize_time(self.grad_log_pdf, self.model, theta_bar.copy(),
                                                               momentum_bar.copy(), epsilon).discretize_time()

            _, log_bar = self.grad_log_pdf(theta_bar.copy(), self.model).get_gradient_log_pdf()
            # log_m_1 = log(theta_m) or log(theta_m_1)
            _, log_m_1 = self.grad_log_pdf(theta_m.copy(), self.model).get_gradient_log_pdf()

            # Metropolis acceptance probability
            alpha = min(1, np.exp(log_bar - log_m_1 - 0.5 *
                                  np.float(momentum_bar.T * momentum_bar - momentum0.T * momentum0)))

            # Accept or reject the new proposed value of theta, i.e theta_bar
            if np.random.rand() < alpha:
                theta_m = theta_bar.copy()

            samples.append(theta_m.copy())

            # Adaptation of epsilon till num_adapt iterations
            if i <= num_adapt:
                # Burn-in updates
                estimate = 1.0 / (i + t0)
                hbar = (1 - estimate) * hbar + estimate * (self.delta - alpha)

                epsilon = np.exp(mu - sqrt(i) / gamma * hbar)
                i_kappa = i ** (-kappa)
                epsilon_bar = np.exp(i_kappa * np.log(epsilon) + (1 - i_kappa) * np.log(epsilon_bar))

            else:
                # Burn-in finished used the last value
                epsilon = epsilon_bar

        return samples

    def generate_sample(self, theta0, num_adapt, num_samples, epsilon=None):
        """
        Method returns a generator type object whose each iteration yields a sample
        using Hamiltonian Monte Carlo

        Parameters
        ----------
        theta0: A 1d array type object or a row matrix of shape 1 X d
                or d X 1.(Will be converted to d X 1)
                Vector representing values of parameter theta, the starting
                state in markov chain.

        num_adapt: int
                The number of interations to run the adaptation of epsilon,
                after Madapt iterations adaptation will be stopped

        num_samples: int
                     Number of samples to be generated

        epsilon: float , defaults to None
                 The step size for descrete time method
                 If None, then will be choosen suitably

        Returns
        -------
        genrator: yielding a numpy.matrix type object for a sample

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

        if epsilon is None:
            epsilon = self._find_resonable_epsilon(theta0)

        mu = np.log(10.0 * epsilon)  # freely chosen point, after each iteration xt(/theta) is shrunk towards it
        # log(10 * epsilon) large values to save computation
        epsilon_bar = 1.0
        hbar = 0.0
        gamma = 0.05  # free parameter that controls the amount of shrinkage towards mu
        t0 = 10.0  # free parameter that stabilizes the initial iterations of the algorithm
        kappa = 0.75
        # See equation (6) section 3.2.1 for details
        theta_m = theta0.copy()
        for i in range(0, num_samples):
            # Genrating sample
            # Resampling momentum
            momentum0 = np.matrix(np.reshape(np.random.normal(0, 1, len(theta0)), (len(theta0), 1)))
            # theta_m here will be the previous sampled value of theta
            theta_bar, momentum_bar = theta_m.copy(), momentum0.copy()
            # Number of steps L to run discretize time algorithm
            lsteps = int(max(1, round(self.Lambda / epsilon, 0)))

            for j in range(lsteps):
                # Taking L steps in time
                theta_bar, momentum_bar = self.discretize_time(self.grad_log_pdf, self.model, theta_bar.copy(),
                                                               momentum_bar.copy(), epsilon).discretize_time()

            _, log_bar = self.grad_log_pdf(theta_bar.copy(), self.model).get_gradient_log_pdf()
            # log_m_1 = log(theta_m) or log(theta_m_1)
            _, log_m_1 = self.grad_log_pdf(theta_m.copy(), self.model).get_gradient_log_pdf()

            # Metropolis acceptance probability
            alpha = min(1, np.exp(log_bar - log_m_1 - 0.5 *
                                  np.float(momentum_bar.T * momentum_bar - momentum0.T * momentum0)))

            # Accept or reject the new proposed value of theta, i.e theta_bar
            if np.random.rand() < alpha:
                theta_m = theta_bar.copy()

            # Adaptation of epsilon till num_adapt iterations
            if i <= num_adapt:
                # Burn-in updates
                estimate = 1.0 / (i + t0)
                hbar = (1 - estimate) * hbar + estimate * (self.delta - alpha)

                epsilon = np.exp(mu - sqrt(i) / gamma * hbar)
                i_kappa = i ** (-kappa)
                epsilon_bar = np.exp(i_kappa * np.log(epsilon) + (1 - i_kappa) * np.log(epsilon_bar))

            else:
                # Burn-in finished used the last value
                epsilon = epsilon_bar

            yield theta_m.copy()
