"""
    A collection of methods for sampling from continuous models in pgmpy
"""
from math import sqrt

import numpy as np

from pgmpy.inference.continuous import (LeapFrog, BaseGradLogPDF, BaseSimulateDynamics)


class HamiltonianMC(object):
    """
    Class for performing sampling using simple
    Hamiltonian Monte Carlo

    Parameters:
    -----------
    model: An instance pgmpy.models.Continuous
        Model from which sampling has to be done

    grad_log_pdf: A subclass of pgmpy.inference.base_continuous.BaseGradLogPDF
        A class to find log and gradient of log distribution

    simulate_dynamics: A subclass of pgmpy.inference.base_continuous.BaseSimulateDynamics
        A class to propose future values of momentum and position in time by simulating
        Hamiltonian Dynamics

    Public Methods:
    ---------------
    sample()
    generate_sample()

    Example:
    --------
    >>> from pgmpy.inference import HamiltonianMC as HMC
    >>> from pgmpy.inference import JointGaussianDistribution as JGD, GradLogPDFGaussian as GLPG
    >>> from pgmpy.inference import LeapFrog
    >>> import numpy as np
    >>> mean = np.array([1, 1])
    >>> covariance = np.array([[1, 0.7], [0.7, 1]])
    >>> model = JGD(mean, covariance)
    >>> sampler = HMC(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
    >>> samples = sampler.sample(np.array([[1], [1]]), num_samples = 10000, trajectory_length=2, stepsize=None)
    >>> samples_array = np.concatenate(samples, axis=1)
    >>> np.cov(samples_array)
    array([[ 0.64321553,  0.63513749],
           [ 0.63513749,  0.98544953]])

    References
    ----------
    R.Neal. Handbook of Markov Chain Monte Carlo,
    chapter 5: MCMC Using Hamiltonian Dynamics.
    CRC Press, 2011.
    """

    def __init__(self, model, grad_log_pdf, simulate_dynamics=LeapFrog):

        if not issubclass(grad_log_pdf, BaseGradLogPDF):
            raise TypeError("grad_log_pdf must be an instance of" +
                            "pgmpy.inference.base_continuous.BaseGradLogPDF")

        if not issubclass(simulate_dynamics, BaseSimulateDynamics):
            raise TypeError("split_time must be an instance of" +
                            "pgmpy.inference.base_continuous.BaseSimulateDynamics")

        self.model = model
        self.grad_log_pdf = grad_log_pdf
        self.simulate_dynamics = simulate_dynamics

    def _acceptance_prob(self, position, position_bar, momentum, momentum_bar):
        """
        Returns the acceptance probability for given new position(position) and momentum
        """

        # Parameters to help in evaluating P(position, momentum)
        _, logp = self.grad_log_pdf(position, self.model).get_gradient_log_pdf()
        _, logp_bar = self.grad_log_pdf(position_bar, self.model).get_gradient_log_pdf()

        # acceptance_prob = P(position_bar, momentum_bar)/ P(position, momentum)
        potential_change = logp_bar - logp  # Negative change
        kinetic_change = 0.5 * np.float(np.dot(momentum_bar.T, momentum_bar) - np.dot(momentum.T, momentum))

        return np.exp(potential_change - kinetic_change)  # acceptance probability

    def _find_reasonable_stepsize(self, position, stepsize_app=1):
        """
        Method for choosing initial value of stepsize

        References
        -----------
        Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
        Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
        Machine Learning Research 15 (2014) 1351-1381
        Algorithm 4 : Heuristic for choosing an initial value of epsilon
        """
        # momentum = N(0, I)
        momentum = np.reshape(np.random.normal(0, 1, len(position)), position.shape)

        # Take a single step in time
        position_bar, momentum_bar = self.simulate_dynamics(self.grad_log_pdf, self.model,
                                                            position, momentum, stepsize_app).get_proposed_values()

        acceptance_prob = self._acceptance_prob(position, position_bar, momentum, momentum_bar)

        # a = 2I[acceptance_prob] -1
        a = 2 * (acceptance_prob > 0.5) - 1

        condition = (acceptance_prob ** a) > (2 ** (-a))

        while condition:
            stepsize_app = (2 ** a) * stepsize_app

            position_bar, momentum_bar = self.simulate_dynamics(self.grad_log_pdf, self.model,
                                                                position, momentum, stepsize_app).get_proposed_values()

            acceptance_prob = self._acceptance_prob(position, position_bar, momentum, momentum_bar)

            condition = (acceptance_prob ** a) > (2 ** (-a))

        return stepsize_app

    def sample(self, position0, num_samples, trajectory_length, stepsize=None):
        """
        Method to return samples using Hamiltonian Monte Carlo

        Parameters
        ----------
        position0: A 1d array like object
            Vector representing values of parameter position, the starting
            state in markov chain.

        num_samples: int
            Number of samples to be generated

        trajectory_length: int or float
            Target trajectory length, stepsize * number of steps(L),
            where L is the number of steps taken per HMC iteration,
            and stepsize is step size for splitting time method.

        stepsize: float , defaults to None
            The stepsize for proposing new values of position and momentum in simulate_dynamics
            If None, then will be choosen suitably


        Returns
        -------
        list: A list of numpy array type objects containing samples

        Examples
        --------
        >>> from pgmpy.inference import HamiltonianMC as HMC
        >>> from pgmpy.inference import JointGaussianDistribution as JGD, GradLogPDFGaussian as GLPG
        >>> from pgmpy.inference import LeapFrog
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 1]])
        >>> model = JGD(mean, covariance)
        >>> sampler = HMC(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
        >>> samples = sampler.sample(np.array([[1], [1]]), num_samples = 10000, trajectory_length=2, stepsize=None)
        >>> samples_array = np.concatenate(samples, axis=1)
        >>> np.cov(samples_array)
        array([[ 0.64321553,  0.63513749],
               [ 0.63513749,  0.98544953]])
        """

        if isinstance(position0, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            position0 = np.array(position0).flatten()
            position0 = np.matrix(np.reshape(position0, (len(position0), 1)))
        else:
            raise TypeError("position should be a 1d array type object")

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(position0)

        samples = [position0.copy()]
        position_m = position0.copy()

        for i in range(1, num_samples):
            # Genrating sample
            # Resampling momentum
            momentum0 = np.reshape(np.random.normal(0, 1, len(position0)), position0.shape)
            # position_m here will be the previous sampled value of position
            position_bar, momentum_bar = position_m.copy(), momentum0.copy()
            # Number of steps L to run discretize time algorithm
            lsteps = int(max(1, round(trajectory_length / stepsize, 0)))

            for _ in range(lsteps):
                # Taking L steps in time
                position_bar, momentum_bar = self.simulate_dynamics(self.grad_log_pdf, self.model, position_bar,
                                                                    momentum_bar, stepsize).get_proposed_values()

            acceptance_prob = self._acceptance_prob(position_m.copy(), position_bar.copy(), momentum0, momentum_bar)
            # Metropolis acceptance probability
            alpha = min(1, acceptance_prob)
            # Accept or reject the new proposed value of position, i.e position_bar
            if np.random.rand() < alpha:
                position_m = position_bar.copy()

            samples.append(position_m.copy())

        return samples

    def generate_sample(self, position0, num_samples, trajectory_length, stepsize=None):
        """
        Method returns a generator type object whose each iteration yields a sample
        using Hamiltonian Monte Carlo

        Parameters
        ----------
        position0: A 1d array like object
            Vector representing values of parameter position, the starting
            state in markov chain.

        num_samples: int
            Number of samples to be generated

        trajectory_length: int or float
            Target trajectory length, stepsize * number of steps(L),
            where L is the number of steps taken per HMC iteration,
            and stepsize is step size for splitting time method.

        stepsize: float , defaults to None
            The stepsize for proposing new values of position and momentum in simulate_dynamics
            If None, then will be choosen suitably

        Returns
        -------
        genrator: yielding a numpy.array type object for a sample

        Examples
        --------
        >>> from pgmpy.inference import HamiltonianMC as HMC
        >>> from pgmpy.inference import JointGaussianDistribution as JGD, GradLogPDFGaussian as GLPG
        >>> from pgmpy.inference import ModifiedEuler
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 1]])
        >>> model = JGD(mean, covariance)
        >>> sampler = HMC(model=model, grad_log_pdf=GLPG, simulate_dynamics=ModifiedEuler)
        >>> gen_samples = sampler.generate_sample(np.array([[1], [1]]), num_samples = 10000,
                                                  trajectory_length=2, stepsize=None)
        >>> samples = [sample for sample in gen_samples]
        >>> samples_array = np.concatenate(samples, axis=1)
        >>> np.cov(samples_array)
        array([[ 1.84321553,  0.33513749],
               [ 0.33513749,  1.98544953]])
        >>> # LeapFrog performs best with HMC algorithm
        """

        if isinstance(position0, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            position0 = np.array(position0).flatten()
            position0 = np.reshape(position0, (len(position0), 1))
        else:
            raise TypeError("position should be a 1d array type object")

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(position0)

        position_m = position0.copy()
        for i in range(0, num_samples):
            # Genrating sample
            # Resampling momentum
            momentum0 = np.reshape(np.random.normal(0, 1, len(position0)), position0.shape)
            # position_m here will be the previous sampled value of position
            position_bar, momentum_bar = position_m.copy(), momentum0.copy()
            # Number of steps L to run discretize time algorithm
            lsteps = int(max(1, round(trajectory_length / stepsize, 0)))

            for _ in range(lsteps):
                # Taking L steps in time
                position_bar, momentum_bar = self.simulate_dynamics(self.grad_log_pdf, self.model, position_bar,
                                                                    momentum_bar, stepsize).get_proposed_values()

            acceptance_prob = self._acceptance_prob(position_m.copy(), position_bar.copy(), momentum0, momentum_bar)
            # Metropolis acceptance probability
            alpha = min(1, acceptance_prob)
            # Accept or reject the new proposed value of position, i.e position_bar
            if np.random.rand() < alpha:
                position_m = position_bar.copy()

            yield position_m.copy()


class HamiltonianMCda(HamiltonianMC):
    """
    Class for performing sampling in Continuous model
    using Hamiltonian Monte Carlo with dual averaging for
    adaptaion of parameter stepsize.

    Parameters:
    -----------
    model: An instance pgmpy.models.Continuous
        Model from which sampling has to be done

    grad_log_pdf: A subclass of pgmpy.inference.base_continuous.GradientLogPDF
        Class to compute the log and gradient log of distribution

    simulate_dynamics: A subclass of pgmpy.inference.base_continuous.BaseSimulateDynamics
        Class to propose future states of position and momentum in time by simulating
        HamiltonianDynamics

    delta: float (in between 0 and 1), defaults to 0.65
        The target HMC acceptance probability

    Public Methods:
    ---------------
    sample()
    generate_sample()

    Example:
    --------
    >>> from pgmpy.inference import HamiltonianMCda as HMCda
    >>> from pgmpy.inference import JointGaussianDistribution as JGD, GradLogPDFGaussian as GLPG
    >>> from pgmpy.inference import LeapFrog
    >>> import numpy as np
    >>> mean = np.array([1, 1])
    >>> covariance = np.array([[1, 0.7], [0.7, 3]])
    >>> model = JGD(mean, covariance)
    >>> sampler = HMCda(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
    >>> samples = sampler.sample(np.array([[1], [1]]), num_adapt=10000,
    ...                          num_samples = 10000, trajectory_length=2, stepsize=None)
    >>> samples_array = np.concatenate(samples, axis=1)
    >>> np.cov(samples_array)
    array([[ 0.98432155,  0.66517394],
           [ 0.66517394,  2.95449533]])

    References
    -----------
    Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
    Machine Learning Research 15 (2014) 1351-1381
    Algorithm 5 : Hamiltonian Monte Carlo with dual averaging
    """

    def __init__(self, model, grad_log_pdf, simulate_dynamics=LeapFrog, delta=0.65):

        if not isinstance(delta, float) or delta > 1.0 or delta < 0.0:
            raise AttributeError(
                "delta should be a floating value in between 0 and 1")

        self.delta = delta

        super(HamiltonianMCda, self).__init__(model=model, grad_log_pdf=grad_log_pdf,
                                              simulate_dynamics=simulate_dynamics)

    def _adapt_params(self, stepsize, stepsize_bar, h_bar, mu, index_i, alpha):
        """
        Run tha adaptation for stepsize for better proposals of position
        """
        gamma = 0.05  # free parameter that controls the amount of shrinkage towards mu
        t0 = 10.0  # free parameter that stabilizes the initial iterations of the algorithm
        kappa = 0.75
        # See equation (6) section 3.2.1 for details

        estimate = 1.0 / (index_i + t0)
        h_bar = (1 - estimate) * h_bar + estimate * (self.delta - alpha)

        stepsize = np.exp(mu - sqrt(index_i) / gamma * h_bar)
        i_kappa = index_i ** (-kappa)
        stepsize_bar = np.exp(i_kappa * np.log(stepsize) + (1 - i_kappa) * np.log(stepsize_bar))

        return stepsize, stepsize_bar, h_bar

    def sample(self, position0, num_adapt, num_samples, trajectory_length, stepsize=None):
        """
        Method to return samples using Hamiltonian Monte Carlo

        Parameters
        ----------
        position0: A 1d array like object
            Vector representing values of parameter position, the starting
            state in markov chain.

        num_adapt: int
            The number of interations to run the adaptation of stepsize

        num_samples: int
            Number of samples to be generated

        trajectory_length: int or float
            Target trajectory length, stepsize * number of steps(L),
            where L is the number of steps taken per HMC iteration,
            and stepsize is step size for splitting time method.

        stepsize: float , defaults to None
            The stepsize for proposing new values of position and momentum in simulate_dynamics
            If None, then will be choosen suitably

        Returns
        -------
        list: A list of numpy array type objects containing samples

        Examples
        ---------
        >>> from pgmpy.inference import HamiltonianMCda as HMCda
        >>> from pgmpy.inference import JointGaussianDistribution as JGD, GradLogPDFGaussian as GLPG
        >>> from pgmpy.inference import LeapFrog
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 3]])
        >>> model = JGD(mean, covariance)
        >>> sampler = HMCda(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
        >>> samples = sampler.sample(np.array([[1], [1]]), num_adapt=10000,
        ...                          num_samples = 10000, trajectory_length=2, stepsize=None)
        >>> samples_array = np.concatenate(samples, axis=1)
        >>> np.cov(samples_array)
        array([[ 0.98432155,  0.66517394],
               [ 0.66517394,  2.95449533]])

        """

        if isinstance(position0, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            position0 = np.array(position0).flatten()
            position0 = np.reshape(position0, (len(position0), 1))
        else:
            raise TypeError("position should be a 1d array type object")

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(position0)

        if num_adapt <= 1:  # Return samples genrated using Simple HMC algorithm
            return HamiltonianMC.sample(self, position0, num_samples, stepsize)

        # stepsize is epsilon
        mu = np.log(10.0 * stepsize)  # freely chosen point, after each iteration xt(/position) is shrunk towards it
        # log(10 * stepsize) large values to save computation
        # stepsize_bar is epsilon_bar
        stepsize_bar = 1.0
        h_bar = 0.0
        # See equation (6) section 3.2.1 for details
        samples = [position0.copy()]
        position_m = position0.copy()
        for i in range(1, num_samples):
            # Genrating sample
            # Resampling momentum
            momentum0 = np.reshape(np.random.normal(0, 1, len(position0)), position0.shape)
            # position_m here will be the previous sampled value of position
            position_bar, momentum_bar = position_m.copy(), momentum0.copy()
            # Number of steps L to run discretize time algorithm
            lsteps = int(max(1, round(trajectory_length / stepsize, 0)))

            for _ in range(lsteps):
                # Taking L steps in time
                position_bar, momentum_bar = self.simulate_dynamics(self.grad_log_pdf, self.model, position_bar,
                                                                    momentum_bar, stepsize).get_proposed_values()

            acceptance_prob = self._acceptance_prob(position_m.copy(), position_bar.copy(), momentum0, momentum_bar)
            # Metropolis acceptance probability
            alpha = min(1, acceptance_prob)

            # Accept or reject the new proposed value of position, i.e position_bar
            if np.random.rand() < alpha:
                position_m = position_bar.copy()

            samples.append(position_m.copy())

            # Adaptation of stepsize till num_adapt iterations
            if i <= num_adapt:
                stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar, mu, i, alpha)
            else:
                stepsize = stepsize_bar

        return samples

    def generate_sample(self, position0, num_adapt, num_samples, trajectory_length, stepsize=None):
        """
        Method returns a generator type object whose each iteration yields a sample
        using Hamiltonian Monte Carlo

        Parameters
        ----------
        position0: A 1d array like object
            Vector representing values of parameter position, the starting
            state in markov chain.

        num_adapt: int
            The number of interations to run the adaptation of stepsize

        num_samples: int
            Number of samples to be generated

        trajectory_length: int or float
            Target trajectory length, stepsize * number of steps(L),
            where L is the number of steps taken to propose new values of position and momentum
            per HMC iteration and stepsize is step size.

        stepsize: float , defaults to None
            The stepsize for proposing new values of position and momentum in simulate_dynamics
            If None, then will be choosen suitably

        Returns
        -------
        genrator: yielding a numpy.array type object for a sample

        Examples
        --------
        >>> from pgmpy.inference import HamiltonianMCda as HMCda
        >>> from pgmpy.inference import JointGaussianDistribution as JGD, GradLogPDFGaussian as GLPG
        >>> from pgmpy.inference import LeapFrog
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 3]])
        >>> model = JGD(mean, covariance)
        >>> sampler = HMCda(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
        >>> gen_samples = sampler.generate_sample(np.array([[1], [1]]), num_adapt=10000,
        ...                                       num_samples = 10000, trajectory_length=2, stepsize=None)
        >>> samples = [sample for sample in gen_samples]
        >>> samples_array = np.concatenate(samples, axis=1)
        >>> np.cov(samples_array)
        array([[ 0.98432155,  0.69517394],
               [ 0.69517394,  2.95449533]])
        """

        if isinstance(position0, (np.matrix, np.ndarray, list, tuple, set, frozenset)):
            position0 = np.array(position0).flatten()
            position0 = np.reshape(position0, (len(position0), 1))
        else:
            raise TypeError("position should be a 1d array type object")

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(position0)

        if num_adapt <= 1:  # return sample generated using Simple HMC algorithm
            for sample in HamiltonianMC.generate_sample(self, position0, num_samples, stepsize, trajectory_length):
                yield sample
            return
        # stepsize is epsilon
        mu = np.log(10.0 * stepsize)  # freely chosen point, after each iteration xt(/position) is shrunk towards it
        # log(10 * stepsize) large values to save computation
        # stepsize_bar is epsilon_bar
        stepsize_bar = 1.0
        h_bar = 0.0
        # See equation (6) section 3.2.1 for details
        position_m = position0.copy()
        num_adapt += 1

        for i in range(1, num_samples + 1):
            # Genrating sample
            # Resampling momentum
            momentum0 = np.reshape(np.random.normal(0, 1, len(position0)), position0.shape)
            # position_m here will be the previous sampled value of position
            position_bar, momentum_bar = position_m.copy(), momentum0.copy()
            # Number of steps L to run discretize time algorithm
            lsteps = int(max(1, round(trajectory_length / stepsize, 0)))

            for _ in range(lsteps):
                # Taking L steps in time
                position_bar, momentum_bar = self.simulate_dynamics(self.grad_log_pdf, self.model, position_bar,
                                                                    momentum_bar, stepsize).get_proposed_values()

            acceptance_prob = self._acceptance_prob(position_m.copy(), position_bar.copy(), momentum0, momentum_bar)
            # Metropolis acceptance probability
            alpha = min(1, acceptance_prob)
            # Accept or reject the new proposed value of position, i.e position_bar
            if np.random.rand() < alpha:
                position_m = position_bar.copy()

            # Adaptation of stepsize till num_adapt iterations
            if i <= num_adapt:
                stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar, mu, i, alpha)
            else:
                stepsize = stepsize_bar

            yield position_m.copy()
