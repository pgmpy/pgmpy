"""
    A collection of methods for sampling from continuous models in pgmpy
"""
from math import sqrt

import numpy as np
import pandas as pd

from pgmpy.inference.continuous import LeapFrog, BaseGradLogPDF, BaseSimulateHamiltonianDynamics
from pgmpy.utils import _check_1d_array_object, _check_length_equal


class HamiltonianMC(object):
    """
    Class for performing sampling using simple
    Hamiltonian Monte Carlo

    Parameters:
    -----------
    model: An instance pgmpy.models
        Model from which sampling has to be done

    grad_log_pdf: A subclass of pgmpy.inference.continuous.BaseGradLogPDF
        A class to find log and gradient of log distribution

    simulate_dynamics: A subclass of pgmpy.inference.continuous.BaseSimulateHamiltonianDynamics
        A class to propose future values of momentum and position in time by simulating
        Hamiltonian Dynamics

    Public Methods:
    ---------------
    sample()
    generate_sample()

    Example:
    --------
    >>> from pgmpy.inference.continuous import HamiltonianMC as HMC, GradLogPDFGaussian as GLPG, LeapFrog
    >>> from pgmpy.models import JointGaussianDistribution as JGD
    >>> import numpy as np
    >>> mean = np.array([1, 1])
    >>> covariance = np.array([[1, 0.7], [0.7, 1]])
    >>> model = JGD(['x', 'y'], mean, covariance)
    >>> sampler = HMC(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
    >>> samples = sampler.sample(initial_pos=np.array([1, 1]), num_samples = 10000,
    ...                          trajectory_length=2, stepsize=None)
    >>> samples_array = np.concatenate(samples, axis=1)
    >>> np.cov(samples_array)
    array([[ 1.00028107,  0.64565895],
           [ 0.64565895,  0.84694746]])

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

        if not issubclass(simulate_dynamics, BaseSimulateHamiltonianDynamics):
            raise TypeError("split_time must be an instance of" +
                            "pgmpy.inference.base_continuous.BaseSimulateHamiltonianDynamics")

        self.model = model
        self.grad_log_pdf = grad_log_pdf
        self.simulate_dynamics = simulate_dynamics
        self.acceptance_rate = 0.0

    def _acceptance_prob(self, position, position_bar, momentum, momentum_bar):
        """
        Returns the acceptance probability for given new position(position) and momentum
        """

        # Parameters to help in evaluating Joint distribution P(position, momentum)
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
        position_bar, momentum_bar, grad_bar =\
            self.simulate_dynamics(self.grad_log_pdf, self.model, position,
                                   momentum, stepsize_app).get_proposed_values()

        acceptance_prob = self._acceptance_prob(position, position_bar, momentum, momentum_bar)

        # a = 2I[acceptance_prob] -1
        a = 2 * (acceptance_prob > 0.5) - 1

        condition = (acceptance_prob ** a) > (2 ** (-a))

        while condition:
            stepsize_app = (2 ** a) * stepsize_app

            position_bar, momentum_bar, grad_bar =\
                self.simulate_dynamics(self.grad_log_pdf, self.model, position,
                                       momentum, stepsize_app, grad_bar).get_proposed_values()

            acceptance_prob = self._acceptance_prob(position, position_bar, momentum, momentum_bar)

            condition = (acceptance_prob ** a) > (2 ** (-a))

        return stepsize_app

    def _sample(self, position, trajectory_length, stepsize, lsteps=None):
        """
        Runs a single sampling iteration to return a sample
        """
        # Resampling momentum
        momentum = np.reshape(np.random.normal(0, 1, len(position)), position.shape)

        # position_m here will be the previous sampled value of position
        position_bar, momentum_bar = position.copy(), momentum

        # Number of steps L to simulate dynamics
        if lsteps is None:
            lsteps = int(max(1, round(trajectory_length / stepsize, 0)))

        grad_bar, _ = self.grad_log_pdf(position_bar, self.model).get_gradient_log_pdf()

        for _ in range(lsteps):
            position_bar, momentum_bar, grad_bar =\
                self.simulate_dynamics(self.grad_log_pdf, self.model, position_bar,
                                       momentum_bar, stepsize, grad_bar).get_proposed_values()

        acceptance_prob = self._acceptance_prob(position, position_bar, momentum, momentum_bar)

        # Metropolis acceptance probability
        alpha = min(1, acceptance_prob)

        # Accept or reject the new proposed value of position, i.e position_bar
        if np.random.rand() < alpha:
            position = position_bar.copy()
            self.acceptance_rate += 1.0

        return position, alpha

    def sample(self, initial_pos, num_samples, trajectory_length, stepsize=None):
        """
        Method to return samples using Hamiltonian Monte Carlo

        Parameters
        ----------
        initial_pos: A 1d array like object
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
        >>> from pgmpy.inference.continuous import HamiltonianMC as HMC, GradLogPDFGaussian as GLPG, LeapFrog
        >>> from pgmpy.models import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 1]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMC(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
        >>> samples = sampler.sample(np.array([1, 1]), num_samples = 10000, trajectory_length=2, stepsize=None)
        >>> samples_array = np.concatenate(samples, axis=1)
        >>> np.cov(samples_array)
        array([[ 0.64321553,  0.63513749],
               [ 0.63513749,  0.98544953]])
        """

        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        shape = (len(initial_pos), 1)
        samples = [np.reshape(initial_pos, shape)]
        position_m = initial_pos.copy()

        lsteps = int(max(1, round(trajectory_length / stepsize, 0)))
        for i in range(1, num_samples):

            # Genrating sample
            position_m, _ = self._sample(position_m, trajectory_length, stepsize, lsteps)
            samples.append(np.reshape(position_m, shape))

        self.acceptance_rate /= num_samples

        return samples

    def generate_sample(self, initial_pos, num_samples, trajectory_length, stepsize=None):
        """
        Method returns a generator type object whose each iteration yields a sample
        using Hamiltonian Monte Carlo

        Parameters
        ----------
        initial_pos: A 1d array like object
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
        >>> from pgmpy.inference.continuous import HamiltonianMC as HMC, GradLogPDFGaussian as GLPG, ModifiedEuler
        >>> from pgmpy.models import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 1]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMC(model=model, grad_log_pdf=GLPG, simulate_dynamics=ModifiedEuler)
        >>> gen_samples = sampler.generate_sample(np.array([1, 1]), num_samples = 10000,
                                                  trajectory_length=2, stepsize=None)
        >>> samples = [sample for sample in gen_samples]
        >>> samples_array = np.concatenate(samples, axis=1)
        >>> np.cov(samples_array)
        array([[ 1.84321553,  0.33513749],
               [ 0.33513749,  1.98544953]])
        >>> # LeapFrog performs best with HMC algorithm
        """
        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        shape = (len(initial_pos), 1)
        lsteps = int(max(1, round(trajectory_length / stepsize, 0)))
        position_m = initial_pos.copy()

        for i in range(0, num_samples):

            position_m, _ = self._sample(position_m, trajectory_length, stepsize, lsteps)

            yield np.reshape(position_m, shape)

        self.acceptance_rate /= num_samples


class HamiltonianMCda(HamiltonianMC):
    """
    Class for performing sampling in Continuous model
    using Hamiltonian Monte Carlo with dual averaging for
    adaptaion of parameter stepsize.

    Parameters:
    -----------
    model: An instance pgmpy.models
        Model from which sampling has to be done

    grad_log_pdf: A subclass of pgmpy.inference.continuous.GradientLogPDF
        Class to compute the log and gradient log of distribution

    simulate_dynamics: A subclass of pgmpy.inference.continuous.BaseSimulateHamiltonianDynamics
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
    >>> from pgmpy.inference.continuous import HamiltonianMCda as HMCda, GradLogPDFGaussian as GLPG, LeapFrog
    >>> from pgmpy.models import JointGaussianDistribution as JGD
    >>> import numpy as np
    >>> mean = np.array([1, 1])
    >>> covariance = np.array([[1, 0.7], [0.7, 3]])
    >>> model = JGD(['x', 'y'], mean, covariance)
    >>> sampler = HMCda(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
    >>> samples = sampler.sample(np.array([1, 1]), num_adapt=10000,
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

    def sample(self, initial_pos, num_adapt, num_samples, trajectory_length, stepsize=None):
        """
        Method to return samples using Hamiltonian Monte Carlo

        Parameters
        ----------
        initial_pos: A 1d array like object
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
        >>> from pgmpy.inference.continuous import HamiltonianMCda as HMCda, GradLogPDFGaussian as GLPG, LeapFrog
        >>> from pgmpy.models import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 3]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMCda(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
        >>> samples = sampler.sample(np.array([1, 1]), num_adapt=10000,
        ...                          num_samples = 10000, trajectory_length=2, stepsize=None)
        >>> samples_array = np.concatenate(samples, axis=1)
        >>> np.cov(samples_array)
        array([[ 0.98432155,  0.66517394],
               [ 0.66517394,  2.95449533]])

        """

        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        if num_adapt <= 1:  # Return samples genrated using Simple HMC algorithm
            return HamiltonianMC.sample(self, initial_pos, num_samples, stepsize)

        # stepsize is epsilon
        mu = np.log(10.0 * stepsize)  # freely chosen point, after each iteration xt(/position) is shrunk towards it
        # log(10 * stepsize) large values to save computation
        # stepsize_bar is epsilon_bar
        stepsize_bar = 1.0
        h_bar = 0.0
        # See equation (6) section 3.2.1 for details
        shape = (len(initial_pos), 1)
        samples = [np.reshape(initial_pos, shape)]
        position_m = initial_pos.copy()
        for i in range(1, num_samples):

            # Genrating sample
            position_m, alpha = self._sample(position_m, trajectory_length, stepsize)
            samples.append(np.reshape(position_m, shape))

            # Adaptation of stepsize till num_adapt iterations
            if i <= num_adapt:
                stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar, mu, i, alpha)
            else:
                stepsize = stepsize_bar

        self.acceptance_rate /= num_samples
        return samples

    def generate_sample(self, initial_pos, num_adapt, num_samples, trajectory_length, stepsize=None):
        """
        Method returns a generator type object whose each iteration yields a sample
        using Hamiltonian Monte Carlo

        Parameters
        ----------
        initial_pos: A 1d array like object
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
        >>> from pgmpy.inference.continuous import HamiltonianMCda as HMCda, GradLogPDFGaussian as GLPG, LeapFrog
        >>> from pgmpy.models import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 3]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMCda(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
        >>> gen_samples = sampler.generate_sample(np.array([1, 1]), num_adapt=10000,
        ...                                       num_samples = 10000, trajectory_length=2, stepsize=None)
        >>> samples = [sample for sample in gen_samples]
        >>> samples_array = np.concatenate(samples, axis=1)
        >>> np.cov(samples_array)
        array([[ 0.98432155,  0.69517394],
               [ 0.69517394,  2.95449533]])
        """

        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        if num_adapt <= 1:  # return sample generated using Simple HMC algorithm
            for sample in HamiltonianMC.generate_sample(self, initial_pos, num_samples, stepsize, trajectory_length):
                yield sample
            return
        mu = np.log(10.0 * stepsize)

        stepsize_bar = 1.0
        h_bar = 0.0

        position_m = initial_pos.copy()
        num_adapt += 1
        shape = (len(initial_pos), 1)
        for i in range(1, num_samples + 1):

            position_m, alpha = self._sample(position_m, trajectory_length, stepsize)

            if i <= num_adapt:
                stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar, mu, i, alpha)
            else:
                stepsize = stepsize_bar

            yield np.reshape(position_m, shape)

        self.acceptance_rate /= num_samples
