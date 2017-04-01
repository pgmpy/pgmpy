# -*- coding: UTF-8 -*-
"""
    A collection of methods for sampling from continuous models in pgmpy
"""
from __future__ import division
from math import sqrt

import numpy as np

from pgmpy.utils import _check_1d_array_object, _check_length_equal
from pgmpy.sampling import LeapFrog, BaseSimulateHamiltonianDynamics, BaseGradLogPDF, _return_samples


class HamiltonianMC(object):
    """
    Class for performing sampling using simple
    Hamiltonian Monte Carlo

    Parameters:
    -----------
    model: An instance pgmpy.models
        Model from which sampling has to be done

    grad_log_pdf: A subclass of pgmpy.inference.continuous.BaseGradLogPDF, defaults to None
        A class to find log and gradient of log distribution for a given assignment
        If None, then will use model.get_gradient_log_pdf

    simulate_dynamics: A subclass of pgmpy.inference.continuous.BaseSimulateHamiltonianDynamics
        A class to propose future values of momentum and position in time by simulating
        Hamiltonian Dynamics

    Public Methods:
    ---------------
    sample()
    generate_sample()

    Example:
    --------
    >>> from pgmpy.sampling import HamiltonianMC as HMC, LeapFrog, GradLogPDFGaussian
    >>> from pgmpy.factors.continuous import GaussianDistribution as JGD
    >>> import numpy as np
    >>> mean = np.array([-3, 4])
    >>> covariance = np.array([[3, 0.7], [0.7, 5]])
    >>> model = JGD(['x', 'y'], mean, covariance)
    >>> sampler = HMC(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
    >>> samples = sampler.sample(initial_pos=np.array([1, 1]), num_samples = 10000,
    ...                          trajectory_length=2, stepsize=0.4, return_type='recarray')
    >>> samples
    rec.array([(1.0, 1.0), (-3.1861687131079086, 3.7940994520145654),
     (-1.6920542547310844, 6.347410703806017), ...,
     (-1.8093621120575312, 5.940860883943261),
     (0.3933248026088032, 6.3853098838119235),
     (-0.8654072934719572, 6.023803629334816)],
              dtype=[('x', '<f8'), ('y', '<f8')])

    >>> samples = np.array([samples[var_name] for var_name in model.variables])
    >>> np.cov(samples)
    array([[ 3.0352818 ,  0.71379304],
           [ 0.71379304,  4.91776713]])
    >>> sampler.accepted_proposals
    9932.0
    >>> sampler.acceptance_rate
    0.9932

    References
    ----------
    R.Neal. Handbook of Markov Chain Monte Carlo,
    chapter 5: MCMC Using Hamiltonian Dynamics.
    CRC Press, 2011.
    """

    def __init__(self, model, grad_log_pdf, simulate_dynamics=LeapFrog):

        if not issubclass(grad_log_pdf, BaseGradLogPDF):
            raise TypeError("grad_log_pdf must be an instance of " +
                            "pgmpy.inference.base_continuous.BaseGradLogPDF")

        if not issubclass(simulate_dynamics, BaseSimulateHamiltonianDynamics):
            raise TypeError("split_time must be an instance of " +
                            "pgmpy.inference.base_continuous.BaseSimulateHamiltonianDynamics")

        self.model = model
        self.grad_log_pdf = grad_log_pdf
        self.simulate_dynamics = simulate_dynamics
        self.accepted_proposals = 0.0
        self.acceptance_rate = 0

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

        # acceptance probability
        return np.exp(potential_change - kinetic_change)

    def _get_condition(self, acceptance_prob, a):
        """
        Temporary method to fix issue in numpy 0.12 #852
        """
        if a == 1:
            return (acceptance_prob ** a) > (1/(2**a))
        else:
            return (1/(acceptance_prob ** a)) > (2**(-a))

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
        position_bar, momentum_bar, _ =\
            self.simulate_dynamics(self.model, position, momentum,
                                   stepsize_app, self.grad_log_pdf).get_proposed_values()

        acceptance_prob = self._acceptance_prob(position, position_bar, momentum, momentum_bar)

        # a = 2I[acceptance_prob] -1
        a = 2 * (acceptance_prob > 0.5) - 1

        condition = self._get_condition(acceptance_prob, a)

        while condition:
            stepsize_app = (2 ** a) * stepsize_app

            position_bar, momentum_bar, _ =\
                self.simulate_dynamics(self.model, position, momentum,
                                       stepsize_app, self.grad_log_pdf).get_proposed_values()

            acceptance_prob = self._acceptance_prob(position, position_bar, momentum, momentum_bar)

            condition = self._get_condition(acceptance_prob, a)

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
                self.simulate_dynamics(self.model, position_bar, momentum_bar,
                                       stepsize, self.grad_log_pdf, grad_bar).get_proposed_values()

        acceptance_prob = self._acceptance_prob(position, position_bar, momentum, momentum_bar)

        # Metropolis acceptance probability
        alpha = min(1, acceptance_prob)

        # Accept or reject the new proposed value of position, i.e position_bar
        if np.random.rand() < alpha:
            position = position_bar.copy()
            self.accepted_proposals += 1.0

        return position, alpha

    def sample(self, initial_pos, num_samples, trajectory_length, stepsize=None, return_type='dataframe'):
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

        return_type: string (dataframe | recarray)
            Return type for samples, either of 'dataframe' or 'recarray'.
            Defaults to 'dataframe'

        Returns
        -------
        sampled: A pandas.DataFrame or a numpy.recarray object depending upon return_type argument

        Examples
        --------
        >>> from pgmpy.sampling import HamiltonianMC as HMC, GradLogPDFGaussian, ModifiedEuler
        >>> from pgmpy.factors.continuous import GaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, -1])
        >>> covariance = np.array([[1, 0.2], [0.2, 1]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMC(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=ModifiedEuler)
        >>> samples = sampler.sample(np.array([1, 1]), num_samples = 5,
        ...                          trajectory_length=6, stepsize=0.25, return_type='dataframe')
        >>> samples
                       x              y
        0   1.000000e+00   1.000000e+00
        1   1.592133e+00   1.152911e+00
        2   1.608700e+00   1.315349e+00
        3   1.608700e+00   1.315349e+00
        4   6.843856e-01   6.237043e-01
        >>> mean = np.array([4, 1, -1])
        >>> covariance = np.array([[1, 0.7 , 0.8], [0.7, 1, 0.2], [0.8, 0.2, 1]])
        >>> model = JGD(['x', 'y', 'z'], mean, covariance)
        >>> sampler = HMC(model=model, grad_log_pdf=GLPG)
        >>> samples = sampler.sample(np.array([1, 1]), num_samples = 10000,
        ...                          trajectory_length=6, stepsize=0.25, return_type='dataframe')
        >>> np.cov(samples.values.T)
        array([[ 1.00795398,  0.71384233,  0.79802097],
               [ 0.71384233,  1.00633524,  0.21313767],
               [ 0.79802097,  0.21313767,  0.98519017]])
        """

        self.accepted_proposals = 1.0
        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        types = [(var_name, 'float') for var_name in self.model.variables]
        samples = np.zeros(num_samples, dtype=types).view(np.recarray)

        # Assigning after converting into tuple because value was being changed after assignment
        # Reason for this is unknown
        samples[0] = tuple(initial_pos)
        position_m = initial_pos

        lsteps = int(max(1, round(trajectory_length / stepsize, 0)))
        for i in range(1, num_samples):

            # Genrating sample
            position_m, _ = self._sample(position_m, trajectory_length, stepsize, lsteps)
            samples[i] = position_m

        self.acceptance_rate = self.accepted_proposals / num_samples

        return _return_samples(return_type, samples)

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
        genrator: yielding a 1d numpy.array type object for a sample

        Examples
        --------
        >>> from pgmpy.sampling import HamiltonianMC as HMC, GradLogPDFGaussian as GLPG
        >>> from pgmpy.factors import GaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([4, -1])
        >>> covariance = np.array([[3, 0.4], [0.4, 3]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMC(model=model, grad_log_pdf=GLPG)
        >>> gen_samples = sampler.generate_sample(np.array([-1, 1]), num_samples = 10000,
        ...                                       trajectory_length=2, stepsize=0.25)
        >>> samples_array = np.array([sample for sample in gen_samples])
        >>> samples_array
        array([[ 0.1467264 ,  0.27143857],
               [ 4.0371448 ,  0.15871274],
               [ 3.24656208, -1.03742621],
               ...,
               [ 6.45975905,  1.97941306],
               [ 4.89007171,  0.15413156],
               [ 5.9528083 ,  1.92983158]])
        >>> np.cov(samples_array.T)
        array([[ 2.95692642,  0.4379419 ],
               [ 0.4379419 ,  3.00939434]])
        >>> sampler.acceptance_rate
        0.9969
        """

        self.accepted_proposals = 0
        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        lsteps = int(max(1, round(trajectory_length / stepsize, 0)))
        position_m = initial_pos.copy()

        for i in range(0, num_samples):

            position_m, _ = self._sample(position_m, trajectory_length, stepsize, lsteps)

            yield position_m

        self.acceptance_rate = self.accepted_proposals / num_samples


class HamiltonianMCDA(HamiltonianMC):
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
    >>> from pgmpy.sampling import HamiltonianMCDA as HMCda, LeapFrog, GradLogPDFGaussian as GLPG
    >>> from pgmpy.factors.continuous import GaussianDistribution as JGD
    >>> import numpy as np
    >>> mean = np.array([1, 2, 3])
    >>> covariance = np.array([[2, 0.4, 0.5], [0.4, 3, 0.6], [0.5, 0.6, 4]])
    >>> model = JGD(['x', 'y', 'z'], mean, covariance)
    >>> sampler = HMCda(model=model, grad_log_pdf=GLPG)
    >>> samples = sampler.sample(np.array([0, 0, 0]), num_adapt=10000, num_samples = 10000, trajectory_length=7,
    ...                          return_type='recarray')
    >>> samples_array = np.array([samples[var_name] for var_name in model.variables])
    >>> np.cov(samples_array)
    array([[ 1.83023816,  0.40449162,  0.51200707],
           [ 0.40449162,  2.85863596,  0.76747343],
           [ 0.51200707,  0.76747343,  3.87020982]])
    >>> sampler.acceptance_rate
    0.9929

    References
    -----------
    Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
    Machine Learning Research 15 (2014) 1351-1381
    Algorithm 5 : Hamiltonian Monte Carlo with dual averaging
    """

    def __init__(self, model, grad_log_pdf=None, simulate_dynamics=LeapFrog, delta=0.65):

        if not isinstance(delta, float) or delta > 1.0 or delta < 0.0:
            raise ValueError(
                "delta should be a floating value in between 0 and 1")

        self.delta = delta

        super(HamiltonianMCDA, self).__init__(model=model, grad_log_pdf=grad_log_pdf,
                                              simulate_dynamics=simulate_dynamics)

    def _adapt_params(self, stepsize, stepsize_bar, h_bar, mu, index_i, alpha, n_alpha=1):
        """
        Run tha adaptation for stepsize for better proposals of position
        """
        gamma = 0.05  # free parameter that controls the amount of shrinkage towards mu
        t0 = 10.0  # free parameter that stabilizes the initial iterations of the algorithm
        kappa = 0.75
        # See equation (6) section 3.2.1 for details

        estimate = 1.0 / (index_i + t0)
        h_bar = (1 - estimate) * h_bar + estimate * (self.delta - alpha / n_alpha)

        stepsize = np.exp(mu - sqrt(index_i) / gamma * h_bar)
        i_kappa = index_i ** (-kappa)
        stepsize_bar = np.exp(i_kappa * np.log(stepsize) + (1 - i_kappa) * np.log(stepsize_bar))

        return stepsize, stepsize_bar, h_bar

    def sample(self, initial_pos, num_adapt, num_samples, trajectory_length, stepsize=None, return_type='dataframe'):
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

        return_type: string (dataframe | recarray)
            Return type for samples, either of 'dataframe' or 'recarray'.
            Defaults to 'dataframe'

        Returns
        -------
        sampled: A pandas.DataFrame or a numpy.recarray object depending upon return_type argument

        Examples
        ---------
        >>> from pgmpy.sampling import HamiltonianMCDA as HMCda, GradLogPDFGaussian as GLPG, LeapFrog
        >>> from pgmpy.factors.continuous import GaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 3]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMCda(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
        >>> samples = sampler.sample(np.array([1, 1]), num_adapt=10000, num_samples = 10000,
        ...                          trajectory_length=2, stepsize=None, return_type='recarray')
        >>> samples_array = np.array([samples[var_name] for var_name in model.variables])
        >>> np.cov(samples_array)
        array([[ 0.98432155,  0.66517394],
               [ 0.66517394,  2.95449533]])

        """

        self.accepted_proposals = 1.0

        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        if num_adapt <= 1:  # Return samples genrated using Simple HMC algorithm
            return HamiltonianMC.sample(self, initial_pos, num_samples, trajectory_length, stepsize)

        # stepsize is epsilon
        # freely chosen point, after each iteration xt(/position) is shrunk towards it
        mu = np.log(10.0 * stepsize)
        # log(10 * stepsize) large values to save computation
        # stepsize_bar is epsilon_bar
        stepsize_bar = 1.0
        h_bar = 0.0
        # See equation (6) section 3.2.1 for details

        types = [(var_name, 'float') for var_name in self.model.variables]
        samples = np.zeros(num_samples, dtype=types).view(np.recarray)
        samples[0] = tuple(initial_pos)
        position_m = initial_pos

        for i in range(1, num_samples):

            # Genrating sample
            position_m, alpha = self._sample(position_m, trajectory_length, stepsize)
            samples[i] = position_m

            # Adaptation of stepsize till num_adapt iterations
            if i <= num_adapt:
                stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar, mu, i, alpha)
            else:
                stepsize = stepsize_bar

        self.acceptance_rate = self.accepted_proposals / num_samples

        return _return_samples(return_type, samples)

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
        >>> from pgmpy.sampling import HamiltonianMCDA as HMCda, GradLogPDFGaussian as GLPG, LeapFrog
        >>> from pgmpy.factors.continuous import GaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 3]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMCda(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
        >>> gen_samples = sampler.generate_sample(np.array([1, 1]), num_adapt=10000,
        ...                                       num_samples = 10000, trajectory_length=2, stepsize=None)
        >>> samples_array = np.array([sample for sample in gen_samples])
        >>> np.cov(samples_array.T)
        array([[ 0.98432155,  0.69517394],
               [ 0.69517394,  2.95449533]])
        """
        self.accepted_proposals = 0
        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        if num_adapt <= 1:  # return sample generated using Simple HMC algorithm
            for sample in HamiltonianMC.generate_sample(self, initial_pos, num_samples, trajectory_length, stepsize):
                yield sample
            return
        mu = np.log(10.0 * stepsize)

        stepsize_bar = 1.0
        h_bar = 0.0

        position_m = initial_pos.copy()
        num_adapt += 1

        for i in range(1, num_samples + 1):

            position_m, alpha = self._sample(position_m, trajectory_length, stepsize)

            if i <= num_adapt:
                stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar, mu, i, alpha)
            else:
                stepsize = stepsize_bar

            yield position_m

        self.acceptance_rate = self.accepted_proposals / num_samples
