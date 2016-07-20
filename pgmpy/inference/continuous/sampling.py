# -*- coding: UTF-8 -*-
"""
    A collection of methods for sampling from continuous models in pgmpy
"""
from __future__ import division
from math import sqrt

import numpy as np
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False

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
    >>> from pgmpy.inference.continuous import HamiltonianMC as HMC, LeapFrog, GradLogPDFGaussian
    >>> from pgmpy.factors import JointGaussianDistribution as JGD
    >>> import numpy as np
    >>> mean = np.array([-3, 4])
    >>> covariance = np.array([[3, 0.7], [0.7, 5]])
    >>> model = JGD(['x', 'y'], mean, covariance)
    >>> sampler = HMC(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
    >>> samples = sampler.sample(initial_pos=np.array([1, 1]), num_samples = 10000,
    ...                          trajectory_length=2, stepsize=0.4)
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

        condition = (acceptance_prob ** a) > (2 ** (-a))

        while condition:
            stepsize_app = (2 ** a) * stepsize_app

            position_bar, momentum_bar, _ =\
                self.simulate_dynamics(self.model, position, momentum,
                                       stepsize_app, self.grad_log_pdf).get_proposed_values()

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
        Returns two different types (based on installations)

        pandas.DataFrame: Returns samples as pandas.DataFrame if environment has a installation of pandas

        numpy.recarray: Returns samples in form of numpy recorded arrays (numpy.recarray)

        Examples
        --------
        >>> # Example if pandas is installed in working environment
        >>> from pgmpy.inference.continuous import HamiltonianMC as HMC, GradLogPDFGaussian, ModifiedEuler
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, -1])
        >>> covariance = np.array([[1, 0.2], [0.2, 1]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMC(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=ModifiedEuler)
        >>> samples = sampler.sample(np.array([1, 1]), num_samples = 5,
        ...                          trajectory_length=6, stepsize=0.25)
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
        ...                          trajectory_length=6, stepsize=0.25)
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

        if HAS_PANDAS is True:
            return pd.DataFrame.from_records(samples)

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
        genrator: yielding a 1d numpy.array type object for a sample

        Examples
        --------
        >>> from pgmpy.inference.continuous import HamiltonianMC as HMC, GradLogPDFGaussian as GLPG
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
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
    >>> from pgmpy.inference.continuous import HamiltonianMCda as HMCda, LeapFrog
    >>> from pgmpy.factors import JointGaussianDistribution as JGD
    >>> import numpy as np
    >>> mean = np.array([1, 2, 3])
    >>> covariance = np.array([[2, 0.4, 0.5], [0.4, 3, 0.6], [0.5, 0.6, 4]])
    >>> model = JGD(['x', 'y', 'z'], mean, covariance)
    >>> sampler = HMCda(model=model)
    >>> samples = sampler.sample(np.array([0, 0, 0]), num_adapt=10000, num_samples = 10000, trajectory_length=7)
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

        super(HamiltonianMCda, self).__init__(model=model, grad_log_pdf=grad_log_pdf,
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
        Returns two different types (based on installations)

        pandas.DataFrame: Returns samples as pandas.DataFrame if environment has a installation of pandas

        numpy.recarray: Returns samples in form of numpy recorded arrays (numpy.recarray)

        Examples
        ---------
        >>> from pgmpy.inference.continuous import HamiltonianMCda as HMCda, GradLogPDFGaussian as GLPG, LeapFrog
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, 1])
        >>> covariance = np.array([[1, 0.7], [0.7, 3]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = HMCda(model=model, grad_log_pdf=GLPG, simulate_dynamics=LeapFrog)
        >>> samples = sampler.sample(np.array([1, 1]), num_adapt=10000,
        ...                          num_samples = 10000, trajectory_length=2, stepsize=None)
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

        if HAS_PANDAS is True:
            return pd.DataFrame.from_records(samples)

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
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
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


class NoUTurnSampler(HamiltonianMCda):
    """
    Class for performing sampling in Continuous model
    using No U Turn Sampler (a variant of Hamiltonian Monte Carlo)

    Parameters:
    -----------
    model: An instance pgmpy.models
        Model from which sampling has to be done

    grad_log_pdf: A subclass of pgmpy.inference.continuous.GradientLogPDF
        Class to compute the log and gradient log of distribution

    simulate_dynamics: A subclass of pgmpy.inference.continuous.BaseSimulateHamiltonianDynamics
        Class to propose future states of position and momentum in time by simulating
        HamiltonianDynamics

    Public Methods:
    ---------------
    sample()
    generate_sample()

    Example:
    --------
    >>> from pgmpy.inference.continuous import NoUTurnSampler as NUTS, LeapFrog, GradLogPDFGaussian
    >>> from pgmpy.factors import JointGaussianDistribution as JGD
    >>> import numpy as np
    >>> mean = np.array([1, 2, 3])
    >>> covariance = np.array([[4, 0.1, 0.2], [0.1, 1, 0.3], [0.2, 0.3, 8]])
    >>> model = JGD(['x', 'y', 'z'], mean, covariance)
    >>> sampler = NUTS(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
    >>> samples = sampler.sample(initial_pos=np.array([0.1, 0.9, 0.3]), num_samples=20000,stepsize=0.4)
    >>> samples
    rec.array([(0.1, 0.9, 0.3),
     (-0.27303886844752756, 0.5028580705249155, 0.2895768065049909),
     (1.7139810571103862, 2.809135711303245, 5.690811523613858), ...,
     (-0.7742669710786649, 2.092867703984895, 6.139480724333439),
     (1.3916152816323692, 1.394952482021687, 3.446906546649354),
     (-0.2726336476939125, 2.6230854954595357, 2.923948403903159)],
              dtype=[('x', '<f8'), ('y', '<f8'), ('z', '<f8')])

    References
    ----------
    Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
    Machine Learning Research 15 (2014) 1351-1381
    Algorithm 3 : Efficient No-U-Turn Sampler
    """

    def __init__(self, model, grad_log_pdf, simulate_dynamics=LeapFrog):

        super(NoUTurnSampler, self).__init__(model=model, grad_log_pdf=grad_log_pdf,
                                             simulate_dynamics=simulate_dynamics)

    def _initalize_tree(self, position, momentum, slice_var, stepsize):
        """
        Initalizes root node of the tree, i.e depth = 0
        """

        position_bar, momentum_bar, _ = self.simulate_dynamics(self.model, position, momentum, stepsize,
                                                               self.grad_log_pdf).get_proposed_values()

        _, logp_bar = self.grad_log_pdf(position_bar, self.model).get_gradient_log_pdf()

        hamiltonian = logp_bar - 0.5 * np.dot(momentum_bar, momentum_bar)

        candidate_set_size = slice_var < np.exp(hamiltonian)
        accept_set_bool = hamiltonian > np.log(slice_var) - 10000  # delta_max = 10000

        return position_bar, momentum_bar, candidate_set_size, accept_set_bool

    def _update_acceptance_criteria(self, position_forward, position_backward, momentum_forward, momentum_backward,
                                    accept_set_bool, candidate_set_size, candidate_set_size2):

        # criteria1 = I[(θ+ − θ−)·r− ≥ 0]
        criteria1 = np.dot((position_forward - position_backward), momentum_backward) >= 0

        # criteira2 = I[(θ+ − θ− )·r+ ≥ 0]
        criteria2 = np.dot((position_forward - position_backward), momentum_forward) >= 0

        accept_set_bool = accept_set_bool and criteria1 and criteria2
        candidate_set_size += candidate_set_size2

        return accept_set_bool, candidate_set_size

    def _build_tree(self, position, momentum, slice_var, direction, depth, stepsize):
        """
        Recursively builds a tree for proposing new position and momentum
        """

        # Parameter names in algorithm (here -> representation in algorithm)
        # position -> theta, momentum -> r, slice_var -> u, direction -> v, depth ->j, stepsize -> epsilon
        # candidate_set_size -> n, accept_set_bool -> s
        if depth == 0:
            # Take single leapfrog step in the given direction (direction * stepsize)
            position_bar, momentum_bar, candidate_set_size, accept_set_bool =\
                self._initalize_tree(position, momentum, slice_var, direction * stepsize)

            return (position_bar, momentum_bar, position_bar, momentum_bar, position_bar,
                    candidate_set_size, accept_set_bool)

        else:
            # Build left and right subtrees
            (position_backward, momentum_backward, position_forward, momentum_forward, position_bar,
             candidate_set_size, accept_set_bool) = self._build_tree(position, momentum,
                                                                     slice_var, direction, depth - 1, stepsize)
            if accept_set_bool == 1:
                if direction == -1:
                    # Build tree in backward direction
                    (position_backward, momentum_backward, _, _, position_bar2, candidate_set_size2,
                     accept_set_bool2) = self._build_tree(position_backward, momentum_backward,
                                                          slice_var, direction, depth - 1, stepsize)
                else:
                    # Build tree in forward direction
                    (_, _, position_forward, momentum_forward, position_bar2, candidate_set_size2,
                     accept_set_bool2) = self._build_tree(position_forward, momentum_forward,
                                                          slice_var, direction, depth - 1, stepsize)

                if np.random.rand() < candidate_set_size2 / (candidate_set_size2 + candidate_set_size):
                    position_bar = position_bar2

                accept_set_bool, candidate_set_size =\
                    self._update_acceptance_criteria(position_forward, position_backward, momentum_forward,
                                                     momentum_backward, accept_set_bool2, candidate_set_size,
                                                     candidate_set_size2)

            return (position_backward, momentum_backward, position_forward, momentum_forward,
                    position_bar, candidate_set_size, accept_set_bool)

    def _sample(self, position, stepsize):
        """
        Returns a sample using a single iteration of NUTS
        """

        # Re-sampling momentum
        momentum = np.random.normal(0, 1, len(position))

        # Initializations
        depth = 0
        position_backward, position_forward = position, position
        momentum_backward, momentum_forward = momentum, momentum
        candidate_set_size = accept_set_bool = 1
        _, log_pdf = self.grad_log_pdf(position, self.model).get_gradient_log_pdf()

        # Resample slice variable `u`
        slice_var = np.random.uniform(0, np.exp(log_pdf - 0.5 * np.dot(momentum, momentum)))

        while accept_set_bool == 1:
            direction = np.random.choice([-1, 1], p=[0.5, 0.5])
            if direction == -1:
                # Build a tree in backward direction
                (position_backward, momentum_backward, _, _, position_bar,
                 candidate_set_size2, accept_set_bool2) = self._build_tree(position_backward, momentum_backward,
                                                                           slice_var, direction, depth, stepsize)
            else:
                # Build tree in forward direction
                (_, _, position_forward, momentum_forward, position_bar,
                 candidate_set_size2, accept_set_bool2) = self._build_tree(position_forward, momentum_forward,
                                                                           slice_var, direction, depth, stepsize)
            if accept_set_bool2 == 1:
                if np.random.rand() < candidate_set_size2 / candidate_set_size:
                    position = position_bar.copy()

            accept_set_bool, candidate_set_size = self._update_acceptance_criteria(position_forward, position_backward,
                                                                                   momentum_forward, momentum_backward,
                                                                                   accept_set_bool2, candidate_set_size,
                                                                                   candidate_set_size2)
            depth += 1

        return position

    def sample(self, initial_pos, num_samples, stepsize=None):
        """
        Method to return samples using No U Turn Sampler

        Parameters
        ----------
        initial_pos: A 1d array like object
            Vector representing values of parameter position, the starting
            state in markov chain.

        num_samples: int
            Number of samples to be generated

        stepsize: float , defaults to None
            The stepsize for proposing new values of position and momentum in simulate_dynamics
            If None, then will be choosen suitably

        Returns
        -------
        Returns two different types (based on installations)

        pandas.DataFrame: Returns samples as pandas.DataFrame if environment has a installation of pandas

        numpy.recarray: Returns samples in form of numpy recorded arrays (numpy.recarray)

        Examples
        ---------
        >>> # If environment has a installation of pandas
        >>> from pgmpy.inference.continuous import NoUTurnSampler as NUTS, GradLogPDFGaussian, LeapFrog
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([0, 0, 0])
        >>> covariance = np.array([[6, 0.7, 0.2], [0.7, 3, 0.9], [0.2, 0.9, 1]])
        >>> model = JGD(['x', 'y', 'z'], mean, covariance)
        >>> sampler = NUTS(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
        >>> samples = sampler.sample(initial_pos=np.array([1, 1, 1]), num_samples=10, stepsize=0.4)
        >>> samples
                  x         y         z
        0  1.000000  1.000000  1.000000
        1  1.760756  0.271543 -0.613309
        2  1.883387  0.990745 -0.611720
        3  0.980812  0.340336 -0.916283
        4  0.781338  0.647220 -0.948640
        5  0.040308 -1.391406  0.412201
        6  1.179549 -1.450552  1.105216
        7  1.100320 -1.313926  1.207815
        8  1.484520 -1.349247  0.768599
        9  0.934942 -1.894589  0.471772
        """
        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        types = [(var_name, 'float') for var_name in self.model.variables]
        samples = np.zeros(num_samples, dtype=types).view(np.recarray)

        samples[0] = tuple(initial_pos)
        position_m = initial_pos

        for i in range(1, num_samples):
            # Genrating sample
            position_m = self._sample(position_m, stepsize)
            samples[i] = position_m

        if HAS_PANDAS is True:
            return pd.DataFrame.from_records(samples)

        return samples

    def generate_sample(self, initial_pos, num_samples, stepsize=None):
        """
        Returns a generator type object whose each iteration yields a sample

        Parameters
        ----------
        initial_pos: A 1d array like object
            Vector representing values of parameter position, the starting
            state in markov chain.

        num_samples: int
            Number of samples to be generated

        stepsize: float , defaults to None
            The stepsize for proposing new values of position and momentum in simulate_dynamics
            If None, then will be choosen suitably

        Returns
        -------
        genrator: yielding a numpy.array type object for a sample

        Examples
        ---------
        >>> from pgmpy.inference.continuous import NoUTurnSampler as NUTS, GradLogPDFGaussian
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([11, -6])
        >>> covariance = np.array([[0.7, 0.2], [0.2, 14]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = NUTS(model=model, grad_log_pdf=GradLogPDFGaussian)
        >>> samples = sampler.generate_sample(initial_pos=np.array([1, 1]), num_samples=10, stepsize=0.4)
        >>> samples = np.array([sample for sample in samples])
        >>> samples
        array([[ 10.26357538,   0.10062725],
               [ 12.70600336,   0.63392499],
               [ 10.95523217,  -0.62079273],
               [ 10.66263031,  -4.08135962],
               [ 10.59255762,  -8.48085076],
               [  9.99860242,  -9.47096032],
               [ 10.5733564 ,  -9.83504745],
               [ 11.51302059,  -9.49919523],
               [ 11.31892143,  -8.5873259 ],
               [ 11.29008667,  -0.43809674]])
        """
        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        position_m = initial_pos

        for _ in range(0, num_samples):

            position_m = self._sample(position_m, stepsize)

            yield position_m


class NoUTurnSamplerDA(NoUTurnSampler):
    """
    Class for performing sampling in Continuous model
    using No U Turn sampler with dual averaging for
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
    >>> from pgmpy.inference.continuous import NoUTurnSamplerDA as NUTSda, GradLogPDFGaussian
    >>> from pgmpy.factors import JointGaussianDistribution as JGD
    >>> import numpy as np
    >>> mean = np.array([-1, 12, -3])
    >>> covariance = np.array([[-2, 7, 2], [7, 14, 4], [2, 4, -1]])
    >>> model = JGD(['x', 'v', 't'], mean, covariance)
    >>> sampler = NUTSda(model=model, grad_log_pdf=GradLogPDFGaussian)
    >>> samples = sampler.sample(initial_pos=np.array([0, 0, 0]), num_adapt=10, num_samples=10, stepsize=0.25)
    >>> samples
    rec.array([(0.0, 0.0, 0.0),
     (0.06100992691638076, -0.17118088764170125, 0.14048470935160887),
     (0.06100992691638076, -0.17118088764170125, 0.14048470935160887),
     (-0.7451883138013118, 1.7975387358691155, 2.3090698721374436),
     (-0.6207457594500309, 1.4611049498441024, 2.5890867012835574),
     (0.24043604780911487, 1.8660976216530618, 3.2508715592645347),
     (0.21509819341468212, 2.157760225367607, 3.5749582768731476),
     (0.20699150582681913, 2.0605044285377305, 3.8588980251618135),
     (0.20699150582681913, 2.0605044285377305, 3.8588980251618135),
     (0.085332419611991, 1.7556171374575567, 4.49985082288814)],
              dtype=[('x', '<f8'), ('v', '<f8'), ('t', '<f8')])

    References
    ----------
    Matthew D. Hoffman, Andrew Gelman, The No-U-Turn Sampler: Adaptively
    Setting Path Lengths in Hamiltonian Monte Carlo. Journal of
    Machine Learning Research 15 (2014) 1351-1381
    Algorithm 6 : No-U-Turn Sampler with Dual Averaging
    """

    def __init__(self, model, grad_log_pdf, simulate_dynamics=LeapFrog, delta=0.65):

        if not isinstance(delta, float) or delta > 1.0 or delta < 0.0:
            raise ValueError(
                "delta should be a floating value in between 0 and 1")

        self.delta = delta

        super(NoUTurnSamplerDA, self).__init__(model=model, grad_log_pdf=grad_log_pdf,
                                               simulate_dynamics=simulate_dynamics)

    def _build_tree(self, position, momentum, slice_var, direction, depth, stepsize, position0, momentum0):
        """
        Recursively builds a tree for proposing new position and momentum
        """
        if depth == 0:

            position_bar, momentum_bar, candidate_set_size, accept_set_bool =\
                self._initalize_tree(position, momentum, slice_var, direction * stepsize)

            alpha = min(1, self._acceptance_prob(position, position_bar, momentum, momentum_bar))

            return (position_bar, momentum_bar, position_bar, momentum_bar, position_bar,
                    candidate_set_size, accept_set_bool, alpha, 1)

        else:
            (position_backward, momentum_backward, position_forward, momentum_forward, position_bar,
             candidate_set_size, accept_set_bool, alpha, n_alpha) =\
                self._build_tree(position, momentum, slice_var,
                                 direction, depth - 1, stepsize, position0, momentum0)

            if accept_set_bool == 1:
                if direction == -1:
                    # Build tree in backward direction
                    (position_backward, momentum_backward, _, _, position_bar2, candidate_set_size2, accept_set_bool2,
                     alpha2, n_alpha2) = self._build_tree(position_backward, momentum_backward, slice_var, direction,
                                                          depth - 1, stepsize, position0, momentum0)
                else:
                    # Build tree in forward direction
                    (_, _, position_forward, momentum_forward, position_bar2, candidate_set_size2, accept_set_bool2,
                     alpha2, n_alpha2) = self._build_tree(position_forward, momentum_forward, slice_var, direction,
                                                          depth - 1, stepsize, position0, momentum0)

                if np.random.rand() < candidate_set_size2 / (candidate_set_size2 + candidate_set_size):
                    position_bar = position_bar2

                alpha += alpha2
                n_alpha += n_alpha2
                accept_set_bool, candidate_set_size =\
                    self._update_acceptance_criteria(position_forward, position_backward, momentum_forward,
                                                     momentum_backward, accept_set_bool2, candidate_set_size,
                                                     candidate_set_size2)

            return (position_backward, momentum_backward, position_forward, momentum_forward, position_bar,
                    candidate_set_size, accept_set_bool, alpha, n_alpha)

    def _sample(self, position, stepsize):
        """
        Returns a sample using a single iteration of NUTS with dual averaging
        """

        # Re-sampling momentum
        momentum = np.random.normal(0, 1, len(position))

        # Initializations
        depth = 0
        position_backward, position_forward = position, position
        momentum_backward, momentum_forward = momentum, momentum
        candidate_set_size = accept_set_bool = 1
        position_m_1 = position
        _, log_pdf = self.grad_log_pdf(position, self.model).get_gradient_log_pdf()

        # Resample slice variable `u`
        slice_var = np.random.uniform(0, np.exp(log_pdf - 0.5 * np.dot(momentum, momentum)))

        while accept_set_bool == 1:
            direction = np.random.choice([-1, 1], p=[0.5, 0.5])
            if direction == -1:
                # Build a tree in backward direction
                (position_backward, momentum_backward, _, _, position_bar, candidate_set_size2, accept_set_bool2,
                 alpha, n_alpha) = self._build_tree(position_backward, momentum_backward, slice_var, direction,
                                                    depth, stepsize, position_m_1, momentum)
            else:
                # Build tree in forward direction
                (_, _, position_forward, momentum_forward, position_bar, candidate_set_size2, accept_set_bool2,
                 alpha, n_alpha) = self._build_tree(position_forward, momentum_forward, slice_var, direction,
                                                    depth, stepsize, position_m_1, momentum)

            if accept_set_bool2 == 1:
                if np.random.rand() < candidate_set_size2 / candidate_set_size:
                    position = position_bar

            accept_set_bool, candidate_set_size = self._update_acceptance_criteria(position_forward, position_backward,
                                                                                   momentum_forward, momentum_backward,
                                                                                   accept_set_bool2, candidate_set_size,
                                                                                   candidate_set_size2)

            depth += 1

        return position, alpha, n_alpha

    def sample(self, initial_pos, num_adapt, num_samples, stepsize=None):
        """
        Returns samples using No U Turn Sampler with dual averaging

        Parameters
        ----------
        initial_pos: A 1d array like object
            Vector representing values of parameter position, the starting
            state in markov chain.

        num_adapt: int
            The number of interations to run the adaptation of stepsize

        num_samples: int
            Number of samples to be generated

        stepsize: float , defaults to None
            The stepsize for proposing new values of position and momentum in simulate_dynamics
            If None, then will be choosen suitably

        Returns
        -------
        Returns two different types (based on installations)

        pandas.DataFrame: Returns samples as pandas.DataFrame if environment has a installation of pandas

        numpy.recarray: Returns samples in form of numpy recorded arrays (numpy.recarray)

        Examples
        ---------
        >>> # If environment has a installation of pandas
        >>> from pgmpy.inference.continuous import NoUTurnSamplerDA as NUTSda, GradLogPDFGaussian, LeapFrog
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([10, -13])
        >>> covariance = np.array([[16, -3], [-3, 13]])
        >>> model = JGD(['x', 'y'], mean, covariance)
        >>> sampler = NUTSda(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
        >>> samples = sampler.sample(initial_pos=np.array([12, -4]), num_adapt=10, num_samples=10, stepsize=0.1)
        >>> samples
                   x          y
        0  12.000000  -4.000000
        1  11.864821  -3.696109
        2  10.546986  -4.892169
        3   8.526596 -21.555793
        4   8.526596 -21.555793
        5  11.343194  -6.353789
        6  -1.583269 -12.802931
        7  12.411957 -11.704859
        8  13.253336 -20.169492
        9  11.295901  -7.665058
        """
        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        if num_adapt <= 1:
            return NoUTurnSampler(self.model, self.grad_log_pdf,
                                  self.simulate_dynamics).sample(initial_pos, num_samples, stepsize)

        mu = np.log(10.0 * stepsize)
        stepsize_bar = 1.0
        h_bar = 0.0

        types = [(var_name, 'float') for var_name in self.model.variables]
        samples = np.zeros(num_samples, dtype=types).view(np.recarray)
        samples[0] = tuple(initial_pos)
        position_m = initial_pos

        for i in range(1, num_samples):

            position_m, alpha, n_alpha = self._sample(position_m, stepsize)
            samples[i] = position_m

            if i <= num_adapt:
                stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar, mu,
                                                                   i, alpha, n_alpha)
            else:
                stepsize = stepsize_bar

        if HAS_PANDAS is True:
            return pd.DataFrame.from_records(samples)

        return samples

    def generate_sample(self, initial_pos, num_adapt, num_samples, stepsize=None):
        """
        Returns a generator type object whose each iteration yields a sample

        Parameters
        ----------
        initial_pos: A 1d array like object
            Vector representing values of parameter position, the starting
            state in markov chain.

        num_adapt: int
            The number of interations to run the adaptation of stepsize

        num_samples: int
            Number of samples to be generated

        stepsize: float , defaults to None
            The stepsize for proposing new values of position and momentum in simulate_dynamics
            If None, then will be choosen suitably

        Returns
        -------
        genrator: yielding a numpy.array type object for a sample

        Examples
        --------
        >>> from pgmpy.inference.continuous import NoUTurnSamplerDA as NUTSda, GradLogPDFGaussian
        >>> from pgmpy.factors import JointGaussianDistribution as JGD
        >>> import numpy as np
        >>> mean = np.array([1, -100])
        >>> covariance = np.array([[-12, 45], [45, -10]])
        >>> model = JGD(['a', 'b'], mean, covariance)
        >>> sampler = NUTSda(model=model, grad_log_pdf=GradLogPDFGaussian, simulate_dynamics=LeapFrog)
        >>> samples = sampler.generate_sample(initial_pos=np.array([12, -4]), num_adapt=10,
        ...                                   num_samples=10, stepsize=0.1)
        >>> samples
        <generator object NoUTurnSamplerDA.generate_sample at 0x7f4fed46a4c0>
        >>> samples_array = np.array([sample for sample in samples])
        >>> samples_array
        array([[ 11.89963386,  -4.06572636],
               [ 10.3453755 ,  -7.5700289 ],
               [-26.56899659, -15.3920684 ],
               [-29.97143077, -12.0801625 ],
               [-29.97143077, -12.0801625 ],
               [-33.07960829,  -8.90440347],
               [-55.28263496, -17.31718524],
               [-55.28263496, -17.31718524],
               [-56.63440044, -16.03309364],
               [-63.880094  , -19.19981944]])
        """
        initial_pos = _check_1d_array_object(initial_pos, 'initial_pos')
        _check_length_equal(initial_pos, self.model.variables, 'initial_pos', 'model.variables')

        if stepsize is None:
            stepsize = self._find_reasonable_stepsize(initial_pos)

        if num_adapt <= 1:  # return sample generated using Simple HMC algorithm
            for sample in NoUTurnSampler(self.model, self.grad_log_pdf,
                                         self.simulate_dynamics).generate_sample(initial_pos, num_samples, stepsize):
                yield sample
            return
        mu = np.log(10.0 * stepsize)

        stepsize_bar = 1.0
        h_bar = 0.0

        position_m = initial_pos.copy()
        num_adapt += 1

        for i in range(1, num_samples + 1):

            position_m, alpha, n_alpha = self._sample(position_m, stepsize)

            if i <= num_adapt:
                stepsize, stepsize_bar, h_bar = self._adapt_params(stepsize, stepsize_bar, h_bar, mu,
                                                                   i, alpha, n_alpha)
            else:
                stepsize = stepsize_bar

            yield position_m
