from __future__ import division

from six import with_metaclass
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy import integrate


class BaseDiscretizer(with_metaclass(ABCMeta)):
    """
    Base class for the discretizer classes in pgmpy. The discretizer
    classes are used to discretize a continuous random variable
    distribution into discrete probability masses.

    Parameters
    ----------
    factor: A ContinuousNode or a ContinuousFactor object
        the continuous node or factor representing the distribution
        to be discretized.

    low, high: float
        the range over which the function will be discretized.

    cardinality: int
        the number of states required in the discretized output.

    Examples
    --------
    >>> from scipy.stats import norm
    >>> from pgmpy.factors.continuous import ContinuousNode
    >>> normal = ContinuousNode(norm(0, 1).pdf)
    >>> from pgmpy.discretize import BaseDiscretizer
    >>> class ChildDiscretizer(BaseDiscretizer):
    ...     def get_discrete_values(self):
    ...         pass
    >>> discretizer = ChildDiscretizer(normal, -3, 3, 10)
    >>> discretizer.factor
    <pgmpy.factors.continuous.ContinuousNode.ContinuousNode object at 0x04C98190>
    >>> discretizer.cardinality
    10
    >>> discretizer.get_labels()
    ['x=-3.0', 'x=-2.4', 'x=-1.8', 'x=-1.2', 'x=-0.6', 'x=0.0', 'x=0.6', 'x=1.2', 'x=1.8', 'x=2.4']

    """

    def __init__(self, factor, low, high, cardinality):
        self.factor = factor
        self.low = low
        self.high = high
        self.cardinality = cardinality

    @abstractmethod
    def get_discrete_values(self):
        """
        This method implements the algorithm to discretize the given
        continuous distribution.

        It must be implemented by all the subclasses of BaseDiscretizer.

        Returns
        -------
        A list of discrete values or a DiscreteFactor object.
        """
        pass

    def get_labels(self):
        """
        Returns a list of strings representing the values about
        which the discretization method calculates the probabilty
        masses.

        Default value is the points -
        [low, low+step, low+2*step, ......... , high-step]
        unless the method is overridden by a subclass.

        Examples
        --------
        >>> from pgmpy.factors import ContinuousNode
        >>> from pgmpy.discretize import BaseDiscretizer
        >>> class ChildDiscretizer(BaseDiscretizer):
        ...     def get_discrete_values(self):
        ...         pass
        >>> from scipy.stats import norm
        >>> node = ContinuousNode(norm(0).pdf)
        >>> child = ChildDiscretizer(node, -5, 5, 20)
        >>> chld.get_labels()
        ['x=-5.0', 'x=-4.5', 'x=-4.0', 'x=-3.5', 'x=-3.0', 'x=-2.5',
         'x=-2.0', 'x=-1.5', 'x=-1.0', 'x=-0.5', 'x=0.0', 'x=0.5', 'x=1.0',
         'x=1.5', 'x=2.0', 'x=2.5', 'x=3.0', 'x=3.5', 'x=4.0', 'x=4.5']

        """
        step = (self.high - self.low) / self.cardinality
        labels = ['x={i}'.format(i=str(i)) for i in np.round(
            np.arange(self.low, self.high, step), 3)]
        return labels


class RoundingDiscretizer(BaseDiscretizer):
    """
    This class uses the rounding method for discretizing the
    given continuous distribution.

    For the rounding method,

    The probability mass is,
    cdf(x+step/2)-cdf(x), for x = low

    cdf(x+step/2)-cdf(x-step/2), for low < x <= high

    where, cdf is the cumulative density function of the distribution
    and step = (high-low)/cardinality.

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.factors.continuous import ContinuousNode
    >>> from pgmpy.factors.continuous import RoundingDiscretizer
    >>> std_normal_pdf = lambda x : np.exp(-x*x/2) / (np.sqrt(2*np.pi))
    >>> std_normal = ContinuousNode(std_normal_pdf)
    >>> std_normal.discretize(RoundingDiscretizer, low=-3, high=3,
    ...                       cardinality=12)
    [0.001629865203424451, 0.009244709419989363, 0.027834684208773178,
     0.065590616803038182, 0.120977578710013, 0.17466632194020804,
     0.19741265136584729, 0.17466632194020937, 0.12097757871001302,
     0.065590616803036905, 0.027834684208772664, 0.0092447094199902269]
    """

    def get_discrete_values(self):
        step = (self.high - self.low) / self.cardinality

        # for x=[low]
        discrete_values = [self.factor.cdf(self.low + step/2) - self.factor.cdf(self.low)]

        # for x=[low+step, low+2*step, ........., high-step]
        points = np.linspace(self.low + step, self.high - step, self.cardinality - 1)
        discrete_values.extend([self.factor.cdf(i + step/2) - self.factor.cdf(i - step/2) for i in points])

        return discrete_values


class UnbiasedDiscretizer(BaseDiscretizer):
    """
    This class uses the unbiased method for discretizing the
    given continuous distribution.

    The unbiased method for discretization is the matching of the
    first moment method. It involves calculating the first order
    limited moment of the distribution which is done by the _lim_moment
    method.

    For this method,

    The probability mass is,
    (E(x) - E(x + step))/step + 1 - cdf(x), for x = low

    (2 * E(x) - E(x - step) - E(x + step))/step, for low < x < high

    (E(x) - E(x - step))/step - 1 + cdf(x), for x = high

    where, E(x) is the first limiting moment of the distribution
    about the point x, cdf is the cumulative density function
    and step = (high-low)/cardinality.

    Reference
    ---------
    Klugman, S. A., Panjer, H. H. and Willmot, G. E.,
    Loss Models, From Data to Decisions, Fourth Edition,
    Wiley, section 9.6.5.2 (Method of local monment matching) and
    exercise 9.41.

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.factors import ContinuousNode
    >>> from pgmpy.factors.continuous import UnbiasedDiscretizer
    # exponential distribution with rate = 2
    >>> exp_pdf = lambda x: 2*np.exp(-2*x) if x>=0 else 0
    >>> exp_node = ContinuousNode(exp_pdf)
    >>> exp_node.discretize(UnbiasedDiscretizer, low=0, high=5, cardinality=10)
    [0.39627368905806137, 0.4049838434034298, 0.13331784003148325,
     0.043887287876647259, 0.014447413395300212, 0.0047559685431339703,
     0.0015656350182896128, 0.00051540201980112557, 0.00016965346326140994,
     3.7867260839208328e-05]

    """
    def get_discrete_values(self):
        lev = self._lim_moment
        step = (self.high - self.low) / (self.cardinality - 1)

        # for x=[low]
        discrete_values = [(lev(self.low) - lev(self.low + step)) / step +
                           1 - self.factor.cdf(self.low)]

        # for x=[low+step, low+2*step, ........., high-step]
        points = np.linspace(self.low + step, self.high - step, self.cardinality - 2)
        discrete_values.extend([(2 * lev(i) - lev(i - step) - lev(i + step)) / step for i in points])

        # for x=[high]
        discrete_values.append((lev(self.high) - lev(self.high - step)) / step - 1 + self.factor.cdf(self.high))

        return discrete_values

    def _lim_moment(self, u, order=1):
        """
        This method calculates the kth order limiting moment of
        the distribution. It is given by -

        E(u) = Integral (-inf to u) [ (x^k)*pdf(x) dx ] + (u^k)(1-cdf(u))

        where, pdf is the probability density function and cdf is the
        cumulative density function of the distribution.

        Reference
        ---------
        Klugman, S. A., Panjer, H. H. and Willmot, G. E.,
        Loss Models, From Data to Decisions, Fourth Edition,
        Wiley, definition 3.5 and equation 3.8.

        Parameters
        ----------
        u: float
            The point at which the moment is to be calculated.

        order: int
            The order of the moment, default is first order.
        """
        def fun(x):
            return np.power(x, order) * self.factor.pdf(x)
        return (integrate.quad(fun, -np.inf, u)[0] +
                np.power(u, order)*(1 - self.factor.cdf(u)))

    def get_labels(self):
        labels = list('x={i}'.format(i=str(i)) for i in np.round
                      (np.linspace(self.low, self.high, self.cardinality), 3))
        return labels
