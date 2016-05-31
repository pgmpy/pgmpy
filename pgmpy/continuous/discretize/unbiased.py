from __future__ import division

import numpy as np
from scipy import integrate

from pgmpy.factors import ContinuousFactor
from pgmpy.continuous.discretize import BaseDiscretizer


class UnbiasedDiscretizer(BaseDiscretizer):
    """
    This class uses the unbiased method for discretizing the
    given continuous distribution.

    The unbiased method for discretization is the matching of the
    first moment method. It involves calculating the first order
    limited moment of the distribution which is done by the _lim_moment
    method.

    For this method,

    The probability mass in x=[low] is,
    (E(x) - E(x + step))/step + 1 - cdf(x)

    The probability mass in x=[low+step, low+2*step, ........., high-step],
    (2 * E(x) - E(x - step) - E(x + step))/step

    for x=[high],
    (E(x) - E(x - step))/step - 1 + cdf(x)

    where, E(x) is the first limiting moment of the distribution
    about the point x, cdf is the cumulative density function
    and step = (high-low)/cardinality.

    For details, refer Klugman, S. A., Panjer, H. H. and Willmot,
    G. E., Loss Models, From Data to Decisions, Fourth Edition,
    Wiley, section 9.6.5.2 (Method of local monment matching) and
    exercise 9.41.

    Examples
    --------
    >>> from pgmpy.factors import ContinuousFactor
    >>> from pgmpy.continuous.discretize import UnbiasedDiscretizer
    # exponential distribution with rate = 2
    >>> exp_pdf = lambda x: 2*np.exp(-2*x) if x>=0 else 0
    >>> exp_node = ContinuousFactor(exp_pdf)
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
        discrete_values = [(lev(self.low) - lev(self.low + step)) / step
                           + 1 - self.factor.cdf(self.low)]

        # for x=[low+step, low+2*step, ........., high-step]
        x = np.linspace(self.low + step, self.high - step, self.cardinality-2)
        for i in x:
            discrete_values.append((2 * lev(i) - lev(i - step) -
                                    lev(i + step)) / step)

        # for x=[high]
        discrete_values.append((lev(self.high) - lev(self.high - step)) /
                               step - 1 + self.factor.cdf(self.high))

        return discrete_values

    def _lim_moment(self, u, order=1):
        """
        This method calculates the kth order limiting moment of
        the distribution. It is given by -

        E(u) = Integral (-inf to u) [ (x^k)*pdf(x) dx ] + (u^k)(1-cdf(u))

        where, pdf is the probability density function and cdf is the
        cumulative density function of the distribution.

        For details, refer Klugman, S. A., Panjer, H. H. and Willmot,
        G. E., Loss Models, From Data to Decisions, Fourth Edition,
        Wiley, definition 3.5 and equation 3.8.

        Parameters
        ----------
        u: float
            The point at which the moment is to be calculated.
        order: int
            The order of the moment, default is first order.
        """
        fun = lambda x: np.power(x, order)*self.factor.pdf(x)
        return (integrate.quad(fun, -np.inf, u)[0] +
                np.power(u, order)*(1 - self.factor.cdf(u)))

    def get_labels(self):
        labels = list('x={i}'.format(i=str(i))
                      for i in np.round(np.linspace(self.low, self.high, self.cardinality), 3))
        return labels
