from __future__ import division

import numpy as np

from pgmpy.factors import ContinuousFactor
from pgmpy.continuous.discretize import BaseDiscretizer


class RoundingDiscretizer(BaseDiscretizer):
    """
    This class uses the rounding method for discretizing the
    given continuous distribution.

    For the rounding method,

    The probability mass in x=[low] is,
    cdf(x+step/2)-cdf(x)

    The probability mass in x=[low+step, low+2*step, ......... , high-step] is,
    cdf(x+step/2)-cdf(x-step/2)

    where, cdf is the cumulative density function of the distribution
    and step = (high-low)/cardinality.

    Examples
    --------
    >>> from pgmpy.factors import ContinuousFactor
    >>> std_normal_pdf = lambda x : np.exp(-x*x/2) / (np.sqrt(2*np.pi))
    >>> std_normal = ContinuousFactor(std_normal_pdf)
    >>> std_normal.discretize(RoundingDiscretizer, low=-3, high=3, cardinality=12)
    [0.001629865203424451, 0.009244709419989363, 0.027834684208773178,
     0.065590616803038182, 0.120977578710013, 0.17466632194020804,
     0.19741265136584729, 0.17466632194020937, 0.12097757871001302,
     0.065590616803036905, 0.027834684208772664, 0.0092447094199902269]
    """

    def get_discrete_values(self):
        step = (self.high - self.low) / self.cardinality

        # for x=[low]
        discrete_values = [self.factor.cdf(self.low + step/2)
                           - self.factor.cdf(self.low)]

        # for x=[low+step, low+2*step, ........., high-step]
        x = np.arange(self.low + step, self.high, step)
        for i in x:
            discrete_values.append(self.factor.cdf(i + step/2)
                                   - self.factor.cdf(i - step/2))

        return discrete_values
