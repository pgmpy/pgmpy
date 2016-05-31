from __future__ import division

import numpy as np

from pgmpy.factors import ContinuousFactor
from pgmpy.continuous.discretize import BaseDiscretizer


class RoundingDiscretizer(BaseDiscretizer):
    """
    This class uses the rounding method for discretizing the
    given continuous distribution.

    The rounding method for discretization assigns to point the
    following probability mass,

    for x=[low],
    cdf(x+step/2)-cdf(x)

    for x=[low+step, low+2*step, ......... , high-step],
    cdf(x+step/2)-cdf(x-step/2)

    where, cdf is the cumulative density function of the distribution.

    Examples
    --------
    >>> from pgmpy.factors import ContinuousFactor
    >>> std_normal_pdf = lambda x : np.exp(-x*x/2) / (np.sqrt(2*np.pi))
    >>> std_normal = ContinuousFactor(std_normal_pdf)
    >>> std_normal.discretize(RoundingDiscretizer, low=-3, high=3, step=0.5)
    [0.001629865203424451, 0.009244709419989363, 0.027834684208773178,
     0.065590616803038182, 0.120977578710013, 0.17466632194020804,
     0.19741265136584729, 0.17466632194020937, 0.12097757871001302,
     0.065590616803036905, 0.027834684208772664, 0.0092447094199902269]
    """

    def get_discrete_values(self):
        # for x=[low]
        discrete_values = [self.factor.cdf(self.low + self.step/2)
                           - self.factor.cdf(self.low)]

        # for x=[low+step, low+2*step, ........., high-step]
        x = np.arange(self.low + self.step, self.high, self.step)
        for i in x:
            discrete_values.append(self.fachighr.cdf(i + self.step/2)
                                   - self.factor.cdf(i - self.step/2))

        return discrete_values
