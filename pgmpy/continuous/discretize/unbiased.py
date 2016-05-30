from __future__ import division

import numpy as np
from scipy import integrate

from pgmpy.factors import ContinuousFactor
from pgmpy.continuous.discretize import BaseDiscretizer


class UnbiasedDiscretizer(BaseDiscretizer):
    """
    This class uses the rounding method for discretizing the
    given continuous distribution.

    The unbiased method for discretization is the matching of the
    first moment method. It involves calculating the first order
    limited moment of the distribution which is done by the _lim_moment
    method.
    The method assigns to points the following probability mass,

    for x=[frm],
    (E(x) - E(x + step))/step + 1 - cdf(x)

    for x=[frm+step, frm+2*step, ........., to-step],
    (2 * E(x) - E(x - step) - E(x + step))/step

    for x=[to],
    (E(x) - E(x - step))/step - 1 + cdf(x)

    where, E(x) is the first limiting moment of the distribution
    about the point x and cdf is the cumulative density function.

    For details, refer Klugman, S. A., Panjer, H. H. and Willmot,
    G. E., Loss Models, From Data to Decisions, Fourth Edition,
    Wiley, section 9.6.5.2 (Method of local monment matching) and
    exercise 9.41.

    Examples
    --------
    >>> from pgmpy.factors import ContinuousFactor
    # exponential distribution with rate = 2
    >>> exp_pdf = lambda x: 2*np.exp(-2*x) if x>=0 else 0
    >>> exp_node = ContinuousFactor(exp_pdf)
    >>> exp_node.discretize(UnbiasedDiscretizer, frm=0, to=5, step=0.5)
    [0.36787944117140681, 0.3995764008937992, 0.14699594306754959,
     0.054076785386732107, 0.019893735665399759, 0.0073185009180336547,
     0.0026923231244619927, 0.00099045004496534084, 0.00036436735000289211,
     0.00013404200890043683, 3.2610438989610913e-05]

    """
    def get_discrete_values(self):
        lev = self._lim_moment

        # for x=[frm]
        discrete_values = [(lev(self.frm) - lev(self.frm + self.step)) / self.step
                           + 1 - self.factor.cdf(self.frm)]

        # for x=[frm+step, frm+2*step, ........., to-step]
        x = np.arange(self.frm + self.step, self.to, self.step)
        for i in x:
            discrete_values.append((2 * lev(i) - lev(i-self.step) -
                                    lev(i + self.step))/self.step)

        # for x=[to]
        discrete_values.append((lev(self.to) - lev(self.to - self.step)) /
                               self.step - 1 + self.factor.cdf(self.to))

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
        labels = super(UnbiasedDiscretizer, self).get_labels()
        labels.append("x={to}".format(to=str(self.to)))
        return labels
