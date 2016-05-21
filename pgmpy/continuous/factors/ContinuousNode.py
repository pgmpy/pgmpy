from scipy import integrate
from scipy.stats import rv_continuous

import numpy as np


class ContinuousNode(rv_continuous):
    """
    Class for continuous node representation. This is a subclass of
    scipy.stats.rv_continuous.
    This allows representation of user defined continuous distribution
    by specifying a function to compute the probability density function
    of the distribution.
    All methods of the scipy.stats.rv_continuous class can be used on
    the objects.
    This supports an extra method to discretize the continuous distribution
    into a discrete factor using various methods.
    """
    def __init__(self, pdf, lb=None, ub=None):
        """
        Parameters
        ----------
        pdf : function
        The user defined probability density function.

        lb : float, optional
        Lower bound of the support of the distribution, default is minus
        infinity.

        ub : float, optional
        Upper bound of the support of the distribution, default is plus
        infinity.

        Examples
        --------
        >>> from pgmpy.factors import ContinuousNode
        >>> custom_pdf = lambda x : 0.5 if x > -1 and x < 1 else 0
        >>> node = ContinuousNode(custom_pdf, -3, 3)
        """
        self.pdf = pdf
        super(ContinuousNode, self).__init__(momtype=0, a=lb, b=ub)

    def _pdf(self, *args):
        """
        Defines the probability density function of the given
        continuous variable.
        """
        return self.pdf(*args)

    def _rounding_method(self, frm, to, step):
        """
        The rounding method for discretization assigns to point the
        following probability mass,

        for x=[frm],
        cdf(x+step/2)-cdf(x)

        for x=[frm+step, frm+2*step, ......... , to-step],
        cdf(x+step/2)-cdf(x-step/2)

        where, cdf is the cumulative density function of the distribution.

        Examples
        --------
        >>> from pgmpy.factors import ContinuousNode
        >>> std_normal_pdf = lambda x : np.exp(-x*x/2) / (np.sqrt(2*np.pi))
        >>> std_normal = ContinuousNode(std_normal_pdf)
        >>> std_normal.discretize(frm=-3, to=3, step=0.5,
        ...                        method_type='rounding')
        [0.001629865203424451, 0.009244709419989363, 0.027834684208773178,
         0.065590616803038182, 0.120977578710013, 0.17466632194020804,
         0.19741265136584729, 0.17466632194020937, 0.12097757871001302,
         0.065590616803036905, 0.027834684208772664, 0.0092447094199902269]
        """
        # for x=[frm]
        factor = [self.cdf(frm+step/2) - self.cdf(frm)]

        # for x=[frm+step, frm+2*step, ........., to-step]
        x = np.arange(frm+step, to, step)
        for i in x:
            factor.append(self.cdf(i+step/2) - self.cdf(i-step/2))

        return factor

    def _unbiased_method(self, frm, to, step):
        """
        The unbiased method for discretization is the matching of the
        first moment method. It invovles calculating the first order
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
        >>> from pgmpy.factors import ContinuousNode
        # exponential distribution with rate = 2
        >>> exp_pdf = lambda x: 2*np.exp(-2*x) if x>=0 else 0
        >>> exp_node = ContinuousNode(exp_pdf)
        >>> exp_node.discretize(frm=0, to=5, step=0.5, method='unbiased')
        [0.36787944117140681, 0.3995764008937992, 0.14699594306754959,
         0.054076785386732107, 0.019893735665399759, 0.0073185009180336547,
         0.0026923231244619927, 0.00099045004496534084, 0.00036436735000289211,
         0.00013404200890043683, 3.2610438989610913e-05]
        """
        lev = self._lim_moment

        # for x=[frm]
        factor = [(lev(frm) - lev(frm+step))/step + 1 - self.cdf(frm)]

        # for x=[frm+step, frm+2*step, ........., to-step]
        x = np.arange(frm+step, to, step)
        for i in x:
            factor.append((2 * lev(i) - lev(i-step) - lev(i+step))/step)

        # for x=[to]
        factor.append((lev(to) - lev(to-step))/step - 1 + self.cdf(to))

        return factor

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
        fun = lambda x: np.power(x, order)*self.pdf(x)
        return (integrate.quad(fun, -np.inf, u)[0] +
                np.power(u, order)*(1-self.cdf(u)))

    def _custom_method(self, frm, to, step, function, *args, **kwargs):
        """
        The custom method allows the users to define their own
        discretization technique.
        This can be done by defining a function. This user defined
        function must take a ContinuousNode object and a float
        (the point at which the function will be applied) as its first
        two parameters. It can have other arguments but the first two
        argument types are fixed.
        The method assigns to points,

        x=[frm, frm+step, ......, to-step],
        the probability mass - fun(x)

        where, fun is the user defined function.

        Examples
        --------
        >>> def custom_discretize(node, x, h):
        ...     return node.cdf(x+h) - node.cdf(x)

        # exponential distribution with rate = 1
        >>> exp_pdf = lambda x: np.exp(-x) if x>=0 else 0
        >>> node = ContinuousNode(exp_pdf)

        # using the custom function, custom_discretize
        # with h = 0.5
        >>> exp_node.discretize(frm=0, to=5, step=0.5,
        ...        method=custom_discretize, h=0.5)
        [0.39346934028736286, 0.23865121854119481, 0.14474928101648521,
        0.087794876918315445, 0.05325028455482439, 0.03229793031395356,
        0.019589684955426567, 0.011881744524418814, 0.0072066422895440407,
        0.0043710495993888321]
        """
        x = np.arange(frm, to, step)
        factor = [function(self, i, *args, **kwargs) for i in x]
        return factor

    def discretize(self, frm, to, step, method='rounding', *args, **kwargs):
        """
        Discretizes the continuous distribution into a discrete
        factor using various methods.

        Parameters
        ----------
        frm, to: float
            the range over which the function will be discretized.
        step: float
            the discretization step (or span, or lag)
        method: string or function
            The method to be used for discretization.
            This parameter can have the following values -
            1. rounding (string) for the rounding method
            2. unbiased (string) for the unbiased method
            3. A user defined function for the custom method
            For more information on these methods, refer the docstring
            of the specific method (_rounding_method, _unbiased_method,
            _custom_method).
        *args, **kwargs:
            These are only used with the custom method to pass the specific
            parameters of the user defined function.

        """
        if method == 'rounding':
            return self._rounding_method(frm, to, step)
        elif method == 'unbiased':
            return self._unbiased_method(frm, to, step)
        else:
            return self._custom_method(frm, to, step, method, *args, **kwargs)
