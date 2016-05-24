import numpy as np
from scipy import integrate
from scipy.stats import rv_continuous


class ContinuousFactor(rv_continuous):
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
        >>> from pgmpy.factors import ContinuousFactor
        >>> custom_pdf = lambda x : 0.5 if x > -1 and x < 1 else 0
        >>> node = ContinuousFactor(custom_pdf, -3, 3)
        """
        self.pdf = pdf
        super(ContinuousFactor, self).__init__(momtype=0, a=lb, b=ub)

    def _pdf(self, *args):
        """
        Defines the probability density function of the given
        continuous variable.
        """
        return self.pdf(*args)

    def discretize(self, method, *args, **kwargs):
        """
        Discretizes the continuous distribution into discrete
        probability masses using various methods.

        Parameters
        ----------
        method : A Discretizer Class from pgmpy.continuous.discretize

        *args, **kwargs:
            The parameters to be given to the Discretizer Class.

        Examples
        --------
        >>> from pgmpy.factors import ContinuousFactor
        >>> exp_pdf = lambda x: 2*np.exp(-2*x) if x>=0 else 0
        >>> exp_node = ContinuousFactor(exp_pdf)
        >>> from pgmpy.continuous.discretize import UnbiasedDiscretizer
        >>> exp_node.discretize(UnbiasedDiscretizer, exp_node, 0, 5, 0.5)
        [0.36787944117140681, 0.3995764008937992, 0.14699594306754959,
         0.054076785386732107, 0.019893735665399759, 0.0073185009180336547,
         0.0026923231244619927, 0.00099045004496534084, 0.00036436735000289211,
         0.00013404200890043683, 3.2610438989610913e-05]

        (Refer the various Discretization methods, in pgmpy.continuous.discretize
         for details regarding the input parameters.)

        """
        from pgmpy.continuous.discretize import BaseDiscretizer

        if not issubclass(method, BaseDiscretizer):
            raise TypeError("{} must be Discretizer class.".format(method))

        return method(self, *args, **kwargs).get_discrete_values()
