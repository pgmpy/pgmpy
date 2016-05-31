import numpy as np

from pgmpy.factors import ContinuousFactor


class BaseDiscretizer(object):
    """
    Base class for the discretizer classes in pgmpy. The discretizer
    classes are used to discretize a continuous random variable
    distribution into discrete probability masses.

    Parameters
    ----------
    factor: ContinuousFactor
        the continuous factor representing the distribution
        to be discretized.
    low, high: float
        the range over which the function will be discretized.
    step: float
        the discretization step (or span, or lag)

    """

    def __init__(self, factor, low, high, step):
        if not isinstance(factor, ContinuousFactor):
            raise ValueError("{factor} is not a valid ContinuousFactor object."
                             .format(factor=factor))

        self.factor = factor
        self.low = low
        self.high = high
        self.step = step

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
        >>> from pgmpy.factors import ContinuousFactor
        >>> from pgmpy.continuous.discretize import BaseDiscretizer
        >>> from scipy.stats import norm
        >>> node = ContinuousFactor(norm(0))
        >>> base = BaseDiscretizer(node, -5, 5, 0.5)
        >>> base.get_labels()
        ['x=-5.0', 'x=-4.5', 'x=-4.0', 'x=-3.5', 'x=-3.0', 'x=-2.5',
         'x=-2.0', 'x=-1.5', 'x=-1.0', 'x=-0.5', 'x=0.0', 'x=0.5',
         'x=1.0', 'x=1.5', 'x=2.0', 'x=2.5', 'x=3.0', 'x=3.5', 'x=4.0', 'x=4.5']

        """
        labels = list('x={i}'.format(i=str(i))
                      for i in np.arange(self.low, self.high, self.step))
        return labels
