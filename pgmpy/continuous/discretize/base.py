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
    cardinality: int
        the number of states required in the discretized output.

    """

    def __init__(self, factor, low, high, cardinality):
        if not isinstance(factor, ContinuousFactor):
            raise ValueError("{factor} is not a valid ContinuousFactor object."
                             .format(factor=factor))

        self.factor = factor
        self.low = low
        self.high = high
        self.cardinality = cardinality

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
        >>> node = ContinuousFactor(norm(0).pdf)
        >>> base = BaseDiscretizer(node, -5, 5, 20)
        >>> base.get_labels()
        ['x=-5.0', 'x=-4.5', 'x=-4.0', 'x=-3.5', 'x=-3.0', 'x=-2.5',
         'x=-2.0', 'x=-1.5', 'x=-1.0', 'x=-0.5', 'x=0.0', 'x=0.5',
         'x=1.0', 'x=1.5', 'x=2.0', 'x=2.5', 'x=3.0', 'x=3.5', 'x=4.0', 'x=4.5']

        """
        step = (self.high - self.low) / self.cardinality
        labels = list('x={i}'.format(i=str(i))
                      for i in np.round(np.arange(self.low, self.high, step), 3))
        return labels
