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
    frm, to: float
        the range over which the function will be discretized.
    step: float
        the discretization step (or span, or lag)

    """

    def __init__(self, factor, frm, to, step):
        if not isinstance(factor, ContinuousFactor):
            raise ValueError("{} is not a valid ContinuousFactor object."
                             .format(factor))

        self.factor = factor
        self.frm = frm
        self.to = to
        self.step = step

    def get_labels(self):
        """
        Returns a list of strings representing the values about
        which the discretization method calculates the probabilty
        masses.
        Default value is the points -
        [frm, frm+step, frm+2*step, ......... , to-step]
        unless the method is overridden by a subclass.
        """
        x = np.arange(self.frm, self.to, self.step)
        labels = list('x='+str(i) for i in x)
        return labels
