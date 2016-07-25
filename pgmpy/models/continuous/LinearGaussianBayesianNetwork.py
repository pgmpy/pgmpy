from __future__ import division

import numpy as np

from pgmpy.models import BayesianModel
from pgmpy.factors import LinearGaussianCPD


class LinearGaussianBayesianNetwork(BayesianModel):
    """
    A Linear Gaussain Bayesian Network is a Bayesian Network, all
    of whose variables are continuous, and where all of the CPDs
    are linear Gaussians.

    An important result is that the linear Gaussian Bayesian Networks
    are an alternative representation for the class of multivariate
    Gaussian distributions.
    """
    def __init__(self, ebunch=None):
        super(LinearGaussianBayesianNetwork, self).__init__()

        self._jgd = None
