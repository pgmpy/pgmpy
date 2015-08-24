"""
Not complete and have no clear idea what to do with this.
"""

from itertools import product

import numpy as np

from pgmpy.factors import Factor
from pgmpy.independencies import Independencies


class JointProbabilityDistribution(Factor):
    """
    Base class for Joint Probability Distribution

    Public Methods
    --------------
    conditional_distribution(values)
    create_bayesian_model()
    get_independencies()
    pmap()
    marginal_distribution(variables)
    minimal_imap()
    """
    def __init__(self, variables, cardinality, values):
        """
        Initialize a Joint Probability Distribution class.

        Defined above, we have the following mapping from variable
        assignments to the index of the row vector in the value field:

        +-----+-----+-----+-------------------------+
        |  x1 |  x2 |  x3 |    P(x1, x2, x2)        |
        +-----+-----+-----+-------------------------+
        | x1_0| x2_0| x3_0|    P(x1_0, x2_0, x3_0)  |
        +-----+-----+-----+-------------------------+
        | x1_1| x2_0| x3_0|    P(x1_1, x2_0, x3_0)  |
        +-----+-----+-----+-------------------------+
        | x1_0| x2_1| x3_0|    P(x1_0, x2_1, x3_0)  |
        +-----+-----+-----+-------------------------+
        | x1_1| x2_1| x3_0|    P(x1_1, x2_1, x3_0)  |
        +-----+-----+-----+-------------------------+
        | x1_0| x2_0| x3_1|    P(x1_0, x2_0, x3_1)  |
        +-----+-----+-----+-------------------------+
        | x1_1| x2_0| x3_1|    P(x1_1, x2_0, x3_1)  |
        +-----+-----+-----+-------------------------+
        | x1_0| x2_1| x3_1|    P(x1_0, x2_1, x3_1)  |
        +-----+-----+-----+-------------------------+
        | x1_1| x2_1| x3_1|    P(x1_1, x2_1, x3_1)  |
        +-----+-----+-----+-------------------------+

        Parameters
        ----------
        variables: list
            List of scope of Joint Probability Distribution.
        cardinality: list, array_like
            List of cardinality of each variable
        value: list, array_like
            List or array of values of factor.
            A Joint Probability Distribution's values are stored in a row
            vector in the value using an ordering such that the left-most
            variables as defined in the variable field cycle through their
            values the fastest.

        Examples
        --------
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8)/8)
        >>> print(prob)
            print(prob)
            x1      x2      x3      P(x1, x2, x3)
            x1_0    x2_0    x3_0    0.125
            x1_0    x2_0    x3_1    0.125
            x1_0    x2_1    x3_0    0.125
            x1_0    x2_1    x3_1    0.125
            x1_1    x2_0    x3_0    0.125
            x1_1    x2_0    x3_1    0.125
            x1_1    x2_1    x3_0    0.125
            x1_1    x2_1    x3_1    0.125
        """
        if np.isclose(np.sum(values), 1):
            Factor.__init__(self, variables, cardinality, values)
        else:
            raise ValueError("The probability values doesn't sum to 1.")

    def __repr__(self):
        var_card = ", ".join(['{var}:{card}'.format(var=var, card=card)
                              for var, card in zip(self.variables, self.cardinality)])
        return "<Joint Distribution representing P({var_card}) at {address}>".format(address=hex(id(self)),
                                                                                     var_card=var_card)

    def __str__(self):
        return self._str(phi_or_p='P')

    def marginal_distribution(self, variables, inplace=True):
        """
        Returns the marginal distribution over variables.

        Parameters
        ----------
        variables: string, list, tuple, set, dict
                Variable or list of variables over which marginal distribution needs
                to be calculated

        Examples
        --------
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> values = np.random.rand(12)
        >>> prob = JointProbabilityDistribution(['x1, x2, x3'], [2, 3, 2], values/np.sum(values))
        >>> prob.marginal_distribution(['x1', 'x2'])
        >>> print(prob)
            x1      x2      P(x1, x2)
            x1_0    x2_0    0.290187723512
            x1_0    x2_1    0.203569992198
            x1_0    x2_2    0.00567786144202
            x1_1    x2_0    0.116553704043
            x1_1    x2_1    0.108469538521
            x1_1    x2_2    0.275541180284
        """
        return self.marginalize(list(set(list(self.variables)) -
                                     set(variables if isinstance(
                                         variables, (list, set, dict, tuple)) else [variables])),
                                inplace=inplace)

    def check_independence(self, event1, event2, event3=None):
        """
        Check if the Joint Probability Distribution satisfies the given independence condition.

        Parameters
        ----------
        event1: list or string
            random variable whose independence is to be checked.
        event2: list or string
            random variable from which event1 is independent.
        event3: list or string
            event1 is independent of event2 given event3.

        For random variables say X, Y, Z to check if X is independent of Y given Z.
        event1 should be either X or Y.
        event2 should be either Y or X.
        event3 should Z.

        Examples
        --------
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12)/12)
        >>> prob.check_independence('x1', 'x2')
        True
        >>> prob.check_independence(['x1'], ['x2'], 'x3')
        True
        """
        if event3:
            self.conditional_distribution(event3)
        for variable_pair in product(event1, event2):
            if (self.marginal_distribution(variable_pair, inplace=False) !=
                    self.marginal_distribution(variable_pair[0], inplace=False) *
                        self.marginal_distribution(variable_pair[1], inplace=False)):
                return False
        return True

    def get_independencies(self, condition=None):
        """
        Returns the independent variables in the joint probability distribution.
        Returns marginally independent variables if condition=None.
        Returns conditionally independent variables if condition!=None

        Parameter
        ---------
        condition: array_like
                Random Variable on which to condition the Joint Probability Distribution.

        Examples
        --------
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(8)/8)
        >>> prob.get_independencies()
        """
        if condition:
            self.conditional_distribution(condition)
        independencies = Independencies()
        from itertools import combinations
        for variable_pair in combinations(list(self.variables), 2):
            from copy import deepcopy
            if JointProbabilityDistribution.marginal_distribution(deepcopy(self), variable_pair) == \
                    JointProbabilityDistribution.marginal_distribution(deepcopy(self), variable_pair[0]) * \
                    JointProbabilityDistribution.marginal_distribution(deepcopy(self), variable_pair[1]):
                independencies.add_assertions(variable_pair)
        return independencies

    def conditional_distribution(self, values):
        """
        Returns Conditional Probability Distribution after setting values to 1.

        Parameters
        ----------
        values: string or array_like
            The values on which to condition the Joint Probability Distribution.

        Examples
        --------
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8)/8)
        >>> prob.conditional_distribution(('x1', 1))
        >>> print(prob)
            x2      x3      P(x1, x2)
            x2_0    x3_0    0.25
            x2_0    x3_1    0.25
            x2_1    x3_0    0.25
            x2_1    x3_1    0.25
        """
        self.reduce(values)
        self.normalize()

    def minimal_imap(self, order):
        """
        Returns a Bayesian Model which is minimal IMap of the Joint Probability Distribution
        considering the order of the variables.

        Parameters
        ----------
        order: array-like
            The order of the random variables.

        Examples
        --------
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12)/12)
        >>> bayesian_model = prob.minimal_imap(order=['x2', 'x1', 'x3'])
        >>> bayesian_model
        <pgmpy.models.models.models at 0x7fd7440a9320>
        """
        from pgmpy import models as bm
        import itertools

        def combinations(u):
            for r in range(len(u) + 1):
                for i in itertools.combinations(u, r):
                    yield i

        G = bm.BayesianModel()
        for variable_index in range(len(order)):
            u = order[:variable_index]
            for subset in combinations(u):
                if self.check_independence(order[variable_index], set(u)-set(subset), subset):
                    G.add_edges_from([(variable, order[variable_index]) for variable in subset])
        return G

    def pmap(self):
        pass
