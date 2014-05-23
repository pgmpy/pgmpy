from pgmpy.Factor import Factor
from pgmpy.Independencies import Independencies
import numpy as np


class JointProbabilityDistribution(Factor):
    """
    Base class for Joint Probability Distribution

    Public Methods
    --------------
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
        >>> from pgmpy.Factor import JointProbabilityDistribution
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

    def __str__(self):
        return self._str('P')

    def marginal_distribution(self, variables):
        """
        Returns the marginal distribution over variables.

        Parameters
        ----------
        variables: string, list, tuple, set, dict
                Variable or list of variables over which marginal distribution needs
                to be calculated

        Examples
        --------
        >>> from pgmpy.Factor import JointProbabilityDistribution
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
        self.marginalize(list(set(list(self.variables)) - set(variables if isinstance(variables, (list, set, dict, tuple))
                                                              else [variables])))

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
        >>> from pgmpy.Factor import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(8)/8)
        >>> prob.get_independencies()
        """
        if condition:
            # condition the Joint Probability Distribution
            pass
        independencies = Independencies()
        from itertools import combinations
        for variable_pair in combinations(list(self.variables), 2):
            if self.marginal_distribution(variable_pair) == self.marginal_distribution(variable_pair[0]) * self.marginal_distribution(variable_pair[1]):
                independencies.add_assertions(variable_pair)
        return independencies

    def minimal_imap(self, order):
        pass

    def pmap(self):
        pass