import itertools
from operator import mul

import numpy as np

from pgmpy.factors import Factor
from pgmpy.independencies import Independencies
from pgmpy.extern.six.moves import range, zip
from pgmpy.extern import six


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
    is_imap(model)
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
        >>> import numpy as np
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8)/8)
        >>> print(prob)
        x1    x2    x3      P(x1,x2,x3)
        ----  ----  ----  -------------
        x1_0  x2_0  x3_0         0.1250
        x1_0  x2_0  x3_1         0.1250
        x1_0  x2_1  x3_0         0.1250
        x1_0  x2_1  x3_1         0.1250
        x1_1  x2_0  x3_0         0.1250
        x1_1  x2_0  x3_1         0.1250
        x1_1  x2_1  x3_0         0.1250
        x1_1  x2_1  x3_1         0.1250
       """
        if np.isclose(np.sum(values), 1):
            super(JointProbabilityDistribution, self).__init__(variables, cardinality, values)
        else:
            raise ValueError("The probability values doesn't sum to 1.")

    def __repr__(self):
        var_card = ", ".join(['{var}:{card}'.format(var=var, card=card)
                              for var, card in zip(self.variables, self.cardinality)])
        return "<Joint Distribution representing P({var_card}) at {address}>".format(address=hex(id(self)),
                                                                                     var_card=var_card)

    def __str__(self):
        if six.PY2:
            return self._str(phi_or_p='P', tablefmt='pqsl')
        else:
            return self._str(phi_or_p='P')

    def marginal_distribution(self, variables, inplace=True):
        """
        Returns the marginal distribution over variables.

        Parameters
        ----------
        variables: string, list, tuple, set, dict
                Variable or list of variables over which marginal distribution needs
                to be calculated
        inplace: Boolean (default True)
                If False return a new instance of JointProbabilityDistribution

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> values = np.random.rand(12)
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 3, 2], values/np.sum(values))
        >>> prob.marginal_distribution(['x1', 'x2'])
        >>> print(prob)
        x1    x2      P(x1,x2)
        ----  ----  ----------
        x1_0  x2_0      0.1502
        x1_0  x2_1      0.1626
        x1_0  x2_2      0.1197
        x1_1  x2_0      0.2339
        x1_1  x2_1      0.1996
        x1_1  x2_2      0.1340
        """
        return self.marginalize(list(set(list(self.variables)) -
                                     set(variables if isinstance(
                                         variables, (list, set, dict, tuple)) else [variables])),
                                inplace=inplace)

    def check_independence(self, event1, event2, event3=None, condition_random_variable=False):
        """
        Check if the Joint Probability Distribution satisfies the given independence condition.

        Parameters
        ----------
        event1: list
            random variable whose independence is to be checked.
        event2: list
            random variable from which event1 is independent.
        values: 2D array or list like or 1D array or list like
            A 2D list of tuples of the form (variable_name, variable_state).
            A 1D list or array-like to condition over randome variables (condition_random_variable must be True)
            The values on which to condition the Joint Probability Distribution.
        condition_random_variable: Boolean (Default false)
            If true and event3 is not None than will check independence condition over random variable.

        For random variables say X, Y, Z to check if X is independent of Y given Z.
        event1 should be either X or Y.
        event2 should be either Y or X.
        event3 should Z.

        Examples
        --------
        >>> from pgmpy.factors import JointProbabilityDistribution as JPD
        >>> prob = JPD(['I','D','G'],[2,2,3],
                       [0.126,0.168,0.126,0.009,0.045,0.126,0.252,0.0224,0.0056,0.06,0.036,0.024])
        >>> prob.check_independence(['I'], ['D'])
        True
        >>> prob.check_independence(['I'], ['D'], [('G', 1)])  # Conditioning over G_1
        False
        >>> # Conditioning over random variable G
        >>> prob.check_independence(['I'], ['D'], ('G',), condition_random_variable=True)
        False
        """
        JPD = self.copy()
        if isinstance(event1, six.string_types):
            raise TypeError('Event 1 should be a list or array-like structure')

        if isinstance(event2, six.string_types):
            raise TypeError('Event 2 should be a list or array-like structure')

        if event3:
            if isinstance(event3, six.string_types):
                raise TypeError('Event 3 cannot of type string')

            elif condition_random_variable:
                if not all(isinstance(var, six.string_types) for var in event3):
                    raise TypeError('Event3 should be a 1d list of strings')
                event3 = list(event3)
                # Using the definition of conditional independence
                # If P(X,Y|Z) = P(X|Z)*P(Y|Z)
                # This can be expanded to P(X,Y,Z)*P(Z) == P(X,Z)*P(Y,Z)
                jpd_z = JPD.marginal_distribution(event3, inplace=False)
                for variable_pair in itertools.product(event1, event2):
                    jpd_xyz = JPD.marginal_distribution(event3 + list(variable_pair), inplace=False)
                    jpd_xz = JPD.marginal_distribution(event3 + [variable_pair[0]], inplace=False)
                    jpd_yz = JPD.marginal_distribution(event3 + [variable_pair[1]], inplace=False)
                    if jpd_xyz * jpd_z != jpd_xz * jpd_yz:
                        return False
                return True
            else:
                JPD.conditional_distribution(event3)

        for variable_pair in itertools.product(event1, event2):
            if (JPD.marginal_distribution(variable_pair, inplace=False) !=
                    JPD.marginal_distribution(variable_pair[0], inplace=False) *
                    JPD.marginal_distribution(variable_pair[1], inplace=False)):
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
        >>> import numpy as np
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12)/12)
        >>> prob.get_independencies()
        (x1 _|_ x2)
        (x1 _|_ x3)
        (x2 _|_ x3)
        """
        JPD = self.copy()
        if condition:
            JPD.conditional_distribution(condition)
        independencies = Independencies()
        for variable_pair in itertools.combinations(list(JPD.variables), 2):
            if (JPD.marginal_distribution(variable_pair, inplace=False) ==
                    JPD.marginal_distribution(variable_pair[0], inplace=False) *
                    JPD.marginal_distribution(variable_pair[1], inplace=False)):
                independencies.add_assertions(variable_pair)
        return independencies

    def conditional_distribution(self, values, inplace=True):
        """
        Returns Conditional Probability Distribution after setting values to 1.

        Parameters
        ----------
        values: list or array_like
            A list of tuples of the form (variable_name, variable_state).
            The values on which to condition the Joint Probability Distribution.
        inplace: Boolean (default True)
            If False returns a new instance of JointProbabilityDistribution

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8)/8)
        >>> prob.conditional_distribution([('x1', 1)])
        >>> print(prob)
        x2    x3      P(x2,x3)
        ----  ----  ----------
        x2_0  x3_0      0.2500
        x2_0  x3_1      0.2500
        x2_1  x3_0      0.2500
        x2_1  x3_1      0.2500
        """
        JPD = self if inplace else self.copy()
        JPD.reduce(values)
        JPD.normalize()
        if not inplace:
            return JPD

    def copy(self):
        """
        Returns A copy of JointProbabilityDistribution object
        
        Examples
        ---------
        >>> import numpy as np
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12)/12)
        >>> prob_copy = prob.copy()
        >>> prob_copy.values == prob.values
        True
        >>> prob_copy.variables == prob.variables
        True
        >>> prob_copy.variables[1] = 'y'
        >>> prob_copy.variables == prob.variables
        False
        """
        return JointProbabilityDistribution(self.scope(), self.cardinality, self.values)

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
        >>> import numpy as np
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> prob = JointProbabilityDistribution(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(12)/12)
        >>> bayesian_model = prob.minimal_imap(order=['x2', 'x1', 'x3'])
        >>> bayesian_model
        <pgmpy.models.models.models at 0x7fd7440a9320>
        >>> bayesian_model.edges()
        [('x1', 'x3'), ('x2', 'x3')]
        """
        from pgmpy.models import BayesianModel

        def get_subsets(u):
            for r in range(len(u) + 1):
                for i in itertools.combinations(u, r):
                    yield i

        G = BayesianModel()
        for variable_index in range(len(order)):
            u = order[:variable_index]
            for subset in get_subsets(u):
                if (len(subset) < len(u) and
                        self.check_independence([order[variable_index]], set(u) - set(subset), subset, True)):
                    G.add_edges_from([(variable, order[variable_index]) for variable in subset])
        return G

    def is_imap(self, model):
        """
        Checks whether the given BayesianModel is Imap of JointProbabilityDistribution

        Parameters
        -----------
        model : An instance of BayesianModel Class, for which you want to
            check the Imap

        Returns
        --------
        boolean : True if given bayesian model is Imap for Joint Probability Distribution
                False otherwise
        Examples
        --------
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.factors import TabularCPD
        >>> from pgmpy.factors import JointProbabilityDistribution
        >>> bm = BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> diff_cpd = TabularCPD('diff', 2, [[0.2], [0.8]])
        >>> intel_cpd = TabularCPD('intel', 3, [[0.5], [0.3], [0.2]])
        >>> grade_cpd = TabularCPD('grade', 3,
        ...                        [[0.1,0.1,0.1,0.1,0.1,0.1],
        ...                         [0.1,0.1,0.1,0.1,0.1,0.1],
        ...                         [0.8,0.8,0.8,0.8,0.8,0.8]],
        ...                        evidence=['diff', 'intel'],
        ...                        evidence_card=[2, 3])
        >>> bm.add_cpds(diff_cpd, intel_cpd, grade_cpd)
        >>> val = [0.01, 0.01, 0.08, 0.006, 0.006, 0.048, 0.004, 0.004, 0.032,
                   0.04, 0.04, 0.32, 0.024, 0.024, 0.192, 0.016, 0.016, 0.128]
        >>> JPD = JointProbabilityDistribution(['diff', 'intel', 'grade'], [2, 3, 3], val)
        >>> JPD.is_imap(bm)
        True
        """
        from pgmpy.models import BayesianModel
        if not isinstance(model, BayesianModel):
            raise TypeError("model must be an instance of BayesianModel")
        factors = [cpd.to_factor() for cpd in model.get_cpds()]
        factor_prod = six.moves.reduce(mul, factors)
        JPD_fact = Factor(self.variables, self.cardinality, self.values)
        if JPD_fact == factor_prod:
            return True
        else:
            return False

    def pmap(self):
        pass
