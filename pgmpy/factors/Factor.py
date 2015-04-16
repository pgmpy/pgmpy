import functools
from operator import lt
from itertools import product, starmap
from collections import OrderedDict

import numpy as np

from pgmpy.exceptions import Exceptions
from pgmpy.extern import tabulate


class Factor:
    """
    Base class for *Factor*.

    Public Methods
    --------------
    assignment(index)
    get_cardinality(variable)
    marginalize([variable_list])
    normalize()
    product(*Factor)
    reduce([variable_values_list])
    """
    def __init__(self, variables, cardinality, value):
        """
        Initialize a factors class.

        Defined above, we have the following mapping from variable
        assignments to the index of the row vector in the value field:

        +-----+-----+-----+-------------------+
        |  x1 |  x2 |  x3 |    phi(x1, x2, x2)|
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_0|     phi.value(0)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_0| x3_1|     phi.value(1)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_0|     phi.value(2)  |
        +-----+-----+-----+-------------------+
        | x1_0| x2_1| x3_1|     phi.value(3)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_0|     phi.value(4)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_0| x3_1|     phi.value(5)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_0|     phi.value(6)  |
        +-----+-----+-----+-------------------+
        | x1_1| x2_1| x3_1|     phi.value(7)  |
        +-----+-----+-----+-------------------+

        Parameters
        ----------
        variables: list
            List of scope of factor
        cardinality: list, array_like
            List of cardinality of each variable
        value: list, array_like
            List or array of values of factor.
            A Factor's values are stored in a row vector in the value
            using an ordering such that the left-most variables as defined in
            the variable field cycle through their values the fastest. More
            concretely, for factor

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 2, 2], np.ones(8))
        """
        self.variables = OrderedDict()
        if len(variables) != len(cardinality):
            raise ValueError("The size of variables and cardinality should be same")
        for variable, card in zip(variables, cardinality):
            self.variables[variable] = [variable + '_' + str(index)
                                        for index in range(card)]
        self.cardinality = np.array(cardinality)
        self.values = np.array(value, dtype=np.double)
        self.strides = {}
        if not self.values.shape[0] == np.prod(self.cardinality):
            raise Exceptions.SizeError("Incompetant value array")

    def scope(self):
        """
        Returns the scope of the factor.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], np.ones(8))
        >>> phi.scope()
        ['x1', 'x2', 'x3']
        """
        return list(self.variables)

    def assignment(self, index):
        """
        Returns a list of assignments for the corresponding index.

        Parameters
        ----------
        index: integer, list-type, ndarray
            index or indices whose assignment is to be computed

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['diff', 'intel'], [2, 2], np.ones(4))
        >>> phi.assignment([1, 2])
        [['diff_0', 'intel_1'], ['diff_1', 'intel_0']]
        """
        if not isinstance(index, np.ndarray):
            index = np.atleast_1d(index)
        max_index = np.prod(self.cardinality) - 1
        if not all(i <= max_index for i in index):
            raise IndexError("Index greater than max possible index")
        assignments = []
        for ind in index:
            assign = []
            for card in self.cardinality[::-1]:
                assign.insert(0, ind % card)
                ind = ind/card
            assignments.append(map(int, assign))
        return [[self.variables[key][val] for key, val in
                 zip(self.variables.keys(), values)] for values in assignments]

    def get_cardinality(self, variable):
        """
        Returns cardinality of a given variable

        Parameters
        ----------
        variable: string

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.get_cardinality('x1')
        2
        """
        if variable not in self.variables:
            raise Exceptions.ScopeError("%s not in scope" % variable)
        return self.cardinality[list(self.variables.keys()).index(variable)]

    def identity_factor(self):
        """
        Returns the identity factor.

        When the identity factor of a factor is multiplied with the factor
        it returns the factor itself.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi_identity = phi.identity_factor()
        >>> phi_identity.variables
        OrderedDict([('x1', ['x1_0', 'x1_1']), ('x2', ['x2_0', 'x2_1', 'x2_2']), ('x3', ['x3_0', 'x3_1'])])
        >>> phi_identity.values
        array([ 1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.,  1.])
        """
        return Factor(self.variables, self.cardinality, np.ones(np.product(self.cardinality)))

    def marginalize(self, variables, inplace=True):
        """
        Modifies the factor with marginalized values.

        Parameters
        ----------
        variables: string, list-type
            name of variable to be marginalized

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.marginalize(['x1', 'x3'])
        >>> phi.values
        array([ 14.,  22.,  30.])
        >>> phi.variables
        OrderedDict([('x2', ['x2_0', 'x2_1', 'x2_2'])])
        """
        if not isinstance(variables, list):
            variables = [variables]
        for variable in variables:
            if variable not in self.variables:
                raise Exceptions.ScopeError("{variable} not in scope".format(variable=variable))

        if inplace:
            factor = self
        else:
            factor = Factor(self.scope(), self.cardinality, self.values)
        marginalize_index = np.array(np.where(np.in1d(factor.scope(), variables)))
        assign = np.array(factor.cardinality)
        assign[marginalize_index] = -1
        marginalized_values = []
        for i in product(*[range(index) for index in assign[assign != -1]]):
            assign[assign != -1] = i
            marginalized_values.append(np.sum(factor.values[factor._index_for_assignment(assign)]))
        factor.values = np.array(marginalized_values)
        for variable in variables:
            index = list(factor.variables.keys()).index(variable)
            del(factor.variables[variable])
            factor.cardinality = np.delete(factor.cardinality, index)
        if not inplace:
            return factor
            
    def normalize(self, inplace=True):
        """
        Normalizes the values of factor so that they sum to 1.

        Parameters
        ----------
        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> norm_phi = phi.normalize()
        >>> norm_phi.values
        array([ 0.        ,  0.01515152,  0.03030303,  0.04545455,  0.06060606,
                0.07575758,  0.09090909,  0.10606061,  0.12121212,  0.13636364,
                0.15151515,  0.16666667])

        """
        if inplace:
            self.values = self.values / np.sum(self.values)
        else:
            factor = Factor(self.scope(), self.cardinality, self.values)
            factor.values = factor.values / np.sum(factor.values)
            return factor

    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of given variable values.

        Parameters
        ----------
        values: string, list-type
            name of the variable values

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.reduce(['x1_0', 'x2_0'])
        >>> phi.values
        array([0., 1.])
        """
        if not isinstance(values, list):
            values = [values]

        if inplace:
            factor = self
        else:
            factor = Factor(self.scope(), self.cardinality, self.values)
            
        if not all('_' in value for value in values):
            raise TypeError('Values should be in the form of variablename_index')
            
        value_row = list(zip(*[value.split('_') for value in values]))
        reduced_variables = list(value_row[0])
        value_indices = list(map(int, value_row[1]))
        if not set(reduced_variables).issubset(set(factor.scope())):
            raise Exceptions.ScopeError('%s not in scope' % list(set(reduced_variables)-set(factor.scope())))

        reduced_indices = np.where(np.in1d(factor.scope(), reduced_variables))
        if not all(starmap(lt, zip(value_indices, factor.cardinality[reduced_indices]))):
            raise Exceptions.SizeError('Value is greater than max possible value')

        reduce_assign = np.full(len(factor.cardinality), -1, dtype=int)
        reduce_assign[reduced_indices] = value_indices
        factor.values = factor.values[factor._index_for_assignment(reduce_assign)]
        factor.cardinality = np.delete(factor.cardinality, reduced_indices)
        for var in reduced_variables:
            del factor.variables[var]

        if not inplace:
            return factor

    def product(self, *factors, n_jobs=1):
        """
        Returns the factor product with factors.

        Parameters
        ----------
        *factors: Factor1, Factor2, ...
            Factors to be multiplied

        Example
        -------
        >>> from pgmpy.factors import Factor
        >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = Factor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
        >>> phi = phi1.product(phi2)
        >>> phi.variables
        OrderedDict([('x1', ['x1_0', 'x1_1']), ('x2', ['x2_0', 'x2_1', 'x2_2']),
                ('x3', ['x3_0', 'x3_1']), ('x4', ['x4_0', 'x4_1'])])
        """
        return factor_product(self, *factors, n_jobs=n_jobs)

    def divide(self, factor, n_jobs=1):
        """
        Returns a new factors instance after division by factor.

        Parameters
        ----------
        factor : factors
            The denominator

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi2 = Factor(['x3', 'x1'], [2, 2], [x+1 for x in range(4)])
        >>> phi = phi1.divide(phi2)
        >>> phi
        x1	x2	x3	phi(x1, x2, x3)
        x1_0	x2_0	x3_0	0.0
        x1_1	x2_0	x3_0	0.5
        x1_0	x2_1	x3_0	0.666666666667
        x1_1	x2_1	x3_0	0.75
        x1_0	x2_2	x3_0	4.0
        x1_1	x2_2	x3_0	2.5
        x1_0	x2_0	x3_1	2.0
        x1_1	x2_0	x3_1	1.75
        x1_0	x2_1	x3_1	8.0
        x1_1	x2_1	x3_1	4.5
        x1_0	x2_2	x3_1	3.33333333333
        x1_1	x2_2	x3_1	2.75
        """
        return factor_divide(self, factor, n_jobs=n_jobs)

    def maximize(self, variable, inplace=True):
        """
        Maximizes the factor with respect to the variable.

        Parameters
        ----------
        variable: int, string, any hashable python object or list
            A variable or a list of variables with respect to which factor is to be maximized

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [3, 2, 2], [0.25, 0.35, 0.08, 0.16, 0.05, 0.07,
        ...                                              0.00, 0.00, 0.15, 0.21, 0.09, 0.18])
        >>> phi.maximize('x2')
        >>> print(phi)
        x1      x3      phi(x1, x3)
        -----------------------------
        x1_0    x3_0    0.25
        x1_0    x3_1    0.35
        x1_1    x3_0    0.05
        x1_1    x3_1    0.07
        x1_2    x3_0    0.15
        x1_2    x3_1    0.21
        """
        indexes = np.where(np.in1d(self.scope(), variable))[0]
        assign = np.array(self.cardinality)
        assign[indexes] = -1
        new_values = np.array([])
        for i in product(*[range(i) for i in self.cardinality[np.where(assign != -1)[0]]]):
            assign[assign != -1] = i
            new_values = np.append(new_values, np.max(self.values[self._index_for_assignment(assign)]))
        new_variables = np.array(self.scope())[~np.in1d(self.scope(),
                                                        [variable] if isinstance(variable, str) else variable)]
        new_card = self.cardinality[assign != -1]

        if inplace:
            return self.__init__(new_variables, new_card, new_values)
        else:
            return Factor(new_variables, new_card, new_values)

    def _index_for_assignment(self, assignment):
        """
        Returns the index of values for a given assignment.
        If -1 passed for any variable, returns all the indexes ignoring variables corresponding to -1.

        Parameters
        ----------
        assignment: array-like
            An array for the states of each variable whose index is to be calculated.
            If any element is -1, that variable is ignored and all indexes for other variables
            are returned ignoring the variables corresponding to -1.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 3], np.arange(18))
        >>> phi._index_for_assignment([1, 1, 1])
        array([13])
        >>> phi._index_for_assignment([1, -1, 1])
        array([ 10,  13,  16])
        >>> phi._index_for_assignment([1, -1, -1])
        array([  9,  10,  11,  12,  13,  14,  15,  16,  17])
        """
        assignment = np.array(assignment)
        card_cumprod = np.delete(np.concatenate((np.array([1]), np.cumprod(self.cardinality[::-1])), axis=1)[::-1], 0)
        if -1 in assignment:
            indexes = np.where(assignment == -1)[0]
            cardinalities = self.cardinality[indexes]
            array_to_return = np.array([])
            for i in product(*[range(card) for card in cardinalities]):
                temp_assignment = np.array(assignment)
                temp_assignment[temp_assignment == -1] = i
                array_to_return = np.append(array_to_return, np.sum(temp_assignment * card_cumprod))
            return array_to_return.astype('int')
        else:
            return np.array([np.sum(assignment * card_cumprod)])

    def __str__(self):
        return self._str(html=False)

    def _str(self, html=False):
        string_list = []
        if html:
            html_string_header = """<table><caption>Factor</caption>"""
            string_list.append(html_string_header)

        if html:
            html_string_header = "%s%s%s" % ("""<tr>""", ''.join(
                ["""<td><b>%s</b></td>""" % var for var in self.variables]),
                """<td><b>phi(%s)</b><d></tr>""" % ', '.join(self.variables))
            string_list.append(html_string_header)
        else:
            string_header = self.scope()
            string_header.append("phi({variables})".format(variables=",".join(string_header)))

        # fun and gen are functions to generate the different values of
        # variables in the table.
        # gen starts with giving fun initial value of b=[0, 0, 0] then fun tries
        # to increment it
        # by 1.
        def fun(b, index=len(self.cardinality)-1):
            b[index] += 1
            if b[index] == self.cardinality[index]:
                b[index] = 0
                fun(b, index-1)
            return b

        def gen():
            b = [0] * len(self.variables)
            yield b
            for i in range(np.prod(self.cardinality)-1):
                yield fun(b)

        value_index = 0
        factor_table = []
        for prob in gen():
            prob_list = ["%s_%d" % (list(self.variables)[i], prob[i])
                         for i in range(len(self.variables))]
            if html:
                html_string = """<tr>%s<td>%4.4f</td></tr>""" % (
                    ''.join(["""<td>%s</td>""" % assignment
                             for assignment in prob_list]),
                    self.values[value_index])
                string_list.append(html_string)
            else:
                prob_list.append(self.values[value_index])
                factor_table.append(prob_list)
            value_index += 1

        if html:
            return "\n".join(string_list)
        else:
            return tabulate(factor_table, headers=string_header, tablefmt="fancy_grid", floatfmt=".4f")

    def __repr__(self):
        var_card = ", ".join(['{var}:{card}'.format(var=var, card=card)
                              for var, card in zip(self.variables, self.cardinality)])
        return "<Factor representing phi({var_card}) at {address}>".format(address=hex(id(self)), var_card=var_card)

    def _repr_html_(self):
        return self._str(html=True)

    def __mul__(self, other):
        return self.product(other)

    def __truediv__(self, other):
        return self.divide(other)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        if set(self.scope()) != set(other.scope()):
            return False
        else:
            order_dict = {}
            other_scope = other.scope()
            for var_index in range(len(self.scope())):
                order_dict[var_index] = other_scope.index(self.scope()[var_index])

            order_change = [order_dict[i] for i in range(len(self.cardinality))]
            if any(self.cardinality[order_change] != other.cardinality):
                return False

            indexes = np.zeros((np.prod(self.cardinality), self.cardinality.shape[0]), dtype=int)
            for k, i in enumerate(product(*[range(i) for i in self.cardinality])):
                indexes[k] = i

            transformed_assign = np.zeros(indexes.shape, dtype=np.int)
            for k, v in order_dict.items():
                transformed_assign[:, v] = indexes[:, k]

            transformed_indexes = np.zeros(np.prod(self.cardinality), dtype=int)
            for k, index in enumerate(map(other._index_for_assignment, transformed_assign)):
                transformed_indexes[k] = index

            if all(other.values[transformed_indexes] == self.values):
                return True
            else:
                return False

    def __hash__(self):
        """
        Returns the hash of the factor object based on the scope of the factor.
        """
        return hash(' '.join(self.variables) + ' '.join(map(str, self.cardinality)) +
                    ' '.join(list(map(str, self.values))))

    def stride(self, variable):
        if variable not in self.strides:
            self.strides[variable] = _stride(variable,
                                             self.variables, self.cardinality)
        return self.strides[variable]


def _stride(variable, variables, cardinalities):
    """
    Return the stride of given variable. A stride of a variable in a
    factor is its step size, that is how many configurations it
    takes from the factor to a variable change its domain value.
    A stride of a variable not in the scope of the Factor is considered
    to be zero.

    Parameters
    ----------
    variable: str
        Variable to be computed the stride.
    variables: list or list-type
        A list with variables.
    cardinalities: list or list-type
        A list with cardinalities of variables
        are the cardinality of the variables.
    """
    if variable not in variables:
        stride = 0
    else:
        stride = 1
        # We used "reversed" here to match the configuration
        # setting used in pgmpy. Ex: (0,0,0), (0,0,1), (0,1,0),
        # (1,0,0), and so on instead of (0,0,0) (1,0,0),
        # (0,1,0), (0,0,1)
        for (cardinality, v) in zip(reversed(cardinalities),
                                    reversed(variables)):
            if v == variable:
                break
            stride = stride * cardinality
    return stride


def _bivar_factor_operation(phi1, phi2, operation, n_jobs=1):
    """
    Returns product of two factors.

    Parameters
    ----------
    phi1: factors

    phi2: factors

    operation: M | D
            M: multiplies phi1 and phi2
            D: divides phi1 by phi2

    Reference
    ----------
    "Probabilistic Graphical Models: Principles and Techniques",
    Daphne Koller, Nir Friedman, 2009, Page 359,
    Algorithm 10.A.1 — Efficient implementation of a factor
    product operation.
    Note: in an old edition of the book Algorithm 10.A.1 was
    missing one line (currently line 9: assignment[l] <- 0).
    """

    # Reference: Koller Defination 10.7
    if len(set(phi1.variables).intersection(
           set(phi2.variables))) == 0 and operation == "D":
            raise ValueError("Factors Division not defined for factors with no"
                             " common scope")

    # Variables, cardinality and values of the product factor
    variables = list(phi1.variables)
    variables.extend([var for var in phi2.variables
                      if var not in phi1.variables])
    cardinality = list(phi1.cardinality)
    cardinality.extend(phi2.get_cardinality(var)
                       for var in phi2.variables
                       if var not in phi1.variables)
    quantity_values = np.prod(np.array(cardinality))
    # Constants for jumping in j and i
    j_star = {}
    k_star = {}
    for idx, variable in zip(reversed(range(len(variables))),
                             reversed(variables)):
        j_star[variable] = (cardinality[idx] - 1) * phi1.stride(variable)
        k_star[variable] = (cardinality[idx] - 1) * phi2.stride(variable)
    # Counter for the assignments within the domain cardinality of variables
    assignment = {var: 0 for var in variables}
    # Loop building the product table
    values = []
    j = 0
    k = 0
    for i in range(quantity_values):
        if operation == "M":
            values.append(phi1.values[j] * phi2.values[k])
        elif operation == "D":
            # Zero division should return zero
            if phi2.values[k] == 0:
                values.append(0)
            else:
                values.append(phi1.values[j] / phi2.values[k])
        for idx, variable in zip(reversed(range(len(variables))),
                                 reversed(variables)):
            assignment[variable] = assignment[variable] + 1
            if assignment[variable] == cardinality[idx]:
                assignment[variable] = 0
                j = j - j_star[variable]
                k = k - k_star[variable]
            else:
                j = j + phi1.stride(variable)
                k = k + phi2.stride(variable)
                break

    # Construct the product Factor
    phi = Factor(variables, cardinality, values)

    # Re-organize the structure for resultant factor
    lhs = []
    if operation == 'M':
        lhs = list(set(phi1.left_hand_side).union(set(phi2.left_hand_side)))
    elif operation == 'D':
        lhs = [v for v in set(phi1.left_hand_side).union(
               set(phi2.left_hand_side))
               if v not in phi2.left_hand_side]
    rhs = [v for v in set(phi1.right_hand_side).union(
            set(phi2.right_hand_side))
           if v not in lhs]
    phi.left_hand_side = lhs
    phi.right_hand_side = rhs

    return phi


def _parallel_helper_m(index, phi1, phi2,
                       phi1_indexes, phi2_indexes,
                       phi1_cumprod, phi2_cumprod):
    """
    Helper function for parallelizing loops in factor product operations.
    """
    index = np.array(index)
    return (phi1.values[np.sum(index[phi1_indexes] * phi1_cumprod)] *
            phi2.values[np.sum(index[phi2_indexes] * phi2_cumprod)])


def _parallel_helper_d(index, phi1, phi2,
                       phi1_indexes, phi2_indexes,
                       phi1_cumprod, phi2_cumprod):
    """
    Helper function for parallelizing loops in factor division operations.
    """
    index = np.array(index)
    try:
        return (phi1.values[np.sum(index[phi1_indexes] * phi1_cumprod)] /
                phi2.values[np.sum(index[phi2_indexes] * phi2_cumprod)])
    except FloatingPointError:
        # zero division error should return 0. ref Koller page 365, Fig 10.7
        return 0


def factor_product(*args, n_jobs=1):
    """
    Returns factor product of multiple factors.

    Parameters
    ----------
    factor1, factor2, .....: factors
        factors to be multiplied

    Examples
    --------
    >>> from pgmpy.factors import Factor, factor_product
    >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = Factor(['x3', 'x4', 'x1'], [2, 2, 2], range(8))
    >>> phi = factor_product(phi1, phi2)
    >>> phi.variables
    OrderedDict([('x1', ['x1_0', 'x1_1']), ('x2', ['x2_0', 'x2_1', 'x2_2']),
                ('x3', ['x3_0', 'x3_1']), ('x4', ['x4_0', 'x4_1'])])

    """
    if not all(isinstance(phi, Factor) for phi in args):
        raise TypeError("Input parameters must be factors")
    return functools.reduce(lambda phi1, phi2: _bivar_factor_operation(phi1, phi2, operation='M',
                                                                       n_jobs=n_jobs), args)


def factor_divide(phi1, phi2, n_jobs=1):
    """
    Returns new factors instance equal to phi1/phi2.

    Parameters
    ----------
    phi1: factors
        The Dividend.

    phi2: factors
        The Divisor.

    Examples
    --------
    >>> from pgmpy.factors import Factor, factor_divide
    >>> phi1 = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
    >>> phi2 = Factor(['x3', 'x1'], [2, 2], [x+1 for x in range(4)])
    >>> assert isinstance(phi1, Factor)
    >>> assert isinstance(phi2, Factor)
    >>> phi = factor_divide(phi1, phi2)
    >>> phi
    x1	x2	x3	phi(x1, x2, x3)
    x1_0	x2_0	x3_0	0.0
    x1_1	x2_0	x3_0	0.5
    x1_0	x2_1	x3_0	0.666666666667
    x1_1	x2_1	x3_0	0.75
    x1_0	x2_2	x3_0	4.0
    x1_1	x2_2	x3_0	2.5
    x1_0	x2_0	x3_1	2.0
    x1_1	x2_0	x3_1	1.75
    x1_0	x2_1	x3_1	8.0
    x1_1	x2_1	x3_1	4.5
    x1_0	x2_2	x3_1	3.33333333333
    x1_1	x2_2	x3_1	2.75

    """
    if not isinstance(phi1, Factor) or not isinstance(phi2, Factor):
        raise TypeError("phi1 and phi2 should be factors instances")
    return _bivar_factor_operation(phi1, phi2, operation='D', n_jobs=n_jobs)
