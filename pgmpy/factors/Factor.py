from collections import OrderedDict, namedtuple
from copy import deepcopy

import numpy as np

from pgmpy.exceptions import Exceptions
from pgmpy.extern import tabulate
from pgmpy.utils.mathext import cartesian


State = namedtuple('State', ['var', 'state'])


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
    def __init__(self, variables, cardinality, values):
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
        values: list, array_like
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
        values = np.array(values)

        if values.size != np.product(cardinality):
            raise ValueError("Values array must be of size: {size}".format(size=np.product(cardinality)))

        self.variables = list(variables)
        self.cardinality = cardinality
        self.values = values.reshape(cardinality)

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
        return self.variables

    #TODO: Fix this method
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
        [[('diff', 0), ('intel', 1)], [('diff', 1), ('intel', 0)]]
        """
        if isinstance(index, (int, np.integer)):
            index = [index]
        index = np.array(index)

        max_index = np.prod(self.cardinality) - 1
        if not all(i <= max_index for i in index):
            raise IndexError("Index greater than max possible index")

        assignments = np.zeros((len(index), len(self.scope())), dtype=np.int)
        rev_card = self.cardinality[::-1]
        for i, card in enumerate(rev_card):
            assignments[:, i] = index % card
            index = index//card

        assignments = assignments[:, ::-1]

        return [[self.variables[key][val] for key, val in zip(self.variables.keys(), values)] for values in assignments]

    def get_cardinality(self, variables):
        """
        Returns cardinality of a given variable

        Parameters
        ----------
        variables: list, array-like
                A list of variable names.

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.get_cardinality('x1')
        {'x1': 2}
        >>> phi.get_cardinality(['x1', 'x2'])
        {'x1': 2, 'x2': 3}
        """
        if not hasattr(variables, '__iter__'):
            variables = np.array([variables])

        if not all([var in self.variables for var in variables]):
            raise ValueError("Variable not in scope")

        return {var: self.cardinality[self.variables.index(var)] for var in variables}

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
        return Factor(self.variables, self.cardinality, np.ones(self.values.size))

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
        if not hasattr(variables, '__iter__'):
            variables = [variables]

        phi = self if inplace else deepcopy(self)

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [] 
        for var in variables:
            var_index = phi.variables.index(var)
            var_indexes.append(var_index)

            del phi.variables[var_index]
            del phi.cardinality[var_index]

        phi.values = np.sum(phi.values, axis=tuple(var_indexes))

        if not inplace:
            return phi

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
        >>> phi.normalize()
        >>> phi.values
        array([ 0.        ,  0.01515152,  0.03030303,  0.04545455,  0.06060606,
                0.07575758,  0.09090909,  0.10606061,  0.12121212,  0.13636364,
                0.15151515,  0.16666667])

        """
        phi = self if inplace else deepcopy(self)
        phi.values = phi.values / phi.values.sum()

        if not inplace:
            return phi

    def reduce(self, values, inplace=True):
        """
        Reduces the factor to the context of given variable values.

        Parameters
        ----------
        values: list-type
            A single tuple of the form (variable_name, variable_state) or a list of tuples of the same form.

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor

        Examples
        --------
        >>> from pgmpy.factors import Factor
        >>> phi = Factor(['x1', 'x2', 'x3'], [2, 3, 2], range(12))
        >>> phi.reduce([('x1', 0), ('x2', 0)])
        >>> phi.values
        array([0., 1.])
        """
        if not hasattr(values, '__iter__'):
            values = list(values)

        phi = self if inplace else deepcopy(self)

        slice_ = [slice(None)] * len(self.variables)
        for var, state in values:
            var_index = phi.variables.index(var)
            slice_[var_index] = state

            del phi.variables[var_index]
            del phi.cardinality[var_index]

        phi.values = phi.values[tuple(slice_)]

        if not inplace:
            return phi

    def product(self, phi1, inplace=True):
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
        phi = self if inplace else deepcopy(self)

        # modifying phi to add new variables
        extra_vars = set(phi1.variables) - set(phi.variables)
        if extra_vars:
            slice_ = [slice(None)] * len(phi.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi.values = phi.values[slice_]

            phi.variables.extend(extra_vars)

            new_var_card = phi1.get_cardinality(extra_vars)
            phi.cardinality = np.append(phi.cardinality, [new_var_card[var] for var in extra_vars])

        # modifying phi1 to add new variables
        extra_vars = set(phi.variables) - set(phi1.variables)
        if extra_vars:
            slice_ = [slice(None)] * len(phi1.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi1.values = phi1.values[slice_]

            phi1.variables.extend(extra_vars)

        # rearranging the axes of phi1 to match phi
        for axis in range(phi.values.ndim):
            exchange_index = phi1.variables.index(phi.variables[axis])
            phi1.variables[axis], phi1.variables[exchange_index] = phi1.variables[exchange_index], phi1.variables[axis]
            phi1.values = phi1.values.swapaxes(axis, exchange_index)

        phi.values = phi.values * phi1.values

        if not inplace:
            return phi

    def divide(self, phi1, inplace=True):
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
        phi = self if inplace else deepcopy(self)

        if set(phi1) - set(phi):
            raise ValueError("Scope of divisor should be a subset of dividend")

        extra_vars = set(phi) - set(phi1)
        if extra_vars:
            slice_ = [slice(None)] * len(phi1.variables)
            slice_.extend([np.newaxis] * len(extra_vars))
            phi1.values = phi1.values[slice_]

            phi1.variables.extend(extra_vars)

        for axis in range(phi.values.ndim):
            exchange_index = phi1.variables.index(phi.variables[axis])
            phi1.variables[axis], phi1.variables[exchange_index] = phi1.variables[exchange_index], phi1.variables[axis]
            phi1.values = phi1.values.swapaxes(axis, exchange_index)

        phi.values = phi.values / phi1.values

        if not inplace:
            return phi

    def maximize(self, variables, inplace=True):
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
        if not hasattr(variables, '__iter__'):
            variables = [variables]

        phi = self if inplace else deepcopy(self)

        for var in variables:
            if var not in phi.variables:
                raise ValueError("{var} not in scope.".format(var=var))

        var_indexes = [] 
        for var in variables:
            var_index = phi.variables.index(var)
            var_indexes.append(var_index)

            del phi.variables[var_index]
            del phi.cardinality[var_index]

        phi.values = np.max(phi.values, axis=tuple(var_indexes))

        if not inplace:
            return phi

    def copy(self):
        return Factor(self.scope(), self.cardinality, self.values)

    def __str__(self):
        return self._str(phi_or_p='phi', html=False)

    def _str(self, phi_or_p, html=False):
        string_list = []
        if html:
            html_string_header = '<table><caption>Factor</caption>'
            string_list.append(html_string_header)

        if html:
            html_string_header = '{tr}{variable_cols}{phi}'.format(
                tr='<tr',
                variable_cols=''.join(['<td><b>{var}</b></td>'.format(var=str(var)) for var in self.variables]),
                phi='<td><b>{phi_or_p}{vars}</b><d></tr>'.format(phi_or_P=phi_or_p,
                                                                 vars=', '.join([str(var) for var in self.variables])))
            string_list.append(html_string_header)
        else:
            string_header = self.scope()
            string_header.append('{phi_or_p}({variables})'.format(phi_or_p=phi_or_p,
                                                                  variables=','.join(string_header)))

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

    def __mul__(self, other):
        return self.product(other)

    def __truediv__(self, other):
        return self.divide(other)

    def __eq__(self, other):
        if type(self) != type(other):
            return False

        elif set(self.scope()) != set(other.scope()):
            return False
        else:
            for axis in range(self.values.ndim):
                exchange_index = other.variables.index(self.variables[axis])
                other.variables[axis], other.variables[exchange_index] = (other.variables[exchange_index],
                                                                          other.variables[axis])
                other.cardinality[axis], other.cardinality[exchange_index] = (other.cardinality[exchange_index],
                                                                              other.variables[axis])
                other.values = other.values.swapaxes(axis, exchange_index)

            if not np.allclose(other.values, self.values):
                return False
            elif not self.cardinality == other.cardinality:
                return False
            else:
                return True

    def __hash__(self):
        """
        Returns the hash of the factor object based on the scope of the factor.
        """
        return hash(' '.join(self.variables) + ' '.join(map(str, self.cardinality)) +
                    ' '.join(list(map(str, self.values))))


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
    """
    # try:
    #     from joblib import Parallel, delayed
    #     use_joblib = True
    # except ImportError:
    #     use_joblib = False
    #
    # def err_handler(type, flag):
    #     raise Exceptions.InvalidValueError(type)
    #
    # np.seterrcall(err_handler)
    # np.seterr(divide='raise', over='raise', under='raise', invalid='call')
    #
    # phi1_vars = list(phi1.variables)
    # phi2_vars = list(phi2.variables)
    # common_var_list = [var for var in phi1_vars if var in phi2_vars]
    # if common_var_list:
    #     variables = phi1_vars
    #     variables.extend([var for var in phi2.variables
    #                      if var not in common_var_list])
    #     cardinality = list(phi1.cardinality)
    #     cardinality.extend(phi2.get_cardinality(var) for var in phi2.variables
    #                        if var not in common_var_list)
    #
    #     phi1_indexes = [i for i in range(len(phi1.variables))]
    #     phi2_indexes = [variables.index(var) for var in phi2.variables]
    #     values = []
    #     phi1_cumprod = np.delete(np.concatenate(
    #         (np.array([1]), np.cumprod(phi1.cardinality[::-1])), axis=1)[::-1], 0)
    #     phi2_cumprod = np.delete(np.concatenate(
    #         (np.array([1]), np.cumprod(phi2.cardinality[::-1])), axis=1)[::-1], 0)
    #
    #     if operation == 'M':
    #         if use_joblib and n_jobs != 1:
    #             values = Parallel(n_jobs=n_jobs, backend='threading')(
    #                 delayed(_parallel_helper_m)(index, phi1, phi2,
    #                                             phi1_indexes, phi2_indexes,
    #                                             phi1_cumprod, phi2_cumprod)
    #                 for index in product(*[range(card) for card in cardinality]))
    #         else:
    #             # TODO: @ankurankan Make this cleaner
    #             indexes = np.array(list(map(list, product(*[range(card) for card in cardinality]))))
    #             values = (phi1.values[np.sum(indexes[:, phi1_indexes] * phi1_cumprod, axis=1).ravel()] *
    #                       phi2.values[np.sum(indexes[:, phi2_indexes] * phi2_cumprod, axis=1).ravel()])
    #
    #     elif operation == 'D':
    #         if use_joblib and n_jobs != 1:
    #             values = Parallel(n_jobs, backend='threading')(
    #                 delayed(_parallel_helper_d)(index, phi1, phi2,
    #                                             phi1_indexes, phi2_indexes,
    #                                             phi1_cumprod, phi2_cumprod)
    #                 for index in product(*[range(card) for card in cardinality]))
    #         else:
    #             # TODO: @ankurankan Make this cleaner and handle case of division by zero
    #             for index in product(*[range(card) for card in cardinality]):
    #                 index = np.array(index)
    #                 try:
    #                     values.append(phi1.values[np.sum(index[phi1_indexes] * phi1_cumprod)] /
    #                                   phi2.values[np.sum(index[phi2_indexes] * phi2_cumprod)])
    #                 except Exceptions.InvalidValueError:
    #                     # zero division error should return 0 if both operands
    #                     # equal to 0. Ref Koller page 365, Fig 10.7
    #                     values.append(0)
    #
    #     phi = Factor(variables, cardinality, values)
    #     return phi
    # else:
    #     values = np.zeros(phi1.values.shape[0] * phi2.values.shape[0])
    #     phi2_shape = phi2.values.shape[0]
    #     if operation == 'M':
    #         for value_index in range(phi1.values.shape[0]):
    #             values[value_index * phi2_shape: (value_index + 1) * phi2_shape] = (phi1.values[value_index] *
    #                                                                                 phi2.values)
    #     elif operation == 'D':
    #         # reference: Koller Defination 10.7
    #         raise ValueError("Factors Division not defined for factors with no"
    #                          " common scope")
    #     variables = phi1_vars + phi2_vars
    #     cardinality = list(phi1.cardinality) + list(phi2.cardinality)
    #     phi = Factor(variables, cardinality, values)
    #     return phi

#
# def _parallel_helper_m(index, phi1, phi2,
#                        phi1_indexes, phi2_indexes,
#                        phi1_cumprod, phi2_cumprod):
#     """
#     Helper function for parallelizing loops in factor product operations.
#     """
#     index = np.array(index)
#     return (phi1.values[np.sum(index[phi1_indexes] * phi1_cumprod)] *
#             phi2.values[np.sum(index[phi2_indexes] * phi2_cumprod)])
#
#
# def _parallel_helper_d(index, phi1, phi2,
#                        phi1_indexes, phi2_indexes,
#                        phi1_cumprod, phi2_cumprod):
#     """
#     Helper function for parallelizing loops in factor division operations.
#     """
#     index = np.array(index)
#     try:
#         return (phi1.values[np.sum(index[phi1_indexes] * phi1_cumprod)] /
#                 phi2.values[np.sum(index[phi2_indexes] * phi2_cumprod)])
#     except FloatingPointError:
#         # zero division error should return 0. ref Koller page 365, Fig 10.7
#         return 0


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
