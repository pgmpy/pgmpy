#!/usr/bin/env python3
"""Contains the different formats of CPDs used in PGM"""
from __future__ import division

from itertools import product
from warnings import warn
import numbers

import numpy as np

from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.extern import tabulate
from pgmpy.extern import six
from pgmpy.extern.six.moves import range, zip
from pgmpy.utils import StateNameInit
from pgmpy.utils import StateNameDecorator


class TabularCPD(DiscreteFactor):
    """
    Defines the conditional probability distribution table (cpd table)

    Examples
    --------
    For a distribution of P(grade|diff, intel)

    +-------+--------------------+------------------+
    |diff   |      easy          |    hard          |
    +-------+-----+------+-------+------+----+------+
    |intel  |dumb |  avg | smart | dumb |avg |smart |
    +-------+-----+------+-------+------+----+------+
    |gradeA |0.1  |  0.1 |  0.1  | 0.1  |0.1 | 0.1  |
    +-------+-----+------+-------+------+----+------+
    |gradeB |0.1  |  0.1 |  0.1  | 0.1  |0.1 | 0.1  |
    +-------+-----+------+-------+------+----+------+
    |gradeC |0.8  |  0.8 |  0.8  | 0.8  |0.8 | 0.8  |
    +-------+-----+------+-------+------+----+------+

    values should be
    [[0.1,0.1,0.1,0.1,0.1,0.1],
    [0.1,0.1,0.1,0.1,0.1,0.1],
    [0.8,0.8,0.8,0.8,0.8,0.8]]

    >>> cpd = TabularCPD('grade',3,[[0.1,0.1,0.1,0.1,0.1,0.1],
                                    [0.1,0.1,0.1,0.1,0.1,0.1],
                                    [0.8,0.8,0.8,0.8,0.8,0.8]],
                                    evidence=['diff', 'intel'], evidence_card=[2,3])
    >>> print(cpd)
    +---------+---------+---------+---------+---------+---------+---------+
    | diff    | diff_0  | diff_0  | diff_0  | diff_1  | diff_1  | diff_1  |
    +---------+---------+---------+---------+---------+---------+---------+
    | intel   | intel_0 | intel_1 | intel_2 | intel_0 | intel_1 | intel_2 |
    +---------+---------+---------+---------+---------+---------+---------+
    | grade_0 | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     |
    +---------+---------+---------+---------+---------+---------+---------+
    | grade_1 | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     |
    +---------+---------+---------+---------+---------+---------+---------+
    | grade_2 | 0.8     | 0.8     | 0.8     | 0.8     | 0.8     | 0.8     |
    +---------+---------+---------+---------+---------+---------+---------+
    >>> cpd.values
    array([[[ 0.1,  0.1,  0.1],
            [ 0.1,  0.1,  0.1]],

           [[ 0.1,  0.1,  0.1],
            [ 0.1,  0.1,  0.1]],

           [[ 0.8,  0.8,  0.8],
            [ 0.8,  0.8,  0.8]]])
    >>> cpd.variables
    ['grade', 'diff', 'intel']
    >>> cpd.cardinality
    array([3, 2, 3])
    >>> cpd.variable
    'grade'
    >>> cpd.variable_card
    3

    Parameters
    ----------
    variable: int, string (any hashable python object)
        The variable whose CPD is defined.

    variable_card: integer
        cardinality of variable

    values: 2d array, 2d list or 2d tuple
        values of the cpd table

    evidence: array-like
        evidences(if any) w.r.t. which cpd is defined

    evidence_card: integer, array-like
        cardinality of evidences (if any)

    Public Methods
    --------------
    get_values()
    marginalize([variables_list])
    normalize()
    reduce([values_list])
    """
    @StateNameInit()
    def __init__(self, variable, variable_card, values,
                 evidence=None, evidence_card=None):

        self.variable = variable
        self.variable_card = None

        variables = [variable]

        if not isinstance(variable_card, numbers.Integral):
            raise TypeError("Event cardinality must be an integer")
        self.variable_card = variable_card

        cardinality = [variable_card]
        if evidence_card is not None:
            if isinstance(evidence_card, numbers.Real):
                raise TypeError("Evidence card must be a list of numbers")
            cardinality.extend(evidence_card)

        if evidence is not None:
            if isinstance(evidence, six.string_types):
                raise TypeError("Evidence must be list, tuple or array of strings.")
            variables.extend(evidence)
            if not len(evidence_card) == len(evidence):
                raise ValueError("Length of evidence_card doesn't match length of evidence")

        values = np.array(values)
        if values.ndim != 2:
            raise TypeError("Values must be a 2D list/array")

        super(TabularCPD, self).__init__(variables, cardinality, values.flatten('C'),
                                         state_names=self.state_names)

    def __repr__(self):
        var_str = '<TabularCPD representing P({var}:{card}'.format(
            var=self.variable, card=self.variable_card)

        evidence = self.variables[1:]
        evidence_card = self.cardinality[1:]
        if evidence:
            evidence_str = ' | ' + ', '.join(['{var}:{card}'.format(var=var, card=card)
                                              for var, card in zip(evidence, evidence_card)])
        else:
            evidence_str = ''

        return var_str + evidence_str + ') at {address}>'.format(address=hex(id(self)))

    def get_values(self):
        """
        Returns the cpd

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd = TabularCPD('grade', 3, [[0.1, 0.1],
        ...                               [0.1, 0.1],
        ...                               [0.8, 0.8]],
        ...                  evidence='evi1', evidence_card=2)
        >>> cpd.get_values()
        array([[ 0.1,  0.1],
               [ 0.1,  0.1],
               [ 0.8,  0.8]])
        """
        if self.variable in self.variables:
            return self.values.reshape(self.cardinality[0], np.prod(self.cardinality[1:]))
        else:
            return self.values.reshape(1, np.prod(self.cardinality))

    def __str__(self):
        return self._make_table_str(tablefmt="grid")

    def _str(self, phi_or_p="p", tablefmt="fancy_grid"):
        return super(self, TabularCPD)._str(phi_or_p, tablefmt)

    def _make_table_str(self, tablefmt="fancy_grid", print_state_names=True):
        headers_list = []
        # build column headers

        evidence = self.variables[1:]
        evidence_card = self.cardinality[1:]
        if evidence:
            col_indexes = np.array(list(product(*[range(i) for i in evidence_card])))
            if self.state_names and print_state_names:
                for i in range(len(evidence_card)):
                    column_header = [str(evidence[i])] + ['{var}({state})'.format(
                        var=evidence[i],
                        state=self.state_names[evidence[i]][d])
                        for d in col_indexes.T[i]]
                    headers_list.append(column_header)
            else:
                for i in range(len(evidence_card)):
                    column_header = [str(evidence[i])] + ['{s}_{d}'.format(
                        s=evidence[i], d=d) for d in col_indexes.T[i]]
                    headers_list.append(column_header)

        # Build row headers
        if self.state_names and print_state_names:
            variable_array = [['{var}({state})'.format
                               (var=self.variable, state=self.state_names[self.variable][i])
                               for i in range(self.variable_card)]]
        else:
            variable_array = [['{s}_{d}'.format(s=self.variable, d=i) for i in range(self.variable_card)]]
        # Stack with data
        labeled_rows = np.hstack((np.array(variable_array).T, self.get_values())).tolist()
        # No support for multi-headers in tabulate
        cdf_str = tabulate(headers_list + labeled_rows, tablefmt=tablefmt)
        return cdf_str

    def copy(self):
        """
        Returns a copy of the TabularCPD object.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd = TabularCPD('grade', 2,
        ...                  [[0.7, 0.6, 0.6, 0.2],[0.3, 0.4, 0.4, 0.8]],
        ...                  ['intel', 'diff'], [2, 2])
        >>> copy = cpd.copy()
        >>> copy.variable
        'grade'
        >>> copy.variable_card
        2
        >>> copy.evidence
        ['intel', 'diff']
        >>> copy.values
        array([[[ 0.7,  0.6],
                [ 0.6,  0.2]],

               [[ 0.3,  0.4],
                [ 0.4,  0.8]]])
        """
        evidence = self.variables[1:] if len(self.variables) > 1 else None
        evidence_card = self.cardinality[1:] if len(self.variables) > 1 else None
        return TabularCPD(self.variable, self.variable_card, self.get_values(),
                          evidence, evidence_card)

    def normalize(self, inplace=True):
        """
        Normalizes the cpd table.

        Parameters
        ----------
        inplace: boolean
            If inplace=True it will modify the CPD itself, else would return
            a new CPD

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd_table = TabularCPD('grade', 2,
        ...                        [[0.7, 0.2, 0.6, 0.2],[0.4, 0.4, 0.4, 0.8]],
        ...                        ['intel', 'diff'], [2, 2])
        >>> cpd_table.normalize()
        >>> cpd_table.get_values()
        array([[ 0.63636364,  0.33333333,  0.6       ,  0.2       ],
               [ 0.36363636,  0.66666667,  0.4       ,  0.8       ]])
        """
        tabular_cpd = self if inplace else self.copy()
        cpd = tabular_cpd.get_values()
        tabular_cpd.values = (cpd / cpd.sum(axis=0)).reshape(tabular_cpd.cardinality)
        if not inplace:
            return tabular_cpd

    def marginalize(self, variables, inplace=True):
        """
        Modifies the cpd table with marginalized values.

        Parameters
        ----------
        variables: list, array-like
            list of variable to be marginalized

        inplace: boolean
            If inplace=True it will modify the CPD itself, else would return
            a new CPD

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd_table = TabularCPD('grade', 2,
        ...                        [[0.7, 0.6, 0.6, 0.2],[0.3, 0.4, 0.4, 0.8]],
        ...                        ['intel', 'diff'], [2, 2])
        >>> cpd_table.marginalize(['diff'])
        >>> cpd_table.get_values()
        array([[ 0.65,  0.4 ],
                [ 0.35,  0.6 ]])
        """
        if self.variable in variables:
            raise ValueError("Marginalization not allowed on the variable on which CPD is defined")

        tabular_cpd = self if inplace else self.copy()

        super(TabularCPD, tabular_cpd).marginalize(variables)
        tabular_cpd.normalize()

        if not inplace:
            return tabular_cpd

    @StateNameDecorator(argument='values', return_val=None)
    def reduce(self, values, inplace=True):
        """
        Reduces the cpd table to the context of given variable values.

        Parameters
        ----------
        values: list, array-like
            A list of tuples of the form (variable_name, variable_state).

        inplace: boolean
            If inplace=True it will modify the factor itself, else would return
            a new factor.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd_table = TabularCPD('grade', 2,
        ...                        [[0.7, 0.6, 0.6, 0.2],[0.3, 0.4, 0.4, 0.8]],
        ...                        ['intel', 'diff'], [2, 2])
        >>> cpd_table.reduce([('diff', 0)])
        >>> cpd_table.get_values()
        array([[ 0.7,  0.6],
               [ 0.3,  0.4]])
        """
        if self.variable in (value[0] for value in values):
            raise ValueError("Reduce not allowed on the variable on which CPD is defined")

        tabular_cpd = self if inplace else self.copy()

        super(TabularCPD, tabular_cpd).reduce(values)
        tabular_cpd.normalize()

        if not inplace:
            return tabular_cpd

    def to_factor(self):
        """
        Returns an equivalent factor with the same variables, cardinality, values as that of the cpd

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> cpd = TabularCPD('grade', 3, [[0.1, 0.1],
        ...                               [0.1, 0.1],
        ...                               [0.8, 0.8]],
        ...                  evidence='evi1', evidence_card=2)
        >>> factor = cpd.to_factor()
        >>> factor
        <DiscreteFactor representing phi(grade:3, evi1:2) at 0x7f847a4f2d68>
        """
        return DiscreteFactor(self.variables, self.cardinality, self.values)

    def reorder_parents(self, new_order, inplace=True):
        """
        Returns a new cpd table according to provided order.

        Parameters
        ----------
        new_order: list
            list of new ordering of variables

        inplace: boolean
            If inplace == True it will modify the CPD itself
            otherwise new value will be returned without affecting old values

        Examples
        --------
        Consider a CPD P(grade| diff, intel)
        >>> cpd = TabularCPD('grade',3,[[0.1,0.1,0.1,0.1,0.1,0.1],
                                        [0.1,0.1,0.1,0.1,0.1,0.1],
                                        [0.8,0.8,0.8,0.8,0.8,0.8]],
                                    evidence=['diff', 'intel'], evidence_card=[2,3])
        >>> print(cpd)
        +---------+---------+---------+---------+---------+---------+---------+
        | diff    | diff_0  | diff_0  | diff_0  | diff_1  | diff_1  | diff_1  |
        +---------+---------+---------+---------+---------+---------+---------+
        | intel   | intel_0 | intel_1 | intel_2 | intel_0 | intel_1 | intel_2 |
        +---------+---------+---------+---------+---------+---------+---------+
        | grade_0 | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     |
        +---------+---------+---------+---------+---------+---------+---------+
        | grade_1 | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     |
        +---------+---------+---------+---------+---------+---------+---------+
        | grade_2 | 0.8     | 0.8     | 0.8     | 0.8     | 0.8     | 0.8     |
        +---------+---------+---------+---------+---------+---------+---------+
        >>> cpd.values
        array([[[ 0.1,  0.1,  0.1],
                [ 0.1,  0.1,  0.1]],

               [[ 0.1,  0.1,  0.1],
                [ 0.1,  0.1,  0.1]],

               [[ 0.8,  0.8,  0.8],
                [ 0.8,  0.8,  0.8]]])
        >>> cpd.variables
        ['grade', 'diff', 'intel']
        >>> cpd.cardinality
        array([3, 2, 3])
        >>> cpd.variable
        'grade'
        >>> cpd.variable_card
        3

        >>> cpd.reorder_parents(['intel', 'diff'])
        array([[ 0.1,  0.1,  0.2,  0.2,  0.1,  0.1],
               [ 0.1,  0.1,  0.1,  0.1,  0.1,  0.1],
               [ 0.8,  0.8,  0.7,  0.7,  0.8,  0.8]])
        >>> print(cpd)
        +---------+---------+---------+---------+---------+---------+---------+
        | intel   | intel_0 | intel_0 | intel_1 | intel_1 | intel_2 | intel_2 |
        +---------+---------+---------+---------+---------+---------+---------+
        | diff    | diff_0  | diff_1  | diff_0  | diff_1  | diff_0  | diff_1  |
        +---------+---------+---------+---------+---------+---------+---------+
        | grade_0 | 0.1     | 0.1     | 0.2     | 0.2     | 0.1     | 0.1     |
        +---------+---------+---------+---------+---------+---------+---------+
        | grade_1 | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     | 0.1     |
        +---------+---------+---------+---------+---------+---------+---------+
        | grade_2 | 0.8     | 0.8     | 0.7     | 0.7     | 0.8     | 0.8     |
        +---------+---------+---------+---------+---------+---------+---------+

        >>> cpd.values
        array([[[ 0.1,  0.1],
                [ 0.2,  0.2],
                [ 0.1,  0.1]],

               [[ 0.1,  0.1],
                [ 0.1,  0.1],
                [ 0.1,  0.1]],

               [[ 0.8,  0.8],
                [ 0.7,  0.7],
                [ 0.8,  0.8]]])

        >>> cpd.variables
        ['grade', 'intel', 'diff']
        >>> cpd.cardinality
        array([3, 3, 2])
        >>> cpd.variable
        'grade'
        >>> cpd.variable_card
        3
        """
        if (len(self.variables) <= 1 or (set(new_order) - set(self.variables)) or
                (set(self.variables[1:]) - set(new_order))):
            raise ValueError("New order either has missing or extra arguments")
        else:
            if new_order != self.variables[1:]:
                evidence = self.variables[1:]
                evidence_card = self.cardinality[1:]
                card_map = dict(zip(evidence, evidence_card))
                old_pos_map = dict(zip(evidence, range(len(evidence))))
                trans_ord = [0] + [(old_pos_map[letter] + 1) for letter in new_order]
                new_values = np.transpose(self.values, trans_ord)

                if inplace:
                    variables = [self.variables[0]] + new_order
                    cardinality = [self.variable_card] + [card_map[var] for var in new_order]
                    super(TabularCPD, self).__init__(variables, cardinality, new_values.flatten('C'))
                    return self.get_values()
                else:
                    return new_values.reshape(self.cardinality[0], np.prod([card_map[var] for var in new_order]))
            else:
                warn("Same ordering provided as current")
                return self.get_values()

    def get_evidence(self):
        return self.variables[:0:-1]
