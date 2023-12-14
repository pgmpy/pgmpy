#!/usr/bin/env python3
"""Contains the different formats of CPDs used in PGM"""
import csv
import numbers
from itertools import chain, product
from shutil import get_terminal_size
from warnings import warn

import numpy as np
import torch

from pgmpy import config
from pgmpy.extern import tabulate
from pgmpy.factors.discrete import DiscreteFactor
from pgmpy.utils import compat_fns


class TabularCPD(DiscreteFactor):
    """
    Defines the conditional probability distribution table (CPD table)

    Parameters
    ----------
    variable: int, string (any hashable python object)
        The variable whose CPD is defined.

    variable_card: integer
        Cardinality/no. of states of `variable`

    values: 2D array, 2D list or 2D tuple
        Values for the CPD table. Please refer the example for the
        exact format needed.

    evidence: array-like
        List of variables in evidences(if any) w.r.t. which CPD is defined.

    evidence_card: array-like
        cardinality/no. of states of variables in `evidence`(if any)

    Examples
    --------
    For a distribution of P(grade|diff, intel)

    +---------+-------------------------+------------------------+
    |diff:    |          easy           |         hard           |
    +---------+------+--------+---------+------+--------+--------+
    |aptitude:| low  | medium |  high   | low  | medium |  high  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeA   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeB   | 0.1  | 0.1    |   0.1   |  0.1 |  0.1   |   0.1  |
    +---------+------+--------+---------+------+--------+--------+
    |gradeC   | 0.8  | 0.8    |   0.8   |  0.8 |  0.8   |   0.8  |
    +---------+------+--------+---------+------+--------+--------+

    values should be
    [[0.1,0.1,0.1,0.1,0.1,0.1],
    [0.1,0.1,0.1,0.1,0.1,0.1],
    [0.8,0.8,0.8,0.8,0.8,0.8]]

    >>> cpd = TabularCPD('grade',3,[[0.1,0.1,0.1,0.1,0.1,0.1],
    ...                             [0.1,0.1,0.1,0.1,0.1,0.1],
    ...                             [0.8,0.8,0.8,0.8,0.8,0.8]],
    ...                             evidence=['diff', 'intel'], evidence_card=[2,3])
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
    """

    def __init__(
        self,
        variable,
        variable_card,
        values,
        evidence=None,
        evidence_card=None,
        state_names={},
    ):
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
            if isinstance(evidence, str):
                raise TypeError("Evidence must be list, tuple or array of strings.")
            variables.extend(evidence)
            if not len(evidence_card) == len(evidence):
                raise ValueError(
                    "Length of evidence_card doesn't match length of evidence"
                )

        if config.BACKEND == "numpy":
            values = np.array(values, dtype=config.get_dtype())
        else:
            values = (
                torch.Tensor(values).type(config.get_dtype()).to(config.get_device())
            )

        if values.ndim != 2:
            raise TypeError("Values must be a 2D list/array")

        if evidence is None:
            expected_cpd_shape = (variable_card, 1)
        else:
            expected_cpd_shape = (variable_card, np.prod(evidence_card))
        if values.shape != expected_cpd_shape:
            raise ValueError(
                f"values must be of shape {expected_cpd_shape}. Got shape: {values.shape}"
            )

        if not isinstance(state_names, dict):
            raise ValueError(
                f"state_names must be of type dict. Got {type(state_names)}"
            )

        super(TabularCPD, self).__init__(
            variables, cardinality, values.flatten(), state_names=state_names
        )

    def __repr__(self):
        var_str = f"<TabularCPD representing P({self.variable}:{self.variable_card}"

        evidence = self.variables[1:]
        evidence_card = self.cardinality[1:]
        if evidence:
            evidence_str = " | " + ", ".join(
                [f"{var}:{card}" for var, card in zip(evidence, evidence_card)]
            )
        else:
            evidence_str = ""

        return var_str + evidence_str + f") at {hex(id(self))}>"

    def get_values(self):
        """
        Returns the values of the CPD as a 2-D array. The order of the
        parents is the same as provided in evidence.

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
            return self.values.reshape(
                tuple([self.cardinality[0], np.prod(self.cardinality[1:])])
            )
        else:
            return self.values.reshape(tuple([np.prod(self.cardinality), 1]))

    def __str__(self):
        return self._make_table_str(tablefmt="grid")

    def _str(self, phi_or_p="p", tablefmt="fancy_grid"):
        return super(self, TabularCPD)._str(phi_or_p, tablefmt)

    def _make_table_str(
        self, tablefmt="fancy_grid", print_state_names=True, return_list=False
    ):
        headers_list = []

        # Build column headers
        evidence = self.variables[1:]
        evidence_card = self.cardinality[1:]
        if evidence:
            col_indexes = np.array(list(product(*[range(i) for i in evidence_card])))
            if self.state_names and print_state_names:
                for i in range(len(evidence_card)):
                    column_header = [str(evidence[i])] + [
                        "{var}({state})".format(
                            var=evidence[i], state=self.state_names[evidence[i]][d]
                        )
                        for d in col_indexes.T[i]
                    ]
                    headers_list.append(column_header)
            else:
                for i in range(len(evidence_card)):
                    column_header = [str(evidence[i])] + [
                        f"{evidence[i]}_{d}" for d in col_indexes.T[i]
                    ]
                    headers_list.append(column_header)

        # Build row headers
        if self.state_names and print_state_names:
            variable_array = [
                [
                    "{var}({state})".format(
                        var=self.variable, state=self.state_names[self.variable][i]
                    )
                    for i in range(self.variable_card)
                ]
            ]
        else:
            variable_array = [
                [f"{self.variable}_{i}" for i in range(self.variable_card)]
            ]
        # Stack with data
        labeled_rows = np.hstack(
            (np.array(variable_array).T, compat_fns.to_numpy(self.get_values()))
        ).tolist()

        if return_list:
            return headers_list + labeled_rows

        # No support for multi-headers in tabulate
        cdf_str = tabulate(headers_list + labeled_rows, tablefmt=tablefmt)

        cdf_str = self._truncate_strtable(cdf_str)

        return cdf_str

    def _truncate_strtable(self, cdf_str):
        terminal_width, terminal_height = get_terminal_size()

        list_rows_str = cdf_str.split("\n")

        table_width, table_height = len(list_rows_str[0]), len(list_rows_str)

        colstr_i = np.array(
            [pos for pos, char in enumerate(list_rows_str[0]) if char == "+"]
        )

        if table_width > terminal_width:
            half_width = terminal_width // 2 - 3

            left_i = colstr_i[colstr_i < half_width][-1]
            right_i = colstr_i[(table_width - colstr_i) < half_width][0]

            new_cdf_str = []
            for temp_row_str in list_rows_str:
                left = temp_row_str[: left_i + 1]
                right = temp_row_str[right_i:]
                if temp_row_str[left_i] == "+":
                    joiner = "-----"
                else:
                    joiner = " ... "
                new_cdf_str.append(left + joiner + right)

            cdf_str = "\n".join(new_cdf_str)

        # TODO: vertical limiter
        # if table_height > terminal_height:
        #     half_height = terminal_height // 2

        return cdf_str

    def to_csv(self, filename):
        """
        Exports the CPD to a CSV file.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> model = get_example_model("alarm")
        >>> cpd = model.get_cpds("SAO2")
        >>> cpd.to_csv(filename="sao2.cs")
        """
        with open(filename, "w") as f:
            writer = csv.writer(f)
            writer.writerows(self._make_table_str(tablefmt="grid", return_list=True))

    def copy(self):
        """
        Returns a copy of the `TabularCPD` object.

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
        return TabularCPD(
            self.variable,
            self.variable_card,
            compat_fns.copy(self.get_values()),
            evidence,
            evidence_card,
            state_names=self.state_names.copy(),
        )

    def normalize(self, inplace=True):
        """
        Normalizes the cpd table. The method modifies each column of values such
        that it sums to 1 without changing the proportion between states.

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
        tabular_cpd.values = (cpd / cpd.sum(axis=0)).reshape(
            tuple(tabular_cpd.cardinality)
        )
        if not inplace:
            return tabular_cpd

    def marginalize(self, variables, inplace=True):
        """
        Modifies the CPD table with marginalized values. Marginalization refers to
        summing out variables, hence that variable would no longer appear in the
        CPD.

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
            raise ValueError(
                "Marginalization not allowed on the variable on which CPD is defined"
            )

        tabular_cpd = self if inplace else self.copy()

        super(TabularCPD, tabular_cpd).marginalize(variables)
        tabular_cpd.normalize()

        if not inplace:
            return tabular_cpd

    def reduce(self, values, inplace=True, show_warnings=True):
        """
        Reduces the cpd table to the context of given variable values. Reduce fixes the
        state of given variable to specified value. The reduced variables will no longer
        appear in the CPD.

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
            raise ValueError(
                "Reduce not allowed on the variable on which CPD is defined"
            )

        tabular_cpd = self if inplace else self.copy()

        super(TabularCPD, tabular_cpd).reduce(values, show_warnings=show_warnings)
        tabular_cpd.normalize()

        if not inplace:
            return tabular_cpd

    def to_factor(self):
        """
        Returns an equivalent factor with the same variables, cardinality, values as that of the CPD.
        Since factor doesn't distinguish between conditional and non-conditional distributions,
        evidence information will be lost.

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
        factor = DiscreteFactor.__new__(DiscreteFactor)
        factor.variables = self.variables.copy()
        factor.cardinality = self.cardinality.copy()
        factor.values = compat_fns.copy(self.values)
        factor.state_names = self.state_names.copy()
        factor.name_to_no = self.name_to_no.copy()
        factor.no_to_name = self.no_to_name.copy()
        return factor

    def reorder_parents(self, new_order, inplace=True):
        """
        Returns a new cpd table according to provided parent/evidence order.

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

        >>> cpd = TabularCPD('grade',3,[[0.1,0.1,0.0,0.4,0.2,0.1],
        ...                             [0.3,0.2,0.1,0.4,0.3,0.2],
        ...                             [0.6,0.7,0.9,0.2,0.5,0.7]],
        ...                  evidence=['diff', 'intel'], evidence_card=[2,3])
        >>> print(cpd)
        +----------+----------+----------+----------+----------+----------+----------+
        | diff     | diff(0)  | diff(0)  | diff(0)  | diff(1)  | diff(1)  | diff(1)  |
        +----------+----------+----------+----------+----------+----------+----------+
        | intel    | intel(0) | intel(1) | intel(2) | intel(0) | intel(1) | intel(2) |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(0) | 0.1      | 0.1      | 0.0      | 0.4      | 0.2      | 0.1      |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(1) | 0.3      | 0.2      | 0.1      | 0.4      | 0.3      | 0.2      |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(2) | 0.6      | 0.7      | 0.9      | 0.2      | 0.5      | 0.7      |
        +----------+----------+----------+----------+----------+----------+----------+
        >>> cpd.values
        array([[[ 0.1,  0.1,  0. ],
                [ 0.4,  0.2,  0.1]],
               [[ 0.3,  0.2,  0.1],
                [ 0.4,  0.3,  0.2]],
               [[ 0.6,  0.7,  0.9],
                [ 0.2,  0.5,  0.7]]])
        >>> cpd.variables
        ['grade', 'diff', 'intel']
        >>> cpd.cardinality
        array([3, 2, 3])
        >>> cpd.variable
        'grade'
        >>> cpd.variable_card
        3
        >>> cpd.reorder_parents(['intel', 'diff'])
        array([[0.1, 0.4, 0.1, 0.2, 0. , 0.1],
               [0.3, 0.4, 0.2, 0.3, 0.1, 0.2],
               [0.6, 0.2, 0.7, 0.5, 0.9, 0.7]])
        >>> print(cpd)
        +----------+----------+----------+----------+----------+----------+----------+
        | intel    | intel(0) | intel(0) | intel(1) | intel(1) | intel(2) | intel(2) |
        +----------+----------+----------+----------+----------+----------+----------+
        | diff     | diff(0)  | diff(1)  | diff(0)  | diff(1)  | diff(0)  | diff(1)  |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(0) | 0.1      | 0.4      | 0.1      | 0.2      | 0.0      | 0.1      |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(1) | 0.3      | 0.4      | 0.2      | 0.3      | 0.1      | 0.2      |
        +----------+----------+----------+----------+----------+----------+----------+
        | grade(2) | 0.6      | 0.2      | 0.7      | 0.5      | 0.9      | 0.7      |
        +----------+----------+----------+----------+----------+----------+----------+
        >>> cpd.values
        array([[[0.1, 0.4],
                [0.1, 0.2],
                [0. , 0.1]],
               [[0.3, 0.4],
                [0.2, 0.3],
                [0.1, 0.2]],
               [[0.6, 0.2],
                [0.7, 0.5],
                [0.9, 0.7]]])
        >>> cpd.variables
        ['grade', 'intel', 'diff']
        >>> cpd.cardinality
        array([3, 3, 2])
        >>> cpd.variable
        'grade'
        >>> cpd.variable_card
        3
        """
        if (
            len(self.variables) <= 1
            or (set(new_order) - set(self.variables))
            or (set(self.variables[1:]) - set(new_order))
        ):
            raise ValueError("New order either has missing or extra arguments")
        else:
            if new_order != self.variables[1:]:
                evidence = self.variables[1:]
                evidence_card = self.cardinality[1:]
                card_map = dict(zip(evidence, evidence_card))
                old_pos_map = dict(zip(evidence, range(len(evidence))))
                trans_ord = [0] + [(old_pos_map[letter] + 1) for letter in new_order]
                new_values = compat_fns.transpose(self.values, tuple(trans_ord))

                if inplace:
                    variables = [self.variables[0]] + new_order
                    cardinality = [self.variable_card] + [
                        card_map[var] for var in new_order
                    ]
                    super(TabularCPD, self).__init__(
                        variables, cardinality, new_values.flatten()
                    )
                    return self.get_values()
                else:
                    return new_values.reshape(
                        tuple(
                            [
                                self.cardinality[0],
                                np.prod([card_map[var] for var in new_order]),
                            ]
                        )
                    )
            else:
                warn("Same ordering provided as current")
                return self.get_values()

    def get_evidence(self):
        """
        Returns the evidence variables of the CPD.
        """
        return self.variables[:0:-1]

    @staticmethod
    def get_random(variable, evidence=None, cardinality=None, state_names={}, seed=42):
        """
        Generates a TabularCPD instance with random values on `variable` with
        parents/evidence `evidence` with cardinality/number of states as given
        in `cardinality`.

        Parameters
        ----------
        variable: str, int or any hashable python object.
            The variable on which to define the TabularCPD.

        evidence: list, array-like
            A list of variable names which are the parents/evidence of `variable`.

        cardinality: dict (default: None)
            A dict of the form {var_name: card} specifying the number of states/
            cardinality of each of the variables. If None, assigns each variable
            2 states.

        state_names: dict (default: {})
            A dict of the form {var_name: list of states} to specify the state names
            for the variables in the CPD. If state_names=None, integral state names
            starting from 0 is assigned.

        Returns
        -------
        Random CPD: pgmpy.factors.discrete.TabularCPD
            A TabularCPD object on `variable` with `evidence` as evidence with random values.

        Examples
        --------
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> TabularCPD(variable='A', evidence=['C', 'D'],
        ...            cardinality={'A': 3, 'B': 2, 'C': 4})
        <TabularCPD representing P(A:3 | C:4, B:2) at 0x7f95e22b8040>
        >>> TabularCPD(variable='A', evidence=['C', 'D'],
        ...            cardinality={'A': 2, 'B': 2, 'C': 2},
        ...            state_names={'A': ['a1', 'a2'],
        ...                         'B': ['b1', 'b2'],
        ...                         'C': ['c1', 'c2']})
        """
        generator = np.random.default_rng(seed=seed)

        if evidence is None:
            evidence = []

        if cardinality is None:
            cardinality = {var: 2 for var in chain([variable], evidence)}
        else:
            for var in chain([variable], evidence):
                if var not in cardinality.keys():
                    raise ValueError(f"Cardinality for variable: {var} not specified.")

        if len(evidence) == 0:
            values = generator.random((cardinality[variable], 1))
            values = values / np.sum(values, axis=0)
            node_cpd = TabularCPD(
                variable=variable,
                variable_card=cardinality[variable],
                values=values,
                state_names=state_names,
            )
        else:
            parent_card = [cardinality[var] for var in evidence]
            values = generator.random((cardinality[variable], np.prod(parent_card)))
            values = values / np.sum(values, axis=0)
            node_cpd = TabularCPD(
                variable=variable,
                variable_card=cardinality[variable],
                values=values,
                evidence=evidence,
                evidence_card=parent_card,
                state_names=state_names,
            )

        return node_cpd
