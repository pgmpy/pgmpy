#!/usr/bin/env python3
"""Contains the different formats of CPDs used in PGM"""

import numpy as np
from pgmpy.Factor import Factor
from pgmpy import Exceptions


class TabularCPD(Factor):
    """
    Public Methods
    --------------
    marginalize([variables_list])
    reduce([values_list])
    normalize()
    """
    def __init__(self, event, event_card, values,
                 evidence=None, evidence_card=None):
        """
        Defines the conditional probability distribution table (cpd table)

        Parameters
        ----------
        event: string
            event whose cpd table is defined
        event_card: integer
            cardinality of event
        values: 2d array, 2d list
            values of the cpd table

            example
            for a distribution of P(grade|intel, diff)

            diff:       easy                 hard
            intel: dumb   avg   smart    dumb  avg   smart
            gradeA: 0.1    0.1    0.1     0.1  0.1    0.1
            gradeB: 0.1    0.1    0.1     0.1  0.1    0.1
            gradeC: 0.8    0.8    0.8     0.8  0.8    0.8

            values should be
            [[0.1,0.1,0.1,0.1,0.1,0.1],
            [0.1,0.1,0.1,0.1,0.1,0.1],
            [0.8,0.8,0.8,0.8,0.8,0.8]]

        evidence: string, list-type
            evidences(if any) w.r.t. which cpd is defined
        evidence_card: integer, list-type
            cardinality of evidences

        """
        if not isinstance(event, str):
            raise TypeError("Event must be a string")
        self.event = event
        variables = [event]
        if not isinstance(event_card, int):
            raise TypeError("Event cardinality must be an integer")
        self.event_card = event_card
        cardinality = [event_card]
        if evidence_card:
            if not isinstance(evidence_card, (list, set, tuple)):
                evidence_card = [evidence_card]
            cardinality.extend(evidence_card)
        self.evidence_card = evidence_card
        if evidence:
            if not isinstance(evidence, (list, set, tuple)):
                evidence = [evidence]
            variables.extend(evidence)
            if not len(evidence_card) == len(evidence):
                raise Exceptions.CardinalityError("Cardinality of all "
                                                  "evidences not specified")
        self.evidence = evidence
        if len(np.array(values).shape) is not 2:
            raise TypeError("Values must be a 2d list/array")
        self.cpd = np.array(values)
        Factor.__init__(self, variables, cardinality, self.cpd.flatten('F'))

    def marginalize(self, variables):
        """
        Modifies the cpd table with marginalized values.

        Paramters
        ---------
        variables: string, list-type
            name of variable to be marginalized

        Examples
        --------
        >>> from pgmpy.Factor.CPD import TabularCPD
        >>> cpd_table = TabularCPD('grade', 2,
        ...                        [[0.7, 0.6, 0.6, 0.2],[0.3, 0.4, 0.4, 0.8]],
        ...                        ['intel', 'diff'], [2, 2])
        >>> cpd_table.marginalize('diff')
        >>> cpd_table.get_cpd()
        array([[ 1.3,  0.8],
               [ 0.7,  1.2]])
        """
        if self.event in variables:
            self.event_card = 1
        Factor.marginalize(self, variables)
        self.cpd = self.values.reshape((self.event_card,
                                        np.product(self.cardinality)/self.event_card),
                                       order='F')
        self.evidence = [var for var in self.variables
                         if var is not self.event]
        self.evidence_card = [self.get_cardinality(variable)
                              for variable in self.evidence]

    def normalize(self):
        """
        Normalizes the cpd table

        Examples
        --------
        >>> from pgmpy.Factor.CPD import TabularCPD
        >>> cpd_table = TabularCPD('grade', 2,
        ...                        [[0.7, 0.2, 0.6, 0.2],[0.4, 0.4, 0.4, 0.8]],
        ...                        ['intel', 'diff'], [2, 2])
        >>> cpd_table.normalize()
        >>> cpd_table.get_cpd()
        array([[ 0.63636364,  0.33333333,  0.6       ,  0.2       ],
               [ 0.36363636,  0.66666667,  0.4       ,  0.8       ]])
        """
        self.cpd = self.cpd / self.cpd.sum(axis=0)

    def reduce(self, values):
        """
        Reduces the cpd table to the context of given variable values.

        Parameters
        ----------
        values: string, list-type
            name of the variable values

        Examples
        --------
        >>> from pgmpy.Factor.CPD import TabularCPD
        >>> cpd_table = TabularCPD('grade', 2,
        ...                        [[0.7, 0.6, 0.6, 0.2],[0.3, 0.4, 0.4, 0.8]],
        ...                        ['intel', 'diff'], [2, 2])
        >>> cpd_table.reduce('diff_0')
        >>> cpd_table.get_cpd()
        array([[ 0.7,  0.6],
               [ 0.3,  0.4]])
        """
        if not isinstance(values, (list, set, tuple)):
            values = [values]
        if any(self.event in value for value in values):
            self.event_card = 1
        Factor.reduce(self, values)
        self.cpd = self.values.reshape((self.event_card,
                                        np.product(self.cardinality)/self.event_card),
                                       order='F')
        self.evidence = [var for var in self.variables
                         if var is not self.event]
        self.evidence_card = [self.get_cardinality(variable)
                              for variable in self.evidence]

    def get_cpd(self):
        """
        Returns the cpd
        """
        return self.cpd
