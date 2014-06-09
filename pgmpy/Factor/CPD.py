#!/usr/bin/env python3
"""Contains the different formats of CPDs used in PGM"""

import numpy as np
import networkx as nx
from pgmpy.Factor import Factor
from pgmpy import Exceptions


class TabularCPD(Factor):
    """
    Defines the conditional probability distribution table (cpd table)

    Example
    -------
    For a distribution of P(grade|intel, diff)

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

    Parameters
    ----------
    event: string
        event whose cpd table is defined
    event_card: integer
        cardinality of event
    values: 2d array, 2d list
        values of the cpd table
    evidence: string, list-type
        evidences(if any) w.r.t. which cpd is defined
    evidence_card: integer, list-type
        cardinality of evidences

    Public Methods
    --------------
    get_cpd()
    marginalize([variables_list])
    normalize()
    reduce([values_list])
    """
    def __init__(self, event, event_card, values,
                 evidence=None, evidence_card=None):
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


class TreeCPD(nx.DiGraph):
    """
    Base Class for Tree CPD.
    """
    def __init__(self, data=None):
        """
        Creates an empty Tree CPD.

        Parameters
        ----------
        data: input tree
            Data to initialize the tree. If data=None (default) an empty
            tree is created. The data can be an edge list with label for
            each edge. Label should be the observed value of the variable.

        Example
        -------
        For P(A|B, C, D), to construct a tree like:
                    B
             0 /         \1
              /          \
        P(A|b_0)         C
                   0/         \1
                   /           \
            P(A|b_1, c_0)      D
                          0/       \
                          /         \
            P(A|b_1,c_1,d_0)      P(A|b_1,c_1,d_1)

        >>> from pgmpy.Factor import CPD, Factor
        >>> tree = CPD.TreeCPD([('B', Factor(['A'], [2], [0.8, 0.2]), '0'),
        >>>                     ('B', 'C', '1'),
        >>>                     ('C', Factor(['A'], [2], [0.1, 0.9]), '0'),
        >>>                     ('C', 'D', '1'),
        >>>                     ('D', Factor(['A'], [2], [0.9, 0.1]), '0'),
        >>>                     ('D', Factor(['A'], [2], [0.4, 0.6]), '1')])

        """
        nx.DiGraph.__init__(self)
        #TODO: Check cycles and self loops.
        for edge in data:
            self.add_edge(edge[0], edge[1], label=edge[2])

    def add_edge(self, u, v, label, attr_dict=None, **attr):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u,v: nodes
            Nodes can be any hashable (and not None) Python object.
        label: string
            Label should be value of the variable observed.
            (underscore separated if multiple variables)
        attr_dict: dictionary, optional (default= no attributes)
            Dictionary of edge attributes. Key/Value pairs will
            update existing data associated with the edge.
        attr: Keyword arguments, optional
            Edge data can be assigned using keyword arguments.

        Examples
        --------

        """
        nx.DiGraph.add_edge(self, u, v, label=label)

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        """
        Add all the edges in ebunch.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the
            graph. The edges must be given as as 3-tuples (u,v,label).
        attr_dict : dictionary, optional (default= no attributes)
            Dictionary of edge attributes.  Key/value pairs will
            update existing data associated with each edge.
        attr : keyword arguments, optional
            Edge data (or labels or objects) can be assigned using
            keyword arguments.

        Examples
        --------
        >>> from pgmpy.Factor import CPD, Factor
        >>> tree = CPD.TreeCPD()
        >>> tree.add_edges_from([('B', 'C', '1'), ('C', 'D', '1'),
        >>>                     ('D', Factor(['A'], [2], [0.6, 0.4]))])
        """
        for edge in ebunch:
            if len(edge) == 2:
                raise ValueError("Each edge tuple must have 3 values (u,v,label).")
        nx.DiGraph.add_edges_from(self, [(edge[0], edge[1], {'label': edge[2]}) for edge in ebunch])

    def to_tabular_cpd(self, variable, parents_order=None):
        root = [node for node, in_degree in self.in_degree().items() if in_degree == 0][0]
        evidence = []
        evidence_card = []

        #dfs for finding the evidences and evidence cardinalities.
        def dfs(node):
            if isinstance(node, tuple):
                evidence.extend(node)
                labels = [value['label'] for value in self.edge[node].values()]
                for i in range(len(node)):
                    evidence_card.append(max([list(map(int, element.split('_'))) for element in labels], key=lambda t: t[i])[i] + 1)

            elif isinstance(node, str):
                evidence.append(node)
                evidence_card.append(self.out_degree(node))

            for out_edge in self.out_edges_iter(node):
                dfs(out_edge[1])

        dfs(root)

        if parents_order:
            #TODO: reorder the evidence and evidence_card list
            pass

    def to_rule_cpd(self):
        pass


class RuleCPD:
    def __init__(self, variable, rules=None):
        """
        Base class for Rule CPD.

        Parameters
        ----------
        variable: str
            The variable for which the CPD is to be defined.

        rules: dict. (optional)
            dict of rules. Each rule should be in the form of
            tuple_of_assignment: probability.
            For example: ('A_0', 'J_0'): 0.8

        Examples
        --------
        For constructing a RuleCPD on variable A with the following rules:
            p1: <A_0, B_0; 0.8>
            p2: <A_1, B_0; 0.2>
            p3: <A_0, B_1, C_0; 0.4>
            p4: <A_1, B_1, C_0; 0.6>
            p5: <A_0, B_1, C_1; 0.9>
            p6: <A_1, B_1, C_1; 0.1>

        >>> from pgmpy.Factor.CPD import RuleCPD
        >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
        >>>                      ('A_1', 'B_0'): 0.2,
        >>>                      ('A_0', 'B_1', 'C_0'): 0.4,
        >>>                      ('A_1', 'B_1', 'C_0'): 0.6,
        >>>                      ('A_0', 'B_1', 'C_!'): 0.9,
        >>>                      ('A_1', 'B_1', 'C_1'): 0.1}
        """
        self.variable = variable
        self.rules = rules
        self._verify(to_delete=True)

    def _verify(self, to_delete=False):
        pass

    def add_rules(self, rules):
        """
        Add one or more rules to the Rule CPD.

        Parameters
        ----------
        rules: dict
            dict of rules. Each rule should be in the form of
            tuple_of_assignment: probability.
            For example: ('A_0', 'J_0'): 0.8

        Examples
        --------
        >>> from pgmpy.Factor.CPD import RuleCPD
        >>> rule = RuleCPD(variable='A')
        >>> rule.add_rules({('A_0', 'B_0'): 0.8,
        >>>                 ('A_1', 'B_0'): 0.2})
        """
        for rule in rules:
            self.rules[rule] = rules[rule]
        if not self._verify(to_delete=False):
            for rule in rules:
                del(self.rules[rule])
            raise ValueError("Please check the values of the rules")

    def scope(self):
        """
        Returns a set of variables which is the scope of the Rule CPD.

        Examples
        --------
        >>> from pgmpy.Factor.CPD import RuleCPD
        >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
        >>>                      ('A_1', 'B_0'): 0.2,
        >>>                      ('A_0', 'B_1', 'C_0'): 0.4,
        >>>                      ('A_1', 'B_1', 'C_0'): 0.6,
        >>>                      ('A_0', 'B_1', 'C_!'): 0.9,
        >>>                      ('A_1', 'B_1', 'C_1'): 0.1}
        >>> rule.scope()
        {'A', 'B', 'C'}
        """
        scope = set()
        for rule in self.rules:
            scope.update([assignment.split('_')[0] for assignment in rule])
        return scope

    def cardinality(self):
        """
        Returns a dict of variable: cardinality.

        Examples
        --------
        >>> from pgmpy.Factor.CPD import RuleCPD
        >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
        >>>                      ('A_1', 'B_0'): 0.2,
        >>>                      ('A_0', 'B_1', 'C_0'): 0.4,
        >>>                      ('A_1', 'B_1', 'C_0'): 0.6,
        >>>                      ('A_0', 'B_1', 'C_!'): 0.9,
        >>>                      ('A_1', 'B_1', 'C_1'): 0.1}
        >>> rule.cardinality()
        {'A': 2, 'B': 2, 'C': 2}
        """
        from itertools import chain
        from collections import Counter
        assignments = list(set(chain(*self.rules)))
        return dict(Counter([element.split('_')[0] for element in assignments]))


    def to_tree_cpd(self):
        pass

    def __str__(self):
        for index, key in enumerate(self.rules):
            key_string = ', '.join(key)
            print('p' + index + ':<' + key_string + '; ' + str(self.rules[key]) + '>')