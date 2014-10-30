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
        Factor.__init__(self, variables, cardinality, self.cpd.flatten('C'))

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
                                       order='C')
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
                                       order='C')
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
        Base Class for Tree CPD.

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
        if data:
            for edge in data:
                self.add_edge(edge[0], edge[1], label=edge[2])

    def add_edge(self, u, v, label):
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

    def add_edges_from(self, ebunch):
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
        """
        Returns a RuleCPD object which represents the TreeCPD

        Examples
        --------
        >>> from pgmpy.Factor import CPD, Factor
        >>> tree = CPD.TreeCPD([('B', Factor(['A'], [2], [0.8, 0.2]), '0'),
        >>>                     ('B', 'C', '1'),
        >>>                     ('C', Factor(['A'], [2], [0.1, 0.9]), '0'),
        >>>                     ('C', 'D', '1'),
        >>>                     ('D', Factor(['A'], [2], [0.9, 0.1]), '0'),
        >>>                     ('D', Factor(['A'], [2], [0.4, 0.6]), '1')])
        >>> tree.to_rule_cpd()

        """
        #TODO: This method assumes that Factor class has a get_variable method. Check this after merging navin's PR.
        root = [node for node, in_degree in self.in_degree().items() if in_degree == 0][0]
        paths_root_to_factors = {target: path for target, path in nx.single_source_shortest_path(self, root).items() if isinstance(target, Factor)}
        for node in self.nodes_iter():
            if isinstance(node, Factor):
                rule_cpd = RuleCPD(node.get_variables()[0])

        for factor, path in paths_root_to_factors.items():
            rule_key = []
            for node_index in range(len(path) - 1):
                rule_key.append(path[node_index] + '_' + self.edge[path[node_index]][path[node_index + 1]]['label'])
            for value_index in range(len(factor.values)):
                rule_key.append(factor.get_variables()[0] + '_' + str(value_index))
                rule_cpd.add_rules({tuple(sorted(rule_key)): factor.values[value_index]})
        return rule_cpd


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
        >>>                      ('A_0', 'B_1', 'C_1'): 0.9,
        >>>                      ('A_1', 'B_1', 'C_1'): 0.1})
        """
        self.variable = variable
        if rules:
            self.rules = {}
            for rule, value in rules.items():
                self.rules[tuple(sorted(rule))] = value
        else:
            self.rules = {}
        verify = self._verify()
        if not verify[0]:
            del self
            raise ValueError(str(verify[1]) + " and " + str(verify[2]) + " point to the same assignment")

    def _verify(self):
        """
        Verifies the RuleCPD for multiple values of the
        assignment.
        """
        from itertools import combinations
        for rule, another_rule in combinations(self.rules, 2):
            rule, another_rule = (rule, another_rule) if len(rule) < len(another_rule) else (another_rule, rule)
            if not set(rule) - set(another_rule):
                return False, rule, another_rule
        return True,

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
        verify = self._verify()
        if not verify[0]:
            for rule in rules:
                del(self.rules[rule])
            raise ValueError(str(verify[1]) + " and " + str(verify[2]) + " point to the same assignment")

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

    def cardinality(self, variable=None):
        """
        Returns a dict of variable: cardinality.

        Parameters
        ----------
        variable: string, list
            variable or list of variables whose cardinality will be returned.

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
        cardinality = dict(Counter([element.split('_')[0] for element in assignments]))
        if variable:
            return cardinality[variable] if isinstance(variable, str) else {var: cardinality[var] for var in variable}
        else:
            return cardinality

    def to_tabular_cpd(self, parents_order=None):
        """
        Returns an equivalent TabularCPD.

        Parameters
        ----------
        parents_order: array-like. list, tuple. (optional)
            The order of the evidence variables.

        Examples
        --------
        >>> from pgmpy.Factor.CPD import RuleCPD
        >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
        >>>                      ('A_1', 'B_0'): 0.2,
        >>>                      ('A_0', 'B_1', 'C_1'): 0.9,
        >>>                      ('A_1', 'B_1', 'C_1'): 0.1,
        >>>                      ('A_0', 'B_1', 'C_0', 'D_0'): 0.4,
        >>>                      ('A_1', 'B_1', 'C_0', 'D_0'): 0.6,
        >>>                      ('A_0', 'B_1', 'C_0', 'D_1'): 0.3,
        >>>                      ('A_1', 'B_1', 'C_0', 'D_1'): 0.7})
        >>> rule.to_tabular_cpd()
        """
        if not parents_order:
            parents_order = sorted(list(self.scope() - {self.variable}))
        cardinality_dict = self.cardinality()
        tabular_cpd = [[0 for i in range(np.product(list(cardinality_dict.values())))]
                       for j in range(cardinality_dict[self.variable])]
        for rule, value in self.rules:
            start, end = 0, np.product(list(cardinality_dict.values()))
            for var in sorted(rule):
                if var.split('_')[0] != self.variable:
                    start, end = start + (end-start)/cardinality_dict[var] * int(var.split('_')[1]), \
                                 start + (end-start)/cardinality_dict[var] * (int(var.split('_')[1]) + 1)
                else:
                    var_assignment = int(var.split('_')[1])
            for index in range(start, end):
                tabular_cpd[var_assignment][index] = value

        return TabularCPD(self.variable, cardinality_dict[self.variable], tabular_cpd,
                          parents_order, [cardinality_dict[var] for var in parents_order])

    def _merge(self):
        """
        Removes the variable from the rules and then merges the rules
        having the same variables.
        For example:
        If we are given these rules:
        ('A_0', 'B_0'): 0.8,
        ('A_1', 'B_0'): 0.2,
        ('A_0', 'B_1', 'C_1'): 0.9,
        ('A_1', 'B_1', 'C_1'): 0.1,
        ('A_0', 'B_1', 'C_0', 'D_0'): 0.4,
        ('A_1', 'B_1', 'C_0', 'D_0'): 0.6,
        ('A_0', 'B_1', 'C_0', 'D_1'): 0.3,
        ('A_1', 'B_1', 'C_0', 'D_1'): 0.7

        then after merging _merge will return this dict:
        {('B_0',): array([ 0.8,  0.2]),
         ('B_1', 'C_0', 'D_1'): array([ 0.3,  0.7]),
         ('B_1', 'C_1'): array([ 0.9,  0.1]),
         ('B_1', 'C_0', 'D_0'): array([ 0.4,  0.6])}
        """
        var_card = self.cardinality(self.variable)
        dict_without_var = {}
        for assignments in self.rules.keys():
            dict_without_var[tuple(sorted([var for var in assignments if not var.startswith(self.variable)]))] = None
        for key in dict_without_var:
            value_list = []
            for assign in range(var_card):
                value_list.append(self.rules[tuple(sorted(list(key) + [(self.variable + '_' + str(assign))]))])
            dict_without_var[key] = np.array(value_list)
        return dict_without_var

    def to_tree_cpd(self):
        """
        Return a TreeCPD object which represents the RuleCPD.

        Examples
        --------
        >>> from pgmpy.Factor.CPD import RuleCPD
        >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
        >>>                      ('A_1', 'B_0'): 0.2,
        >>>                      ('A_0', 'B_1', 'C_1'): 0.9,
        >>>                      ('A_1', 'B_1', 'C_1'): 0.1,
        >>>                      ('A_0', 'B_1', 'C_0', 'D_0'): 0.4,
        >>>                      ('A_1', 'B_1', 'C_0', 'D_0'): 0.6,
        >>>                      ('A_0', 'B_1', 'C_0', 'D_1'): 0.3,
        >>>                      ('A_1', 'B_1', 'C_0', 'D_1'): 0.7})
        >>> rule.to_tree_cpd()
        <CPD.TreeCPD object at 0x7f6b6f952fd0>
        """
        from collections import OrderedDict
        tree_cpd = TreeCPD()
        merged_rules = OrderedDict(sorted(self._merge().items(), key=lambda t: len(t[0])))

        for assignments, value in merged_rules.items():
            for assignment_index in range(len(assignments) - 1):
                tree_cpd.add_edge(assignments[assignment_index].split('_')[0],
                                  assignments[assignment_index+1].split('_')[0],
                                  assignments[assignment_index].split('_')[1])
            tree_cpd.add_edge(assignments[-1].split('_')[0],
                              Factor([self.variable], [len(value)], value),
                              assignments[-1].split('_')[1])
        return tree_cpd

    def __str__(self):
        from collections import OrderedDict
        string = ""
        for index, key in enumerate(OrderedDict(sorted(self.rules.items(), key=lambda t: len(t[0])))):
            key_string = ', '.join(key)
            string += 'p' + str(index) + ': <' + key_string + '; ' + str(self.rules[key]) + '>' + '\n'
        return string

    __repr__ = __str__
