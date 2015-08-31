#!/usr/bin/env python3
"""Contains the different formats of CPDs used in PGM"""

from itertools import product

import networkx as nx
import numpy as np

from pgmpy import exceptions
from pgmpy.factors import Factor
from pgmpy.extern import tabulate


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

    >>> from pgmpy.factors.CPD import TabularCPD
    >>> cpd = TabularCPD('grade', 3, [[0.1, 0.1],
    ...                               [0.1, 0.1],
    ...                               [0.8, 0.8]],
    ...                  evidence='evi1', evidence_card=2)
    >>> cpd.values
    array([ 0.1,  0.1,  0.1,  0.1,  0.8,  0.8])
    >>> cpd.variables
    OrderedDict([('grade', ['grade_0', 'grade_1', 'grade_2']), ('evi1', ['evi1_0', 'evi1_1'])])
    >>> cpd.variable
    'grade'

    Parameters
    ----------
    variable: int, string (any hashable python object)
        The variable whose CPD is defined.

    variable_card: integer
        cardinality of variable

    values: 2d array, 2d list
        values of the cpd table

    evidence: string, array-like
        evidences(if any) w.r.t. which cpd is defined

    evidence_card: integer, array-like
        cardinality of evidences (if any)

    Public Methods
    --------------
    get_cpd()
    marginalize([variables_list])
    normalize()
    reduce([values_list])
    """
    def __init__(self, variable, variable_card, values,
                 evidence=None, evidence_card=None):

        self.variable = variable
        self.variable_card = None
        self.evidence = None
        self.evidence_card = None

        variables = [variable]

        if not isinstance(variable_card, int):
            raise TypeError("Event cardinality must be an integer")
        self.variable_card = variable_card

        cardinality = [variable_card]
        if evidence_card is not None:
            if not isinstance(evidence_card, (list, set, tuple)):
                if isinstance(evidence_card, np.ndarray):
                    evidence_card = evidence_card.tolist()
                elif isinstance(evidence_card, (int, float)):
                    evidence_card = [evidence_card]
                else:
                    raise TypeError("Evidence cardinality must be list, set, "
                                    "tuple, numpy ndarray or simply a number.")
            self.evidence_card = evidence_card
            cardinality.extend(evidence_card[::-1])

        if evidence is not None:
            if not isinstance(evidence, (list, set, tuple)):
                if isinstance(evidence, np.ndarray):
                    evidence = evidence.tolist()
                elif isinstance(evidence, str):
                    evidence = [evidence]
                else:
                    raise TypeError("Evidence must be list, set, tuple or array"
                                    " of strings.")
            self.evidence = evidence
            variables.extend(evidence[::-1])
            if not len(evidence_card) == len(evidence):
                raise exceptions.CardinalityError("Cardinality of all "
                                                  "evidences not specified")
        values = np.array(values)
        if values.ndim != 2:
            raise TypeError("Values must be a 2D list/array")

        super().__init__(variables, cardinality, values.flatten('C'))

    def __repr__(self):
        var_str = '<TabularCPD representing P({var}:{card}'.format(
                            var=self.variable, card=self.variable_card)

        if self.evidence:
            evidence_str = ' | ' + ', '.join(['{var}:{card}'.format(var=var, card=card)
                                              for var, card in zip(self.evidence, self.evidence_card)])
        else:
            evidence_str = ''

        return var_str + evidence_str + ') at {address}>'.format(address=hex(id(self)))

    def get_cpd(self):
        """
        Returns the cpd
        """
        if self.variable in self.variables:
            return self.values.reshape(self.cardinality[0], np.prod(self.cardinality[1:]))
        else:
            return self.values.reshape(1, np.prod(self.cardinality))

    def __str__(self):
        table_list = []
        indexes = np.array(list(product(*[range(i) for i in self.cardinality[1:]])))
        scope = self.scope()
        for i in range(1, len(self.cardinality)):
            row = [scope[i]]
            row.extend(np.array(self.variables[row[0]])[indexes[:, i-1]].tolist())
            table_list.append(row)
        variable_array = np.array([str([v.var, v.state]) for v in self.variables[self.variable]])
        table_list.extend(np.column_stack((variable_array, self.get_cpd())))
        return tabulate(table_list, tablefmt="fancy_grid")

    def _repr_html_(self):
        """
        Print TabularCPD in form of a table in IPython Notebook
        """
        string_list = []

        cpd_value = self.get_cpd()

        html_string_header = (
            """<table><caption>TabularCPD for <b>%s</b></caption>""" % str(self.variable))
        string_list.append(html_string_header)

        if self.evidence:
            cpd_evidence_card = [card for card in self.evidence_card]
            cpd_evidence_card = cpd_evidence_card[::-1]
            cpd_evidence_card.insert(0, 1)
            cum_card = np.cumprod(cpd_evidence_card)
            max_card = cum_card[-1]

            evidence = [var for var in self.evidence]
            evidence = evidence[::-1]

            for i in range(len(evidence)):
                var = str(evidence[i])
                card = cpd_evidence_card[i + 1]
                num_repeat = cum_card[i]
                col_span = max_card / (num_repeat * card)

                html_substring_header = """<tr><td><b>%s</b></td>""" % var
                string_list.append(html_substring_header)

                for j in range(num_repeat):
                    for k in range(card):
                        html_substring = ("<td colspan=%d>%s_%d</td>" %
                                          (col_span, var, k))
                        string_list.append(html_substring)
                string_list.append("</tr>")

        m, n = cpd_value.shape
        for i in range(m):
            html_substring = """<tr><td><b>%s_%d</b></td>""" % (str(self.variable), i)
            string_list.append(html_substring)
            for j in range(n):
                html_substring = """<td>%4.4f</td>""" % cpd_value[i, j]
                string_list.append(html_substring)
            string_list.append("</tr>")

        string_list.append("</table>")

        return ''.join(string_list)

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
        >>> from pgmpy.factors import TabularCPD
        >>> cpd_table = TabularCPD('grade', 2,
        ...                        [[0.7, 0.2, 0.6, 0.2],[0.4, 0.4, 0.4, 0.8]],
        ...                        ['intel', 'diff'], [2, 2])
        >>> cpd_table.normalize()
        >>> cpd_table.get_cpd()
        array([[ 0.63636364,  0.33333333,  0.6       ,  0.2       ],
               [ 0.36363636,  0.66666667,  0.4       ,  0.8       ]])
        """
        if inplace:
            tabular_cpd = self
        else:
            tabular_cpd = TabularCPD(self.variable, self.variable_card,
                                     self.get_cpd(), self.evidence,
                                     self.evidence_card)
        cpd = tabular_cpd.get_cpd()
        tabular_cpd.values = (cpd / cpd.sum(axis=0)).flatten('C')

        if not inplace:
            return tabular_cpd

    def marginalize(self, variables, inplace=True):
        """
        Modifies the cpd table with marginalized values.

        Paramters
        ---------
        variables: string, list-type
            name of variable to be marginalized
        inplace: boolean
            If inplace=True it will modify the CPD itself, else would return
            a new CPD

        Examples
        --------
        >>> from pgmpy.factors import TabularCPD
        >>> cpd_table = TabularCPD('grade', 2,
        ...                        [[0.7, 0.6, 0.6, 0.2],[0.3, 0.4, 0.4, 0.8]],
        ...                        ['intel', 'diff'], [2, 2])
        >>> cpd_table.marginalize('diff')
        >>> cpd_table.get_cpd()
        array([[ 0.48484848,  0.4       ],
               [ 0.51515152,  0.6       ]])
        """
        if inplace:
            tabular_cpd = self
        else:
            tabular_cpd = TabularCPD(self.variable, self.variable_card,
                                     self.get_cpd(), self.evidence,
                                     self.evidence_card)

        super(TabularCPD, tabular_cpd).marginalize(variables)
        tabular_cpd.normalize()

        if not inplace:
            return tabular_cpd

    def reduce(self, values, inplace=True):
        """
        Reduces the cpd table to the context of given variable values.

        Parameters
        ----------
        values: string, list-type
            name of the variable values
        inplace: boolean
            If inplace=True it will modify the CPD itself, else would return
            a new CPD

        Examples
        --------
        >>> from pgmpy.factors import TabularCPD
        >>> cpd_table = TabularCPD('grade', 2,
        ...                        [[0.7, 0.6, 0.6, 0.2],[0.3, 0.4, 0.4, 0.8]],
        ...                        ['intel', 'diff'], [2, 2])
        >>> cpd_table.reduce('diff_0')
        >>> cpd_table.get_cpd()
        array([[ 0.7,  0.6],
               [ 0.3,  0.4]])
        """
        if inplace:
            tabular_cpd = self
        else:
            tabular_cpd = TabularCPD(self.variable, self.variable_card,
                                     self.get_cpd(), self.evidence,
                                     self.evidence_card)

        super(TabularCPD, tabular_cpd).reduce(values)
        tabular_cpd.normalize()

        if not inplace:
            return tabular_cpd

    def to_factor(self):
        """
        Returns an equivalent factor with the same variables, cardinality, values as that of the cpd

        Examples
        --------
        >>> from pgmpy.factors.CPD import TabularCPD
        >>> cpd = TabularCPD('grade', 3, [[0.1, 0.1],
        ...                               [0.1, 0.1],
        ...                               [0.8, 0.8]],
        ...                  evidence='evi1', evidence_card=2)
        >>> factor = cpd.to_factor()
        >>> factor

        """
        return Factor(self.variables, self.cardinality, self.values)


# Commenting out because not used anywhere for now and not implemented in a very good way.
# class TreeCPD(nx.DiGraph):
#     """
#     Base Class for Tree CPD.
#     """
#     def __init__(self, data=None):
#         """
#         Base Class for Tree CPD.
#
#         Parameters
#         ----------
#         data: input tree
#             Data to initialize the tree. If data=None (default) an empty
#             tree is created. The data can be an edge list with label for
#             each edge. Label should be the observed value of the variable.
#
#         Examples
#         --------
#         For P(A|B, C, D), to construct a tree like:
#
#                     B
#              0 /        \1
#               /          \
#         P(A|b_0)          C
#                    0/         \1
#                    /           \
#             P(A|b_1, c_0)      D
#                           0/       \
#                           /         \
#             P(A|b_1,c_1,d_0)      P(A|b_1,c_1,d_1)
#
#         >>> from pgmpy.factors import TreeCPD, Factor
#         >>> tree = TreeCPD([('B', Factor(['A'], [2], [0.8, 0.2]), '0'),
#         ...                 ('B', 'C', '1'),
#         ...                 ('C', Factor(['A'], [2], [0.1, 0.9]), '0'),
#         ...                 ('C', 'D', '1'),
#         ...                 ('D', Factor(['A'], [2], [0.9, 0.1]), '0'),
#         ...                 ('D', Factor(['A'], [2], [0.4, 0.6]), '1')])
#         """
#         nx.DiGraph.__init__(self)
#         # TODO: Check cycles and self loops.
#         if data:
#             for edge in data:
#                 if len(edge) != 3:
#                     raise ValueError("Each edge tuple must have 3 values (u, v, label).")
#                 self.add_edge(edge[0], edge[1], label=edge[2])
#
#     def add_edge(self, u, v, label):
#         """
#         Add an edge between u and v.
#
#         The nodes u and v will be automatically added if they are
#         not already in the graph.
#
#         Parameters
#         ----------
#         u,v: nodes
#             Nodes can be any hashable (and not None) Python object.
#         label: string
#             Label should be value of the variable observed.
#             (underscore separated if multiple variables)
#         attr_dict: dictionary, optional (default= no attributes)
#             Dictionary of edge attributes. Key/Value pairs will
#             update existing data associated with the edge.
#         attr: Keyword arguments, optional
#             Edge data can be assigned using keyword arguments.
#
#         Examples
#         --------
#         >>> from pgmpy.factors import TreeCPD, Factor
#         >>> tree = TreeCPD([('B', Factor(['A'], [2], [0.8, 0.2]), 0),
#         ...                 ('B', 'C', 1)])
#         >>> tree.add_edge('C', Factor(['A'], [2], [0.1, 0.9]), label=0)
#         """
#         if u != v:
#             if u in self.nodes() and v in self.nodes() and nx.has_path(self, v, u):
#                 # check if adding edge (u, v) forms a cycle
#                 raise ValueError(
#                     'Loops are not allowed. Adding the edge from (%s->%s) forms a loop.' % (u, v))
#             else:
#                 super(TreeCPD, self).add_edge(u, v, label=label)
#         else:
#             raise ValueError('Self loops are not allowed. Edge (%s->%s) forms a self loop.' % (u, v))
#
#     def add_edges_from(self, ebunch):
#         """
#         Add all the edges in ebunch.
#
#         Parameters
#         ----------
#         ebunch : container of edges
#             Each edge given in the container will be added to the
#             graph. The edges must be given as as 3-tuples (u,v,label).
#         attr_dict : dictionary, optional (default= no attributes)
#             Dictionary of edge attributes.  Key/value pairs will
#             update existing data associated with each edge.
#         attr : keyword arguments, optional
#             Edge data (or labels or objects) can be assigned using
#             keyword arguments.
#
#         Examples
#         --------
#         >>> from pgmpy.factors import TreeCPD, Factor
#         >>> tree = TreeCPD()
#         >>> tree.add_edges_from([('B', 'C', 1), ('C', 'D', 1),
#         ...                      ('D', Factor(['A'], [2], [0.6, 0.4]))])
#         """
#         for edge in ebunch:
#             if len(edge) == 2:
#                 raise ValueError("Each edge tuple must have 3 values (u, v, label).")
#         nx.DiGraph.add_edges_from(self, [(edge[0], edge[1], {'label': edge[2]}) for edge in ebunch])
#
#     def to_tabular_cpd(self, parents_order=None):
#         edge_attributes = nx.get_edge_attributes(self, 'label')
#         edge_values = {}
#         edge_dict = {}
#         adjlist = {}
#         node_list = []
#         stack = []
#         values = []
#         cardinality = []
#
#         for edge in edge_attributes:
#             edge_dict.setdefault(edge[0], []).append(edge_attributes[edge])
#             if isinstance(edge[1], Factor):
#                 variable = edge[1].scope()
#                 variable_card = edge[1].cardinality
#                 edge_values[(edge[0], edge[0] + edge_attributes.get(edge))] = edge[1].values.tolist()
#             else:
#                 edge_values[(edge[0], edge[0] + edge_attributes.get(edge))] = edge[1]
#         #adjlist
#         for source in self.nodes():
#             if not isinstance(source, Factor):
#                 adjlist[source] = [i[1] for i in edge_attributes if not isinstance(i[1], Factor) and i[0] == source]
#                 adjlist[source] = sorted(adjlist[source], key=lambda x: (len(nx.descendants(self, x)), x))
#
#         root = [node for node, in_degree in self.in_degree().items() if in_degree == 0][0]
#         stack.append(root)
#
#         #dfs
#         while stack:
#             top_node = stack[-1]
#             node_list.append(top_node)
#             stack = stack[:-1]
#             for end_node in adjlist[top_node]:
#                 stack.append(end_node)
#
#         for node in node_list:
#             cardinality.append(len(edge_dict[node]))
#
#         for i in product(*[range(index) for index in cardinality]):
#             edge_list = [a + str(b) for a, b in zip(node_list, i)]
#             current_node = root
#             for edge in edge_list:
#                 if (current_node, edge) in edge_values.keys():
#                     if not isinstance(edge_values[(current_node, edge)], list):
#                         current_node = edge_values[(current_node, edge)]
#                     else:
#                         values.append(edge_values[(current_node, edge)])
#                         break
#
#         values = np.array(values).flatten('F').reshape((len(values[0]), len(values)))
#         return TabularCPD(variable[0], int(variable_card[0]), values, node_list[::-1], cardinality)
#
#     def to_rule_cpd(self):
#         """
#         Returns a RuleCPD object which represents the TreeCPD
#
#         Examples
#         --------
#         >>> from pgmpy.factors import TreeCPD, Factor
#         >>> tree = TreeCPD([('B', factors(['A'], [2], [0.8, 0.2]), '0'),
#         ...                 ('B', 'C', '1'),
#         ...                 ('C', factors(['A'], [2], [0.1, 0.9]), '0'),
#         ...                 ('C', 'D', '1'),
#         ...                 ('D', factors(['A'], [2], [0.9, 0.1]), '0'),
#         ...                 ('D', factors(['A'], [2], [0.4, 0.6]), '1')])
#         >>> tree.to_rule_cpd()
#
#         """
#         # TODO: This method assumes that factors class has a get_variable method. Check this after merging navin's PR.
#         root = [node for node, in_degree in self.in_degree().items() if in_degree == 0][0]
#         paths_root_to_factors = {target: path for target, path in nx.single_source_shortest_path(self, root).items() if
#                                  isinstance(target, Factor)}
#         for node in self.nodes_iter():
#             if isinstance(node, Factor):
#                 rule_cpd = RuleCPD(node.scope()[0])
#
#         for factor, path in paths_root_to_factors.items():
#             rule_key = []
#             for node_index in range(len(path) - 1):
#                 rule_key.append(path[node_index] + '_' + self.edge[path[node_index]][path[node_index + 1]]['label'])
#             for value_index in range(len(factor.values)):
#                 rule_key.append(factor.get_variables()[0] + '_' + str(value_index))
#                 rule_cpd.add_rules({tuple(sorted(rule_key)): factor.values[value_index]})
#         return rule_cpd
#
#
# class RuleCPD:
#     def __init__(self, variable, rules=None):
#         """
#         Base class for Rule CPD.
#
#         Parameters
#         ----------
#         variable: str
#             The variable for which the CPD is to be defined.
#
#         rules: dict. (optional)
#             dict of rules. Each rule should be in the form of
#             tuple_of_assignment: probability.
#             For example: ('A_0', 'J_0'): 0.8
#
#         Examples
#         --------
#         For constructing a RuleCPD on variable A with the following rules:
#             p1: <A_0, B_0; 0.8>
#             p2: <A_1, B_0; 0.2>
#             p3: <A_0, B_1, C_0; 0.4>
#             p4: <A_1, B_1, C_0; 0.6>
#             p5: <A_0, B_1, C_1; 0.9>
#             p6: <A_1, B_1, C_1; 0.1>
#
#         >>> from pgmpy.factors import RuleCPD
#         >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
#         ...                      ('A_1', 'B_0'): 0.2,
#         ...                      ('A_0', 'B_1', 'C_0'): 0.4,
#         ...                      ('A_1', 'B_1', 'C_0'): 0.6,
#         ...                      ('A_0', 'B_1', 'C_1'): 0.9,
#         ...                      ('A_1', 'B_1', 'C_1'): 0.1})
#         """
#         self.variable = variable
#         if rules:
#             self.rules = {}
#             for rule, value in rules.items():
#                 self.rules[tuple(sorted(rule))] = value
#         else:
#             self.rules = {}
#         verify = self._verify()
#         if not verify[0]:
#             del self
#             raise ValueError(str(verify[1]) + " and " + str(verify[2]) + " point to the same assignment")
#
#     def _verify(self):
#         """
#         Verifies the RuleCPD for multiple values of the
#         assignment.
#         """
#         from itertools import combinations
#         for rule, another_rule in combinations(self.rules, 2):
#             rule, another_rule = (rule, another_rule) if len(rule) < len(another_rule) else (another_rule, rule)
#             if not set(rule) - set(another_rule):
#                 return False, rule, another_rule
#         return True,
#
#     def add_rules(self, rules):
#         """
#         Add one or more rules to the Rule CPD.
#
#         Parameters
#         ----------
#         rules: dict
#             dict of rules. Each rule should be in the form of
#             tuple_of_assignment: probability.
#             For example: ('A_0', 'J_0'): 0.8
#
#         Examples
#         --------
#         >>> from pgmpy.factors import RuleCPD
#         >>> rule = RuleCPD(variable='A')
#         >>> rule.add_rules({('A_0', 'B_0'): 0.8,
#         ...                 ('A_1', 'B_0'): 0.2})
#         """
#         for rule in rules:
#             self.rules[rule] = rules[rule]
#         verify = self._verify()
#         if not verify[0]:
#             for rule in rules:
#                 del(self.rules[rule])
#             raise ValueError(str(verify[1]) + " and " + str(verify[2]) + " point to the same assignment")
#
#     def scope(self):
#         """
#         Returns a set of variables which is the scope of the Rule CPD.
#
#         Examples
#         --------
#         >>> from pgmpy.factors import RuleCPD
#         >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
#         >>>                      ('A_1', 'B_0'): 0.2,
#         >>>                      ('A_0', 'B_1', 'C_0'): 0.4,
#         >>>                      ('A_1', 'B_1', 'C_0'): 0.6,
#         >>>                      ('A_0', 'B_1', 'C_!'): 0.9,
#         >>>                      ('A_1', 'B_1', 'C_1'): 0.1}
#         >>> rule.scope()
#         {'A', 'B', 'C'}
#         """
#         scope = set()
#         for rule in self.rules:
#             scope.update([assignment.split('_')[0] for assignment in rule])
#         return scope
#
#     def cardinality(self, variable=None):
#         """
#         Returns a dict of variable: cardinality.
#
#         Parameters
#         ----------
#         variable: string, list
#             variable or list of variables whose cardinality will be returned.
#
#         Examples
#         --------
#         >>> from pgmpy.factors import RuleCPD
#         >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
#         >>>                      ('A_1', 'B_0'): 0.2,
#         >>>                      ('A_0', 'B_1', 'C_0'): 0.4,
#         >>>                      ('A_1', 'B_1', 'C_0'): 0.6,
#         >>>                      ('A_0', 'B_1', 'C_!'): 0.9,
#         >>>                      ('A_1', 'B_1', 'C_1'): 0.1}
#         >>> rule.cardinality()
#         {'A': 2, 'B': 2, 'C': 2}
#         """
#         from itertools import chain
#         from collections import Counter
#         assignments = set(chain.from_iterable(self.rules))
#         cardinality = dict(Counter([element.split('_')[0] for element in assignments]))
#         if variable:
#             return cardinality[variable] if isinstance(variable, str) else {var: cardinality[var] for var in variable}
#         else:
#             return cardinality
#
#     def to_tabular_cpd(self, parents_order=None):
#         """
#         Returns an equivalent TabularCPD.
#
#         Parameters
#         ----------
#         parents_order: array-like. list, tuple. (optional)
#             The order of the evidence variables.
#
#         Examples
#         --------
#         >>> from pgmpy.factors import RuleCPD
#         >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
#         >>>                      ('A_1', 'B_0'): 0.2,
#         >>>                      ('A_0', 'B_1', 'C_1'): 0.9,
#         >>>                      ('A_1', 'B_1', 'C_1'): 0.1,
#         >>>                      ('A_0', 'B_1', 'C_0', 'D_0'): 0.4,
#         >>>                      ('A_1', 'B_1', 'C_0', 'D_0'): 0.6,
#         >>>                      ('A_0', 'B_1', 'C_0', 'D_1'): 0.3,
#         >>>                      ('A_1', 'B_1', 'C_0', 'D_1'): 0.7})
#         >>> rule.to_tabular_cpd()
#         """
#         if not parents_order:
#             parents_order = sorted(self.scope() - {self.variable})
#         cardinality_dict = self.cardinality()
#         cardinality_product = np.product(list(cardinality_dict.values()))
#         tabular_cpd = [[0] * cardinality_product
#                        for _ in range(cardinality_dict[self.variable])]
#         for rule, value in self.rules:
#             start, end = 0, cardinality_product
#             for var in sorted(rule):
#                 if var.split('_')[0] != self.variable:
#                     start, end = (start + (end-start)/cardinality_dict[var] * int(var.split('_')[1]),
#                                   start + (end-start)/cardinality_dict[var] * (int(var.split('_')[1]) + 1))
#                 else:
#                     var_assignment = int(var.split('_')[1])
#             for index in range(start, end):
#                 tabular_cpd[var_assignment][index] = value
#
#         return TabularCPD(self.variable, cardinality_dict[self.variable], tabular_cpd,
#                           parents_order, [cardinality_dict[var] for var in parents_order])
#
#     def _merge(self):
#         """
#         Removes the variable from the rules and then merges the rules
#         having the same variables.
#         For example:
#         If we are given these rules:
#         ('A_0', 'B_0'): 0.8,
#         ('A_1', 'B_0'): 0.2,
#         ('A_0', 'B_1', 'C_1'): 0.9,
#         ('A_1', 'B_1', 'C_1'): 0.1,
#         ('A_0', 'B_1', 'C_0', 'D_0'): 0.4,
#         ('A_1', 'B_1', 'C_0', 'D_0'): 0.6,
#         ('A_0', 'B_1', 'C_0', 'D_1'): 0.3,
#         ('A_1', 'B_1', 'C_0', 'D_1'): 0.7
#
#         then after merging _merge will return this dict:
#         {('B_0',): array([ 0.8,  0.2]),
#          ('B_1', 'C_0', 'D_1'): array([ 0.3,  0.7]),
#          ('B_1', 'C_1'): array([ 0.9,  0.1]),
#          ('B_1', 'C_0', 'D_0'): array([ 0.4,  0.6])}
#         """
#         var_card = self.cardinality(self.variable)
#         dict_without_var = {}
#         for assignments in self.rules.keys():
#             dict_without_var[tuple(sorted([var for var in assignments if not var.startswith(self.variable)]))] = None
#         for key in dict_without_var:
#             value_list = []
#             for assign in range(var_card):
#                 value_list.append(self.rules[tuple(sorted(list(key) + [(self.variable + '_' + str(assign))]))])
#             dict_without_var[key] = np.array(value_list)
#         return dict_without_var
#
#     def to_tree_cpd(self):
#         """
#         Return a TreeCPD object which represents the RuleCPD.
#
#         Examples
#         --------
#         >>> from pgmpy.factors import RuleCPD
#         >>> rule = RuleCPD('A', {('A_0', 'B_0'): 0.8,
#         ...                      ('A_1', 'B_0'): 0.2,
#         ...                      ('A_0', 'B_1', 'C_1'): 0.9,
#         ...                      ('A_1', 'B_1', 'C_1'): 0.1,
#         ...                      ('A_0', 'B_1', 'C_0', 'D_0'): 0.4,
#         ...                      ('A_1', 'B_1', 'C_0', 'D_0'): 0.6,
#         ...                      ('A_0', 'B_1', 'C_0', 'D_1'): 0.3,
#         ...                      ('A_1', 'B_1', 'C_0', 'D_1'): 0.7})
#         >>> rule.to_tree_cpd()
#         <CPD.TreeCPD object at 0x7f6b6f952fd0>
#         """
#         from collections import OrderedDict
#         tree_cpd = TreeCPD()
#         merged_rules = OrderedDict(sorted(self._merge().items(), key=lambda t: len(t[0])))
#
#         for assignments, value in merged_rules.items():
#             for assignment_index in range(len(assignments) - 1):
#                 tree_cpd.add_edge(assignments[assignment_index].split('_')[0],
#                                   assignments[assignment_index+1].split('_')[0],
#                                   assignments[assignment_index].split('_')[1])
#             tree_cpd.add_edge(assignments[-1].split('_')[0],
#                               Factor([self.variable], [len(value)], value),
#                               assignments[-1].split('_')[1])
#         return tree_cpd
#
#     def __str__(self):
#         from collections import OrderedDict
#         string = ""
#         for index, key in enumerate(OrderedDict(sorted(self.rules.items(), key=lambda t: len(t[0])))):
#             key_string = ', '.join(key)
#             string += 'p' + str(index) + ': <' + key_string + '; ' + str(self.rules[key]) + '>' + '\n'
#         return string
