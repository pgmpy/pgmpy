#!/usr/bin/env python3

import itertools
import numpy as np
import networkx as nx

from pgmpy.base import UndirectedGraph


class MarkovModel(nx.Graph):
    """
        Base class for markov model.

        A MarkovModel stores nodes and edges with potentials

        MarkovModel hold undirected edges.

        Parameters
        ----------
        data : input graph
            Data to initialize graph.  If data=None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.

        Examples
        --------
        Create an empty Markov Model with no nodes and no edges.

        >>> from pgmpy.models import MarkovModel
        >>> G = MarkovModel()

        G can be grown in several ways.

        **Nodes:**

        Add one node at a time:

        >>> G.add_node('a')

        Add the nodes from any container (a list, set or tuple or the nodes
        from another graph).

        >>> G.add_nodes_from(['a', 'b'])

        **Edges:**

        G can also be grown by adding edges.

        Add one edge,

        >>> G.add_edge('a', 'b')

        a list of edges,

        >>> G.add_edges_from([('a', 'b'), ('b', 'c')])

        If some edges connect nodes not yet in the model, the nodes
        are added automatically.  There are no errors when adding
        nodes or edges that already exist.

        **Shortcuts:**

        Many common graph features allow python syntax to speed reporting.

        >>> 'a' in G     # check if node in graph
        True
        >>> len(G)  # number of nodes in graph
        3

        Public Methods
        --------------
        add_node('node1')
        add_nodes_from(['node1', 'node2', ...])
        add_edge('node1', 'node2')
        add_edges_from([('node1', 'node2'),('node3', 'node4')])
        add_states({node : [state1, state2]})
        get_states('node1')
        number_of_state('node1')
        set_observations({'node1': ['observed_state1', 'observed_state2'],
                          'node2': 'observed_state1'})
        is_observed('node1')
        """

    def __init__(self, ebunch=None):
        super(MarkovModel, self).__init__(ebunch)
        for node in self.nodes():
            self._set_is_observed(node, False)
        self.factors = []
        self.cardinality = {}

    def add_node(self, node):
        """
        Add a single node to the Graph.

        Parameters
        ----------
        node: node
            A node can be any hashable Python object.

        See Also
        --------
        add_nodes_from : add a collection of nodes

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> G = MarkovModel()
        >>> G.add_node('A')
        """
        super(MarkovModel, self).add_node(node)
        self._set_is_observed(node, False)

    def add_nodes_from(self, nodes):
        """
        Add multiple nodes to the Graph.

        Parameters
        ----------
        nodes: iterable container
               A container of nodes (list, dict, set, etc.).

        See Also
        --------
        add_node : add a single node

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> G = MarkovModel()
        >>> G.add_nodes_from(['A', 'B', 'C'])
        """
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u, v):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u,v : nodes
            Nodes can be any hashable Python object.

        See Also
        --------
        add_edges_from : add a collection of edges

        EXAMPLE
        -------
        >>> from pgmpy.models import MarkovModel
        >>> G = MarkovModel()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edge('Alice', 'Bob')
        """
        # Need to check that there is no self loop.
        if u != v:
            super(MarkovModel, self).add_edge(u, v)
        else:
            raise ValueError('Self loops are not allowed')

    def add_edges_from(self, ebunch):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names should be strings.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        See Also
        --------
        add_edge : Add a single edge

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> G = MarkovModel()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from([('Alice', 'Bob'), ('Bob', 'Charles')])
        """
        for edge in ebunch:
            self.add_edge(*edge)

    # def set_boolean_states(self, nodes):
    #     """
    #     Adds states to the node.
    #
    #     Parameters
    #     ----------
    #     states_dic :  Dictionary ({node : [state1, state2]})
    #             Dictionary of nodes to their list of states
    #
    #     See Also
    #     --------
    #     get_states
    #
    #     Examples
    #     --------
    #     >>> from pgmpy.models import MarkovModel
    #     >>> G = MarkovModel([('diff', 'intel'), ('diff', 'grade'),
    #     >>>                       ('intel', 'sat')])
    #     >>> G.set_boolean_states(['diff','intel'])
    #
    #     """
    #     if isinstance(nodes, str):
    #         nodes = [nodes]
    #     states_dict = {}
    #     for node in nodes:
    #         states_dict[node] = [0, 1]
    #     self.set_states(states_dict)
    #
    # def add_states(self, states_dic):
    #     """
    #     Adds states to the node.
    #
    #     Parameters
    #     ----------
    #     states_dic:  Dictionary ({node : [state1, state2]})
    #             Dictionary of nodes to their list of states
    #
    #     See Also
    #     --------
    #     get_states
    #
    #     Examples
    #     --------
    #     >>> from pgmpy.models import MarkovModel
    #     >>> G = MarkovModel([('diff', 'intel'), ('diff', 'grade'),
    #     >>>                       ('intel', 'sat')])
    #     >>> G.add_states({'diff': ['easy', 'hard'],
    #     ...               'intel': ['dumb', 'smart']})
    #     """
    #     for node, states in states_dic.items():
    #         self.node[node]['_states'].extend(states)
    #
    # def set_states(self, states_dic):
    #     """
    #     Adds states to the node.
    #
    #     Parameters
    #     ----------
    #     states_dic :  Dictionary ({node : [state1, state2]})
    #             Dictionary of nodes to their list of states
    #
    #     See Also
    #     --------
    #     get_states
    #
    #     Examples
    #     --------
    #     >>> from pgmpy.models import MarkovModel
    #     >>> G = MarkovModel([('diff', 'intel'), ('diff', 'grade'),
    #     >>>                       ('intel', 'sat')])
    #     >>> G.set_states({'diff': ['easy', 'hard'],
    #     ...               'intel': ['dumb', 'smart']})
    #     """
    #     for node, states in states_dic.items():
    #         self.node[node]['_states'] = states
    #
    # def remove_all_states(self, node):
    #     """
    #     Remove all the states of a node
    #
    #     Parameters
    #     ----------
    #     node :  The node for which the states have to be removed
    #
    #     See Also
    #     --------
    #     add_states
    #
    #     Examples
    #     --------
    #     >>> from pgmpy.models import MarkovModel
    #     >>> G = MarkovModel([('diff', 'intel'), ('diff', 'grade'),
    #     ...            ('intel', 'sat')])
    #     >>> G.add_states({'diff': ['easy', 'hard'],
    #     ...    'intel': ['dumb', 'smart']})
    #     >>> G.get_states('diff')
    #     ['easy', 'hard']
    #     >>> G.remove_all_states('diff')
    #     >>> G.get_states('diff')
    #     []
    #     """
    #     self.node[node]["_states"] = []
    #
    # def get_states(self, node):
    #     """
    #     Returns a generator object with states in user-defined order
    #
    #     Parameters
    #     ----------
    #     node  :   node
    #             Graph Node. Must be already present in the Model.
    #
    #     See Also
    #     --------
    #     set_states
    #
    #     Examples
    #     --------
    #     >>> from pgmpy.models import MarkovModel
    #     >>> G = MarkovModel([('diff', 'intel'), ('diff', 'grade'),
    #     ...            ('intel', 'sat')])
    #     >>> G.add_states({'diff': ['easy', 'hard'],
    #     ...    'intel': ['dumb', 'smart']})
    #     >>> G.get_states('diff')
    #     ['easy', 'hard']
    #     """
    #
    #     return self.node[node]['_states']
    #
    # def number_of_states(self, node):
    #     """
    #     Returns the number of states of node.
    #
    #     Parameters
    #     ----------
    #     node  :  Graph Node
    #
    #     See Also
    #     --------
    #     set_states
    #     get_states
    #
    #     Examples
    #     --------
    #     >>> from pgmpy.models import MarkovModel
    #     >>> G = MarkovModel([('diff', 'grade'), ('intel', 'grade'),
    #     ...                       ('intel', 'SAT')])
    #     >>> G.set_states({'diff': ['easy', 'hard']})
    #     >>> G.number_of_states('diff')
    #     2
    #     """
    #
    #     return len(self.get_states(node))
    #
    # def get_rule_for_states(self, node):
    #     """
    #     Check the order of states in which factor values are expected
    #
    #     Parameters
    #     ----------
    #     node   : graph node
    #             The node for which to check the rule.
    #
    #     See Also
    #     --------
    #     set_rule_for_states
    #
    #     Examples
    #     --------
    #     >>> from pgmpy.models import MarkovModel
    #     >>> G = MarkovModel([('diff', 'grade'), ('diff', 'intel')])
    #     >>> G.set_states({'diff': ['easy', 'hard'],
    #     ...               'intel': ['dumb', 'smart']})
    #     >>> G.get_rule_for_states('diff')
    #     ['easy', 'hard']
    #     """
    #     return self.get_states(node)
    #
    # def set_rule_for_states(self, node, states):
    #     """
    #     Change the order in which the CPD is expected
    #
    #     Parameters
    #     ----------
    #     node  :  Graph Node
    #             Node for which the order needs to be changed
    #
    #     states : List
    #             List of the states of node in the order in which
    #             CPD will be entered
    #
    #     See Also
    #     --------
    #     get_rule_for_states
    #
    #     Example
    #     -------
    #     >>> from pgmpy.models import MarkovModel
    #     >>> G = MarkovModel([('diff', 'grade'), ('diff', 'intel')])
    #     >>> G.set_states({'diff': ['easy', 'hard'],
    #     ...               'intel': ['dumb', 'smart']})
    #     >>> G.get_rule_for_states('diff')
    #     ['easy', 'hard']
    #     >>> G.set_rule_for_states('diff', ['hard', 'easy'])
    #     >>> G.get_rule_for_states('diff')
    #     ['hard', 'easy']
    #     """
    #     self.node[node]['_states'] = states

    def _set_is_observed(self, node, bool):
        """
        Updates '_observed' attribute of the node.

        If any of the states of a node are observed, node.['_observed']
        is made True. Otherwise, it is False.
        """
        self.node[node]['_is_observed'] = bool

    def is_observed(self, nodes):
        """
        Check if nodes are observed. If observed returns True.

        Returns single boolean value or a list of boolean values
        for each node in nodes.

        Parameters
        ----------
        nodes: single node or list of nodes.
            Nodes whose observed status needs to be returned.

        See Also
        --------
        set_observations
        reset_observations

        Example
        -------
        >>> from pgmpy.models import MarkovModel
        >>> student = MarkovModel()
        >>> student.add_node('grades')
        >>> student.set_observations({'grades': 'A'})
        >>> student.is_observed('grades')
        True
        """
        nodes = [nodes] if not isinstance(nodes, list) else nodes
        return [self.node[node]['_is_observed'] for node in nodes]

    def set_observation(self, node, state):
        """
        Sets state of node as observed.

        Parameters
        ----------
        node : str
            The node for which the observation is set.
        state: int
            The index of the state which is to be set to observed.

        See Also
        --------
        unset_observation
        is_observed

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors import Factor
        >>> G = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
        ...                  ('Charles', 'Debbie'), ('Debbie', 'Alice')])
        >>> G.add_factors(Factor(['Alice', 'Bob'], [2, 2], [30, 5, 1, 10]),
        ...               Factor(['Bob', 'Charles'], [2, 2], [100, 1, 1, 100]),
        ...               Factor(['Charles', 'Debbie'], [2, 2], [1, 100, 100, 1]))
        >>> G.set_observations('Alice', 0)
        """
        self._set_is_observed(node, True)
        # TODO: Raise ValueError if the state is greater than the cardinality
        # of the variable.
        self.node[node]['observed'] = state

    def set_observations(self, observations):
        """
        Sets state of node as observed.

        Parameters
        ----------
        observations : dict
            A dictionary of the form of {node1: state1,
            node2: state1, ..} or {node: state} containing all the
            nodes to be observed.

        See Also
        --------
        unset_observations
        is_observed

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors import Factor
        >>> G = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
        ...                  ('Charles', 'Debbie'), ('Debbie', 'Alice')])
        >>> G.add_factors(Factor(['Alice', 'Bob'], [2, 2], [30, 5, 1, 10]),
        ...               Factor(['Bob', 'Charles'], [2, 2], [100, 1, 1, 100]),
        ...               Factor(['Charles', 'Debbie'], [2, 2], [1, 100, 100, 1]))
        >>> G.set_observations({'Alice': 0, 'Charles': 1})
        """
        for node, state in observations.items():
            self.set_observation(node, state)

    def get_observations(self, nodes):
        """
        Returns the observation of the given nodes.

        Returns a dict of {node: state_observed}.

        Parameters
        ----------
        nodes: node, list of nodes
            The nodes for which the observation is to returned

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors import Factor
        >>> G = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
        ...                  ('Charles', 'Debbie'), ('Debbie', 'Alice')])
        >>> G.add_factors(Factor(['Alice', 'Bob'], [2, 2], [30, 5, 1, 10]),
        ...               Factor(['Bob', 'Charles'], [2, 2], [100, 1, 1, 100]),
        ...               Factor(['Charles', 'Debbie'], [2, 2], [1, 100, 100, 1]))
        >>> G.set_observations({'Alice': 0, 'Charles': 1})
        >>> G.get_observations('Alice')
        >>> {'Alice': 0}
        >>> G.get_observations(['Alice', 'Charles'])
        """
        nodes = [nodes] if not isinstance(nodes, list) else nodes
        return {node: self.node[node]['observed'] for node in nodes}

    def _get_observed_list(self):
        """
        Returns a list of all observed nodes
        """
        return [node for node in self.nodes()
                if self.node[node]['_is_observed']]

    def unset_observations(self, nodes):
        """
        Unset the observation for the node.

        Parameters
        -----------
        nodes: node, list of nodes
            The node for which the observation has to be unset

        Example
        -------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors import Factor
        >>> G = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
        ...                  ('Charles', 'Debbie'), ('Debbie', 'Alice')])
        >>> G.add_factors(Factor(['Alice', 'Bob'], [2, 2], [30, 5, 1, 10]),
        ...               Factor(['Bob', 'Charles'], [2, 2], [100, 1, 1, 100]),
        ...               Factor(['Charles', 'Debbie'], [2, 2], [1, 100, 100, 1]))
        >>> G.set_observations({'Alice': 0, 'Charles': 1})
        >>> G.unset_observation('Alice')
        """
        nodes = [nodes] if not isinstance(nodes, list) else nodes
        for node in nodes:
            self._set_is_observed(node, False)
            self.node[node]['observed'] = None

    def add_factors(self, *factors):
        """
        Associate a factor to the graph.
        See factors class for the order of potential values

        Parameters
        ----------
        *factor: pgmpy.factors.factors object
            A factor object on any subset of the variables of the model which
            is to be associated with the model.

        Returns
        -------
        None

        See Also
        --------
        get_factors

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors import Factor
        >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
        ...                        ('Charles', 'Debbie'), ('Debbie', 'Alice')])
        >>> factor = Factor(['Alice', 'Bob'], cardinality=[3, 2], np.random.rand(6))
        >>> student.add_factors(factor)
        """
        for factor in factors:
            if set(factor.variables) - set(factor.variables).intersection(set(self.nodes())):
                raise ValueError("factors defined on variable that is not in the model", factor)

            self.factors.append(factor)
            for variable_index in factor.variables:
                self.cardinality[factor.variables[variable_index]] = factor.cardinality[variable_index]

    def get_factors(self):
        """
        Returns the factors that have been added till now to the graph

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors import Factor
        >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> factor = Factor(['Alice', 'Bob'], cardinality=[2, 2], np.random.rand(6))
        >>> student.add_factors(factor)
        >>> student.get_factors()
        """
        return self.factors

    def check_clique(self, nodes):
        """
        Check if the given nodes form a clique.

        Parameters
        ----------
        nodes: list of nodes.
            Nodes to check if they are a part of any clique.
        """
        for node1, node2 in itertools.combinations(nodes, 2):
            if not self.has_edge(node1, node2):
                return False
        return True

    def triangulate(self, heuristic='H6', order=None, inplace=False):
        """
        Triangulate the graph.

        If order of deletion is given heuristic algorithm will not be used.

        Parameters
        ----------
        heuristic: H1 | H2 | H3 | H4 | H5 | H6
            The heuristic algorithm to use to decide the deletion order of
            the variables to compute the triangulated graph.

            Let X be the set of variables and X(i) denotes the i-th variable.

            S(i): The size of the clique created by deleting the variable.
            E(i): Cardinality of variable X(i).
            M(i): The maximum size of the cliques of the subgraph given by
                    X(i) and its adjacent nodes.
            C(i): The sum of the size of cliques of the subgraph given by X(i)
                    and its adjacent nodes.

            The heuristic algorithm decide the deletion order if this way:

            H1: Delete the variable with minimal S(i).
            H2: Delete the variable with minimal S(i)/E(i).
            H3: Delete the variable with minimal S(i) - M(i).
            H4: Delete the variable with minimal S(i) - C(i).
            H5: Delete the variable with minimal S(i)/M(i).
            H6: Delete the variable with minimal S(i)/C(i).

        order: list, tuple (array-like)
            The order of deletion of the variables to compute the triagulated
            graph. If order is given heuristic algorithm will not be used.

        inplace: True | False
            if inplace is true then adds the edges to the object from
            which it is called else returns a new object.

        Reference
        ---------
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.3607
        """
        graph_copy = nx.Graph(self.edges())
        edge_set = set()
        if not order:
            order = []
            for index in range(self.number_of_nodes()):
                for node in graph_copy.nodes():
                    S = {}
                    graph_working_copy = nx.Graph(graph_copy.edges())
                    graph_working_copy.add_edges_from(itertools.combinations(graph_working_copy.neighbors(node), 2))
                    graph_working_copy.remove_node(node)
                    clique_dict = nx.cliques_containing_node(graph_working_copy, nodes=graph_copy.neighbours(node))

                    def _common_list(*lists):
                        common = [sorted(li) for li in lists[0]]
                        for i in range(1, len(lists)):
                            list1 = [sorted(li) for li in lists[i]]
                            for list2 in common:
                                if list2 not in list1:
                                    common.remove(list2)
                        return common

                    S[node] = _common_list(*list(clique_dict.values()))

                if heuristic == 'H1':
                    node_to_delete = min(S, key=S.get)

                elif heuristic == 'H2':
                    S_by_E = {S[key]/self.cardinality[key] for key in S}
                    node_to_delete = min(S_by_E, key=S_by_E.get)

                elif heuristic in ('H3', 'H5'):
                    M = {}
                    for node in graph_copy.nodes():
                        graph_working_copy = nx.Graph(graph_copy.edges())
                        neighbors = graph_working_copy.neighbors(node)
                        graph_working_copy.add_edges_from(itertools.combinations(neighbors, 2))
                        graph_working_copy.remove_node(node)
                        cliques = nx.cliques_containing_node(graph_working_copy, nodes=neighbors)

                        common_clique = list(cliques.values())[0]
                        for values in cliques.values():
                            common_clique = [value for value in common_clique if value in values]

                        M[node] = np.prod([self.cardinality[node] for node in common_clique[0]])

                    if heuristic == 'H3':
                        S_minus_M = {S[key] - M[key] for key in S}
                        node_to_delete = min(S_minus_M, key=S_minus_M.get)

                    else:
                        S_by_M = {S[key]/M[key] for key in S}
                        node_to_delete = min(S_by_M, key=S_by_M.get)

                else:
                    C = {}
                    for node in graph_copy.nodes():
                        graph_working_copy = nx.Graph(graph_copy.edges())
                        neighbors = graph_working_copy.neighbors(node)
                        graph_working_copy.add_edges_from(itertools.combinations(neighbors, 2))
                        graph_working_copy.remove_node(node)
                        cliques = nx.cliques_containing_node(graph_working_copy, nodes=neighbors)

                        common_clique = list(cliques.values())[0]
                        for values in cliques.values():
                            common_clique = [value for value in common_clique if value in values]

                        clique_size_sum = 0
                        for r in range(1, len(common_clique)+1):
                            for clique in itertools.combinations(common_clique, r):
                                clique_size_sum += np.prod([self.cardinality[node] for node in clique])

                        C[node] = clique_size_sum

                    if heuristic == 'H4':
                        S_minus_C = {S[key] - C[key] for key in S}
                        node_to_delete = min(S_minus_C, key=S_minus_C.get)

                    else:
                        S_by_C = {S[key]/C[key] for key in S}
                        node_to_delete = min(S_by_C, key=S_by_C.get)

                order.append(node_to_delete)

        graph_copy = nx.Graph(self.edges())
        for node in order:
            for edge in itertools.combinations(graph_copy.neighbors(node), 2):
                graph_copy.add_edge(edge[0], edge[1])
                edge_set.add(edge)
            graph_copy.remove_node(node)

        if inplace:
            for edge in edge_set:
                self.add_edge(edge[0], edge[1])
            return self

        else:
            graph_copy = nx.copy(self)
            for edge in edge_set:
                self.add_edge(edge[0], edge[1])
            return graph_copy

    def _norm_h(self, pos, node_list, value_list):
        """
        Helper function for get_normaliztion_constant_brute_force
        Helping in recursion
        """
        if pos == self.number_of_nodes():
            val = 1
            assignment_dict = {}
            for i in range(0, self.number_of_nodes()):
                assignment_dict[node_list[i]] = value_list[i]
            for factor in self._factors:
                val *= factor.get_value(assignment_dict)
            #print(str(assignment_dict) + str(val))
            return val
        else:
            val = 0
            for i in range(0, len(self.node[node_list[pos]]['_states'])):
                value_list[pos] = i
                val += self._norm_h(pos + 1, node_list, value_list)
            return val

    def normalization_constant_brute_force(self):
        """
        Get the normalization constant using brute force technique

        Parameters
        ----------

        See Also
        --------
        _norm_h

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> student = MarkovModel()
        >>> student.add_nodes_from(['diff', 'intel'])
        >>> student.set_states({'diff': ['hard', 'easy']})
        >>> student.set_states({'intel': ['avg', 'dumb', 'smart']})
        >>> student.add_edge('diff','intel')
        >>> factor = student.add_factors(['diff','intel'], [0.1,0.1,0.1,0.1,0.1,0.1])
        >>> print(student.normalization_constant_brute_force())
        0.6000000000000001
        """
        value_list = [0] * self.number_of_nodes()
        val = self._norm_h(0, self.nodes(), value_list)
        return val

    def make_jt(self, triangulation_technique):
        """
        Makes the junction tree for the MarkovModel

        Parameter
        ---------
        triangulation_technique : int
            Index of the triangulation technique to be used
            See jt_techniques in Undirected Graph for documentation on
            the triangulation techniques and the technique_num for each
            technique

        Example
        -------
        >>> from pgmpy.models import MarkovModel
        >>> student = MarkovModel([('diff', 'intel'), ('diff', 'grade'),
        ...                        ('intel', 'grade')])
        >>> student.add_states({'diff': ['easy', 'hard'],
        ...                     'intel': ['dumb', 'smart'],
        ...                     'grade': ['A','B','C']})
        >>> factor = student.add_factors(['diff','intel'], range(4))
        >>> factor2 = student.add_factors(['intel','grade'], range(6))
        >>> factor3 = student.add_factors(['diff','grade'], range(6))
        >>> jt = student.make_jt(2)
        >>> jt.print_graph("Printing the Junction Tree")
        Printing the graph Printing the Junction Tree<<<
        1	( {'factors': [diff	intel	phi(diff, intel)
        diff_0	intel_0	0.0
        diff_0	intel_1	1.0
        diff_1	intel_0	2.0
        diff_1	intel_1	3.0
        , intel	grade	phi(intel, grade)
        intel_0	grade_0	0.0
        intel_0	grade_1	1.0
        intel_0	grade_2	2.0
        intel_1	grade_0	3.0
        intel_1	grade_1	4.0
        intel_1	grade_2	5.0
        , diff	grade	phi(diff, grade)
        diff_0	grade_0	0.0
        diff_0	grade_1	1.0
        diff_0	grade_2	2.0
        diff_1	grade_0	3.0
        diff_1	grade_1	4.0
        diff_1	grade_2	5.0
        ], 'clique_nodes': ['diff', 'grade', 'intel']} ) : []
        >>>
        """
        jt = UndirectedGraph.make_jt(self, triangulation_technique)
        #jt.print_graph("print junction tree before adding factors ")
        #jt.print_graph("after making the junction tree")
        jt.insert_factors(self.get_factors())
        return jt

    def induced_graph(self, order=None):
        """
        Returns the induced graph resulting from the given variable
        elimination order.

        Parameters
        ----------
        order: list, tuple (array-like)
            The order in which the variables are to be eliminated.
            If order not specified removes variables in a random way.
        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> mm = MarkovModel()
        >>> mm.add_edges_from([('Coherence', 'Difficulty'),
        ...                    ('Difficulty', 'Grade'),
        ...                    ('Grade', 'Happy'),
        ...                    ('Grade', 'Letter'),
        ...                    ('Letter', 'Job'),
        ...                    ('SAT', 'Job'),
        ...                    ('Intelligence', 'SAT'),
        ...                    ('Intelligence', 'Grade'),
        ...                    ('Job', 'Happy')])
        >>> mm.induced_graph(order=['Coherence', 'Difficulty', 'Intelligence', 'Happy', 'Grade',
        ...                         'SAT', 'Letter'])
        """
        from itertools import chain, combinations
        graph_copy = self.copy()
        edges_to_add = set()
        for variable in order:
            var_factors = [factor for factor in graph_copy.factors if variable in factor.scope]
            scope_set = {chain(*[factor.scope for factor in var_factors])}
            other_variables = scope_set.remove(variable)
            for neighbor in graph_copy.neighbors(variable):
                graph_copy.remove_edge((variable, neighbor))
            graph_copy.remove(variable)
            for edge in combinations(other_variables, 2):
                edges_to_add.add(edge)
            graph_copy.add_edges_from(edges_to_add)
        return self.copy().add_edges_from(edges_to_add)
