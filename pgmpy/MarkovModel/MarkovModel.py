#!/usr/bin/env python3

import itertools
import networkx as nx
from pgmpy.Factor.Factor import Factor
from pgmpy.MarkovModel.UndirectedGraph import UndirectedGraph


class MarkovModel(UndirectedGraph):
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

        >>> from pgmpy import MarkovModel
        >>> G = MarkovModel.MarkovModel()

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
        add_edge('node1')
        add_edges_from([('node1', 'node2'),('node3', 'node4')])
        add_states({node : [state1, state2]})
        get_states('node1')
        number_of_state('node1')
        set_observations({'node1': ['observed_state1', 'observed_state2'],
                          'node2': 'observed_state1'})
        is_observed('node1')
        """

    def __init__(self, ebunch=None):
        nx.Graph.__init__(self, ebunch)
        if ebunch:
            for node in self.nodes():
                self._set_is_observed(node, False)
        self.factors = []

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
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.MarkovModel()
        >>> G.add_node('A')
        """
        nx.Graph.add_node(node)
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
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.MarkovModel()
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
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.MarkovModel()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edge('Alice', 'Bob')
        """
        # Need to check that there is no self loop.
        if u != v:
            self.add_edge(self, u, v)

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
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.MarkovModel()
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
    #     >>> from pgmpy import MarkovModel as mm
    #     >>> G = mm.MarkovModel([('diff', 'intel'), ('diff', 'grade'),
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
    #     >>> from pgmpy import MarkovModel
    #     >>> G = MarkovModel.MarkovModel([('diff', 'intel'), ('diff', 'grade'),
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
    #     >>> from pgmpy import MarkovModel as mm
    #     >>> G = mm.MarkovModel([('diff', 'intel'), ('diff', 'grade'),
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
    #     >>> from pgmpy import MarkovModel as mm
    #     >>> G = mm.MarkovModel([('diff', 'intel'), ('diff', 'grade'),
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
    #     >>> from pgmpy import MarkovModel as mm
    #     >>> G = mm.MarkovModel([('diff', 'intel'), ('diff', 'grade'),
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
    #     >>> from pgmpy import MarkovModel as mm
    #     >>> G = mm.MarkovModel([('diff', 'grade'), ('intel', 'grade'),
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
    #     >>> from pgmpy import MarkovModel
    #     >>> G = MarkovModel.MarkovModel([('diff', 'grade'), ('diff', 'intel')])
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
    #     >>> from pgmpy import MarkovModel as mm
    #     >>> G = mm.MarkovModel([('diff', 'grade'), ('diff', 'intel')])
    #     >>> G.set_states({'diff': ['easy', 'hard'],
    #     ...               'intel': ['dumb', 'smart']})
    #     >>> G.get_rule_for_states('diff')
    #     ['easy', 'hard']
    #     >>> G.set_rule_for_states('diff', ['hard', 'easy'])
    #     >>> G.get_rule_for_states('diff')
    #     ['hard', 'easy']
    #     """
    #     self.node[node]['_states'] = states

    def _in_clique(self, nodes):
        """
        Check if the nodes belong to a singhle click in the graph
        """
        for n1 in range(0, len(nodes)):
            for n2 in range(n1 + 1, len(nodes)):
                if not self.has_edge(nodes[n1], nodes[n2]):
                    return False
        return True

    def _set_is_observed(self, node, tf):
        """
        Updates '_observed' attribute of the node.

        If any of the states of a node are observed, node.['_observed']
        is made True. Otherwise, it is False.
        """
        self.node[node]['_is_observed'] = tf

    def is_observed(self, node):
        """
        Check if node is observed. If observed returns True.

        Parameters
        ----------
        node  :  Graph Node

        See Also
        --------
        set_observations
        reset_observations

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> student = mm.MarkovModel()
        >>> student.add_node('grades')
        >>> student.set_states({'grades': ['A', 'B']})
        >>> student.set_observations({'grades': 'A'})
        >>> student.is_observed('grades')
        True
        """
        return self.node[node]['_is_observed']

    def set_observation(self, node, observation):
        """
        Sets state of node as observed.

        Parameters
        ----------
        node : str
            The node for which the observation is set
        observation:str
            The state to which the node is set

        See Also
        --------
        unset_observation
        is_observed

        Examples
        --------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.MarkovModel([('diff', 'grade'), ('intel', 'grade'),
        ...                       ('grade', 'reco'))])
        >>> G.set_state({'diff': ['easy', 'hard']})
        >>> G.set_observation('diff', 'easy')
        """
        self._set_is_observed(node, True)
        if observation in self.node[node]['_states']:
            self.node[node]['observed'] = observation
        else:
            raise ValueError("Observation " + observation + " not found for " + node)

    def set_observations(self, observations):
        """
        Sets state of node as observed.

        Parameters
        ----------
        observations : dict
            A dictionary of the form of {node : [states...]} or
            {node: state} containing all the nodes to be observed.

        See Also
        --------
        unset_observations
        is_observed

        Examples
        --------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.MarkovModel([('diff', 'grade'), ('intel', 'grade'),
        ...                       ('grade', 'reco'))])
        >>> G.set_states({'diff': ['easy', 'hard']})
        >>> G.set_observations({'diff': 'easy'})
        """
        for node, state in observations.items():
            self.set_observation(node, state)

    def get_observation(self, node):
        """
        Returns the observation of the given node

        Parameters
        ----------
        node: str
            The node for which the observation is to returned
        Examples
        --------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.MarkovModel([('diff', 'grade'), ('intel', 'grade'),
        ...                       ('grade', 'reco'))])
        >>> G.set_state({'diff': ['easy', 'hard']})
        >>> G.set_observation('diff', 'easy')
        >>> G.get_set_observation('diff')
        'easy'
        """
        if self.is_observed(node):
            return self.node[node]['observed']
        else:
            raise ValueError('Observation has not been set up for the node ' + node)

    def _get_observed_list(self):
        """
        Returns a list of all observed nodes
        """
        return [node for node in self.nodes()
                if self.node[node]['_is_observed']]

    def unset_observation(self, node):
        """
        Unset the observation for the node

        Parameters
        -----------
        node: The node for which the observation has to be unset

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.MarkovModel([('d', 'g'), ('i', 'g'), ('g', 'l'),
        ...                            ('i', 's')])
        >>> G.add_states({'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'],
        ...               'i': ['dumb', 'smart'], 's': ['bad', 'avg', 'good'],
        ...               'l': ['yes', 'no']})
        >>> G.set_observations({'d': 'easy', 'g': 'A'})
        >>> G.unset_observation('d')
        """
        self._set_is_observed(node, False)
        self.node[node]['observed'] = ''

    def unset_observations(self, node_list):
        """
        Unsets the observation for all the nodes in the node_list

        Parameters
        ----------
        node_list : List of nodes

        Examples
        --------
        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.MarkovModel([('d', 'g'), ('i', 'g'), ('g', 'l'),
        ...                            ('i', 's')])
        >>> G.add_states({'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'],
        ...               'i': ['dumb', 'smart'], 's': ['bad', 'avg', 'good'],
        ...               'l': ['yes', 'no']})
        >>> G.set_observations({'d': 'easy', 'g': 'A'})
        >>> G.unset_observation('d')
        """
        for node in node_list:
            self.unset_observation(node)

    def add_factor(self, *factors):
        """
        Associate a factor to the graph.
        See Factor class for the order of potential values

        Parameters
        ----------
        *factor: pgmpy.Factor.Factor object
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
        >>> from pgmpy import MarkovModel as mm
        >>> from pgmpy.Factor import Factor
        >>> student = mm.MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
        >>>                           ('Charles', 'Debbie'), ('Debbie', 'Alice')])
        >>> factor = Factor(['Alice', 'Bob'], cardinality=[3, 2], np.random.rand(6))
        >>> student.add_factor(factor)
        """
        for factor in factors:
            if set(factor.variables) - set(factor.variables).intersection(set(self.nodes())):
                raise ValueError("Factor defined on variable that is not in the model", factor)

            self.factors.append(factor)

    def get_factors(self):
        """
        Returns the factors that have been added till now to the graph

        Examples
        --------
        >>> from pgmpy import MarkovModel as mm
        >>> student = mm.MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> factor = Factor(['Alice', 'Bob'], cardinality=[2, 2], np.random.rand(6))
        >>> student.add_factor(factor)
        >>> student.get_factors()
        """
        return self.factors

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
        >>> from pgmpy import MarkovModel as mm
        >>> student = mm.MarkovModel()
        >>> student.add_nodes_from(['diff', 'intel'])
        >>> student.set_states({'diff': ['hard', 'easy']})
        >>> student.set_states({'intel': ['avg', 'dumb', 'smart']})
        >>> student.add_edge('diff','intel')
        >>> factor = student.add_factor(['diff','intel'], [0.1,0.1,0.1,0.1,0.1,0.1])
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
        >>> from pgmpy import MarkovModel as mm
        >>> student = mm.MarkovModel([('diff', 'intel'), ('diff', 'grade'),('intel','grade')])
        >>> student.add_states({'diff': ['easy', 'hard'],
        ...             'intel': ['dumb', 'smart'],
        ...             'grade': ['A','B','C']})
        >>> factor = student.add_factor(['diff','intel'], range(4))
        >>> factor2 = student.add_factor(['intel','grade'], range(6))
        >>> factor3 = student.add_factor(['diff','grade'], range(6))
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


