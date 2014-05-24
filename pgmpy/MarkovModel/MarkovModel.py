#!/usr/bin/env python3

import itertools

import networkx as nx

from .Exceptions import ObservationNotFound
from .Exceptions import FactorNodesNotInClique
from .Factor import *


class MarkovModel(nx.Graph):
    """
        Base class for markov model.

        A BayesianModel stores nodes and edges with potentials

        BayesianModel hold undirected edges.

        IMP NOTE : Nodes should be strings.

        Parameters
        ----------
        data : input graph
            Data to initialize graph.  If data=None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.

        Examples
        --------
        Create an empty bayesian model with no nodes and no edges.

        >>> from pgmpy import MarkovModel as mm
        >>> G = mm.BayesianModel()

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
        set_states({node : [state1, state2]})
        get_states('node1')
        number_of_state('node1')
        get_rule_for_states('node1')
        set_rule_for_states({'node1': ['state2', 'state1', ...]})
        get_rule_for_parents('node1')
        set_rule_for_parents({'node1': ['parent1', 'parent2', ...]})
        get_parents('node1')
        number_of_parents('node1')
        set_cpd('node1', cpd1)
        get_cpd('node1')
        set_observations({'node1': ['observed_state1', 'observed_state2'],
                          'node2': 'observed_state1'})
        reset_observations(['node1', 'node2'])
        is_observed('node1')
        active_trail_nodes('node1')
        is_active_trail('node1', 'node2')
        marginal_probability('node1')
        """
    def __init__(self, ebunch=None):
        if ebunch is not None:
            nodes = set(itertools.chain(*ebunch))
            self._check_node_string(nodes)
        nx.Graph.__init__(self, ebunch)
        if ebunch is not None:
            for node in nodes:
                self.setIsObserved(node, False)
        self._factors = []

    def add_node(self, node):
        # if not isinstance(node, str):
        #     raise TypeError("Name of nodes must be strings")
        """
        Add a single node to the Graph.

        Parameters
        ----------
        node: node
              A node can only be a string.

        See Also
        --------
        add_nodes_from : add a collection of nodes

        Examples
        --------
        >>> from pgmpy import MarkovModel as mm
        >>> G =mm.BayesianModel()
        >>> G.add_node('difficulty')
        """
        self._check_node_string([node])
        nx.Graph.add_node(self, node)
        self.setIsObserved(node, False)


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
        >>> G = mm.BayesianModel()
        >>> G.add_nodes_from(['diff', 'intel', 'grade'])
        """
        for node in nodes:
            self.add_node(node)
        #self._update_node_parents(nodes)

    def add_edge(self, u, v):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u,v : nodes
              Nodes must be strings.

        See Also
        --------
        add_edges_from : add a collection of edges

        EXAMPLE
        -------
        >>> from pgmpy import BayesianModel as bm/home/abinash/software_packages/numpy-1.7.1
        >>> G = bm.BayesianModel()
        >>> G.add_nodes_from(['grade', 'intel'])
        >>> G.add_edge('grade', 'intel')
        """
        #string check required because if nodes not present networkx
        #automatically adds those nodes
        if u not in self.nodes():
            self.add_node(u)
        if v not in self.nodes():
            self.add_node(v)
        nx.Graph.add_edge(self, u, v)

        #self._update_node_parents([u, v])

    def add_edges_from(self, ebunch, attr_dict=None, **attr):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names should be strings.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u,v).

        See Also
        --------
        add_edge : Add a single edge

        Examples
        --------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel()
        >>> G.add_nodes_from(['diff', 'intel', 'grade'])
        >>> G.add_edges_from([('diff', 'intel'), ('grade', 'intel')])
        """
        for edge in ebunch:
            self.add_edge(*edge)

    def set_boolean_states(self, nodes):
        """
        Adds states to the node.

        Parameters
        ----------
        states_dic :  Dictionary ({node : [state1, state2]})
                Dictionary of nodes to their list of states

        See Also
        --------
        get_states

        Examples
        --------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'intel'), ('diff', 'grade'),
        >>>                       ('intel', 'sat')])
        >>> G.set_states({'diff': ['easy', 'hard'],
        ...               'intel': ['dumb', 'smart']})
        """
        if isinstance(nodes, str):
            nodes = [nodes]
        states_dict={}
        for node in nodes:
            states_dict[node]=[0,1]
        self.set_states(states_dict)

    def set_states(self, states_dic):
        """
        Adds states to the node.

        Parameters
        ----------
        states_dic :  Dictionary ({node : [state1, state2]})
                Dictionary of nodes to their list of states

        See Also
        --------
        get_states

        Examples
        --------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'intel'), ('diff', 'grade'),
        >>>                       ('intel', 'sat')])
        >>> G.set_states({'diff': ['easy', 'hard'],
        ...               'intel': ['dumb', 'smart']})
        """
        for node, states in states_dic.items():
            if '_states' in self.node[node].keys():
                self.node[node]['_states'].extend(states)
            else:
                self.node[node]['_states'] = states
            self.setIsObserved(node,  False)

    def get_states(self, node):
        """
        Returns a generator object with states in user-defined order

        Parameters
        ----------
        node  :   node
                Graph Node. Must be already present in the Model.

        See Also
        --------
        set_states

        Examples
        --------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'intel'), ('diff', 'grade'),
        >>>                       ('intel', 'sat')])
        >>> G.set_states({'diff': ['easy', 'hard'],
        ...               'intel': ['dumb', 'smart']})
        >>> states = G.get_states('diff')
        >>> for state in states:
        ...     print(state)
        """
        return self.node[node]['_states']

    def number_of_states(self, node):
        """
        Returns the number of states of node.

        Parameters
        ----------
        node  :  Graph Node

        See Also
        --------
        set_states
        get_states

        Examples
        --------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                       ('intel', 'SAT')])
        >>> G.set_states({'diff': ['easy', 'hard']})
        >>> G.number_of_states('diff')
        3
        """
        return len(self.node[node]['_states'])


    @staticmethod
    def _check_node_string(node_list):
        """
        Checks if all the newly added node are strings.
        Called from __init__, add_node, add_nodes_from, add_edge and
        add_edges_from
        """
        for node in node_list:
            if not (isinstance(node, str)):
                raise TypeError("Node names must be strings")

    def get_rule_for_states(self, node):
        """
        Check the order in which the CPD is expected

        Parameters
        ----------
        node   : graph node
                The node for which to check the rule.

        See Also
        --------
        set_rule_for_states
        get_rule_for_parents
        set_rule_for_parents

        Examples
        --------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'intel'), ('diff', 'grade')])
        >>> G.set_states({'diff': ['easy', 'hard'], 'grade': ['C', 'A']})
        >>> G.get_rule_for_states('diff')
        >>> G.get_rule_for_states('grade')
        """
        return self.node[node]['_states']

    def set_rule_for_states(self, node, states):
        """
        Change the order in which the CPD is expected

        Parameters
        ----------
        node  :  Graph Node
                Node for which the order needs to be changed

        states : List
                List of the states of node in the order in which
                CPD will be entered

        See Also
        --------
        get_rule_for_states
        get_rule_for_parents
        set_rule_for_parents

        Example
        -------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'grade'), ('diff', 'intel')])
        >>> G.set_states({'diff': ['easy', 'hard'],
        ...               'intel': ['dumb', 'smart']})
        >>> G.get_rule_for_states('diff')
        ['easy', 'hard']
        >>> G.set_rule_for_states('diff', ['hard', 'easy'])
        >>> G.get_rule_for_states('diff')
        ['hard', 'easy']
        """
        self.node[node]['_states'] = states

    def get_neighbours(self, node):
        # TODO: Update docstrings
        """
        Returns a list of parents of node in order according to the rule
        set for parents.

        Parameters
        ----------
        node  :  Graph Node

        See Also
        --------
        get_rule_for_parents
        set_rule_for_parents

        Example
        -------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        >>>                       ('intel', 'SAT'), ('grade', 'reco')])
        >>> G.get_parents('grade')
        ['diff', 'intel']
        >>> G.set_rule_for_parents('grade', ['intel', 'diff'])
        >>> G.get_parents('grade')
        ['intel', 'diff']
        """
        return nx.Graph.neighbors(self, node)

    def number_of_neighbours(self, node):
        """
        Returns the number of parents of node

        node  :  Graph node

        Example
        -------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                       ('intel', 'sat')])
        >>> G.number_of_parents('grade')
        2
        """
        return len(self.get_neighbours(node))


    def set_potentials(self, nodes, potentials):
        """
        Add CPD (Conditional Probability Distribution) to the node.

        Parameters
        ----------
        node  :  Graph node
                Node to which CPD will be added

        cpd  :  2D list or tuple of the CPD. The 2D array should be
                according to the rule specified for parents and states.

        See Also
        --------
        get_cpd
        get_rule_for_parents
        set_rule_for_parents
        get_rule_for_states
        set_rule_for_states

        EXAMPLE
        -------
        >>> from pgmpy import BayesianModel as bm
        >>> student = bm.BayesianModel([('diff', 'grades'), ('intel', 'grades')])
        >>> student.set_states({'diff': ['easy', 'hard'],
        ...                    'intel': ['dumb', 'avg', 'smart'],
        ...                    'grades': ['A', 'B', 'C']})
        >>> student.set_rule_for_parents('grades', ('diff', 'intel'))
        >>> student.set_rule_for_states('grades', ('A', 'B', 'C'))
        >>> student.set_cpd('grades',
        ...             [[0.1,0.1,0.1,0.1,0.1,0.1],
        ...             [0.1,0.1,0.1,0.1,0.1,0.1],
        ...             [0.8,0.8,0.8,0.8,0.8,0.8]]
        ...             )

        +------+-----------------------+---------------------+
        |diff: |          easy         |         hard        |
        +------+------+------+---------+------+------+-------+
        |intel:| dumb |  avg |  smart  | dumb | avg  | smart |
        +------+------+------+---------+------+------+-------+
        |gradeA| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
        +------+------+------+---------+------+------+-------+
        |gradeB| 0.1  | 0.1  |   0.1   |  0.1 |  0.1 |   0.1 |
        +------+------+------+---------+------+------+-------+
        |gradeC| 0.8  | 0.8  |   0.8   |  0.8 |  0.8 |   0.8 |
        +------+------+------+---------+------+------+-------+
        """
        if not self.inClique(nodes):
            raise FactorNodesNotInClique(nodes)
        node_objects_list = []
        for node in nodes:
            node_objects_list.append((node,self.node[node]))
        factor = Factor(node_objects_list, potentials)
        self._factors.append(factor)
        return factor

    def inClique(self, nodes):
        for n1 in range(0,len(nodes)):
            for n2 in range(n1+1,len(nodes)):
                if not self.has_edge(nodes[n1], nodes[n2]):
                    return False
        return True

    def get_potentials(self, nodes, beautify=False):
        """
        Returns the CPD of the node.

        node  :  Graph node

        beautify : If set to True returns an ASCII art table of the CPD
                    with parents and states else if beautify=False, returns
                    a simple numpy.ndarray object of the CPD table

        See Also
        --------
        set_cpd
        """
        return self._factors

    def setIsObserved(self, node, tf):
        """
        Updates '_observed' attribute of the node.

        If any of the states of a node are observed, node.['_observed']
        is made True. Otherwise, it is False.
        """
        self.node[node]['_isObserved'] = tf

    def getIsObserved(self, node):
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
        >>> student = mm.BayesianModel()
        >>> student.add_node('grades')
        >>> student.set_states({'grades': ['A', 'B']})
        >>> student.set_observations({'grades': 'A'})
        >>> student.is_observed('grades')
        True
        """
        return self.node[node]['_isObserved']

    def unset_observation(self, node):
        self.setIsObserved(node, False)

    def unset_observations(self, node_list):
        for node in node_list:
            self.unset_observation(node)

    def set_observation(self, node, observation):
        self.setIsObserved(node,True)
        if observation in self.node[node]['_states']:
            self.node[node]['observed']=observation
        else:
            raise ObservationNotFound("Observation "+observation+" not found for "+node)

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
        reset_observations
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
            self.set_observation(node,state)

    def _get_observed_list(self):
        """
        Returns a list of all observed nodes
        """
        return [node for node in self.nodes()
                if self.node[node]['_isObserved']]
    def norm_h(self, pos, node_list, value_list):
        if pos == self.number_of_nodes():
            val = 1
            assignment_dict={}
            for i in range(0, self.number_of_nodes()):
                assignment_dict[node_list[i]]=value_list[i]
            for factor in self._factors:
                val *= factor.get_potential_index_assignment(assignment_dict)
            return val
        else:
            val = 0
            for i in range(0, len(self.node[node_list[pos]]['_states'])):
                value_list[pos]=i
                val += self.norm_h(pos+1, node_list, value_list)
            return val


    def get_normalization_constant_brute_force(self):
        """
        Get the normalization constant for all the factors

        Parameters
        ----------

        See Also
        --------
        norm_h

        Examples
        --------
        >>> from pgmpy import MarkovModel as mm
        >>> student = mm.MarkovModel()
        >>> student.add_nodes_from(['diff', 'intel'])
        >>> student.set_states({'diff': ['hard', 'easy']})
        >>> student.set_states({'intel': ['avg', 'dumb', 'smart']})
        >>> student.add_edge('diff','intel')
        >>> factor = student.set_potentials(['diff','intel'], [0.1,0.1,0.1,0.1,0.1,0.1])
        >>> print(student.get_normalization_constant_brute_force())
        0.6000000000000001
        """
        value_list = []
        for i in range(0, self.number_of_nodes()):
            value_list.append(0)
        val = self.norm_h(0, self.nodes(), value_list)
        return val