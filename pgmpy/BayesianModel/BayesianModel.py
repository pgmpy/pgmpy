#!/usr/bin/env python3

import networkx as nx
import numpy as np
import itertools
from scipy import sparse
from pgmpy import Exceptions
from pgmpy.Factor import CPD


class BayesianModel(nx.DiGraph):
    def __init__(self, ebunch=None):
        """
        Base class for bayesian model.

        A BayesianModel stores nodes and edges with conditional probability
        distribution (cpd) and other attributes.

        BayesianModel hold directed edges.  Self loops are not allowed neither
        multiple (parallel) edges.

        Nodes should be strings.

        Edges are represented as links between nodes.

        Parameters
        ----------
        data : input graph
            Data to initialize graph.  If data=None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object.

        See Also
        --------

        Examples
        --------
        Create an empty bayesian model with no nodes and no edges.
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel()

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
        if ebunch is not None:
            self._check_node_string(set(itertools.chain(*ebunch)))

        nx.DiGraph.__init__(self, ebunch)

        if ebunch is not None:
            new_nodes = set(itertools.chain(*ebunch))
            self._check_graph(delete_graph=True)
            self._update_node_parents(new_nodes)
            self._update_node_rule_for_parents(new_nodes)

    def add_node(self, node):
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
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel()
        >>> G.add_node('difficulty')
        """
        # if not isinstance(node, str):
        #     raise TypeError("Name of nodes must be strings")
        self._check_node_string([node])
        nx.DiGraph.add_node(self, node)

        self._update_node_parents([node])
        self._update_node_rule_for_parents([node])

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
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel()
        >>> G.add_nodes_from(['diff', 'intel', 'grade'])
        """
        self._check_node_string(nodes)
        nx.DiGraph.add_nodes_from(self, nodes)

        self._update_node_parents(nodes)
        self._update_node_rule_for_parents(nodes)

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
        self._check_node_string([u, v])
        if self._check_graph([(u, v)], delete_graph=False):
            nx.DiGraph.add_edge(self, u, v)

        self._update_node_parents([u, v])
        self._update_node_rule_for_parents([u, v])

    def add_edges_from(self, ebunch):
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
        if ebunch is not None:
            self._check_node_string(set(itertools.chain(*ebunch)))

        if self._check_graph(ebunch, delete_graph=False):
            nx.DiGraph.add_edges_from(self, ebunch)

        new_nodes = set(itertools.chain(*ebunch))
        self._update_node_parents(new_nodes)
        self._update_node_rule_for_parents(new_nodes)

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
            try:
                self.node[node]['_states'].extend([{'name': state,
                                                    'observed_status': False}
                                                   for state in states])
            except KeyError:
                self.node[node]['_states'] = [
                    {'name': state, 'observed_status': False}
                    for state in states]

            self._update_rule_for_states(node, len(self.node[node]['_states']))
        ################# For Reference ########################
        #   self.node[node]['_rule_for_states'] = [            #
        #       n for n in range(len(states))]                 #
        ########################################################
            self._update_node_observed_status(node)

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
        _list_states = (self.node[node]['_states'][index]['name']
                        for index in self.node[node]['_rule_for_states'])
        return _list_states

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

    def _update_node_parents(self, node_list):
        """
        Function to update each node's _parent attribute.
        This function is called when new node or new edge is added.
        """
        for node in node_list:
            self.node[node]['_parents'] = sorted(self.predecessors(node))

    def _update_node_rule_for_parents(self, node_list):
        """
        Function to update each node's _rule_for_parents attribute.
        This function is called when new node or new edge is added.
        """
        for node in node_list:
            self.node[node]['_rule_for_parents'] = \
                [index for index in range(len(self.predecessors(node)))]
        ################ Just for reference: Earlier Code ####################
        # for head_node in head:                                             #
        #     for tail_node in tail:                                         #
        #         self.add_edge(tail_node, head_node)                        #
        #     self.node[head_node]['_parents'] = self.predecessors(head_node)#
        #     self.node[head_node]['_rule_for_parents'] = [                  #
        #             index for index in range(len(tail))]                   #
        ######################################################################

    def _check_node_string(self, node_list):
        """
        Checks if all the newly added node are strings.
        Called from __init__, add_node, add_nodes_from, add_edge and
        add_edges_from
        """
        for node in node_list:
            if not (isinstance(node, str)):
                raise TypeError("Node names must be strings")

    def _check_graph(self, ebunch=None, delete_graph=False):
        """
        Checks for self loops and cycles in the graph.
        If finds any, reverts the graph to previous state or
        in case when called from __init__ deletes the graph.
        """
        if delete_graph:
            if ebunch is not None:
                for edge in ebunch:
                    if edge[0] == edge[1]:
                        del self
                        raise Exceptions.SelfLoopError("Self Loops are not allowed",
                                                       edge)

            simple_cycles = [loop for loop in nx.simple_cycles(self)]
            if simple_cycles:
                del self
                raise Exceptions.CycleError("Cycles are not allowed",
                                            simple_cycles)
            return True
        else:
            for edge in ebunch:
                if edge[0] == edge[1]:
                    raise Exceptions.SelfLoopError("Self loops are not "
                                                   "allowed", edge)

            import copy
            test_G = copy.deepcopy(self)
            nx.DiGraph.add_edges_from(test_G, ebunch)
            simple_cycles = [loop for loop in nx.simple_cycles(test_G)]
            if simple_cycles:
                del test_G
                raise Exceptions.CycleError("Cycles are not allowed",
                                            simple_cycles)
            return True

    def _update_rule_for_states(self, node, number_of_states):
        """
        Checks if rule_for_states is already present for the node.
        If not present simply sets it to [0, 1, 2 ... ]
        If present the just adds numbers for the new states in the previously
        existing rule.
        """
        try:
            present_rule = self.node[node]['_rule_for_states']
        except KeyError:
            present_rule = None

        if not present_rule:
            self.node[node]['_rule_for_states'] = \
                [n for n in range(number_of_states)]
        else:
            self.node[node]['_rule_for_states'].extend([
                index for index in range(len(present_rule), number_of_states)])

    def _update_node_observed_status(self, node):
        """
        Updates '_observed' attribute of the node.

        If any of the states of a node are observed, node.['_observed']
        is made True. Otherwise, it is False.
        """
        if any(state['observed_status'] for state in
               self.node[node]['_states']):
            self.node[node]['_observed'] = True
        else:
            self.node[node]['_observed'] = False

    def _all_states_present_in_list(self, node, states_list):
        """
        Checks if all the states of node are present in state_list.
        If present returns True else returns False.
        """
        if sorted(states_list) == sorted([state['name'] for state
                                          in self.node[node]['_states']]):
            return True
        else:
            return False

    def _no_missing_states(self, node, states):
        """
        Returns True if all the states of the node are present in the
        argument states.
        """
        node_states = [state['name'] for state in self.node[node]['_states']]
        if sorted(node_states) == sorted(states):
            return True
        else:
            raise Exceptions.MissingStatesError(set(node_states) - set(states))

    def _no_extra_states(self, node, states):
        """"
        Returns True if the argument states contains only the states
         present in Node.
        """
        node_states = [state['name'] for state in self.node[node]['_states']]
        extra_states = set(states) - set(node_states)
        if extra_states:
            raise Exceptions.ExtraStatesError(extra_states)
        else:
            return True

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
        current_rule = self.node[node]['_rule_for_states']
        return [self.node[node]['_states'][index]['name']
                for index in current_rule]

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
        if self._all_states_present_in_list(node, states):
            new_rule = []
            for user_given_state in states:
                for state in self.node[node]['_states']:
                    if state['name'] == user_given_state:
                        new_rule.append(
                            self.node[node]['_states'].index(state))
                        break

            self.node[node]['_rule_for_states'] = new_rule

    def _is_node_parents_equal_parents_list(self, node, parents_list):
        """
        Returns true if parents_list has exactly those elements that are
        node's parents.
        """
        if sorted(parents_list) == sorted(self.node[node]['_parents']):
            return True
        else:
            return False

    def _no_extra_parents(self, node, parents):
        """"
        Returns True if parents has no other element other than those
        present in node's _parents' list.
        """
        extra_parents = set(parents) - set(self.node[node]['_parents'])
        if extra_parents:
            raise Exceptions.ExtraParentsError(extra_parents)
        else:
            return True

    def _no_missing_parents(self, node, parents):
        """"
        Returns True if all parents of node are present in the
        argument parents.
        """
        missing_parents = set(self.node[node]['_parents']) - set(parents)
        if missing_parents:
            raise Exceptions.MissingParentsError(missing_parents)
        else:
            return True

    def get_rule_for_parents(self, node):
        """
        Gets the order of the parents

        Parameters
        ----------
        node  : Graph Node
                Node whose rule is needed

        See Also
        --------
        set_rule_for_parents
        get_rule_for_states
        set_rule_for_states

        Example
        -------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'grade'), ('intel', 'grade')])
        >>> G.get_rule_for_parents('grades')
        ['diff', 'intel']
        >>> G.set_rule_for_states({'grades': ['intel', 'diff']})
        >>> G.get_rule_for_parents('grades')
        ['intel', 'diff']
        """
        current_rule = self.node[node]['_rule_for_parents']
        return [self.node[node]['_parents'][index] for index in current_rule]

    def set_rule_for_parents(self, node, parents):
        """
        Set a new rule for the parents of the node.

        Parameters
        ----------
        node  :  Graph node
                Node for which new rule is to be set.
        parents: List
               A list of names of the parents of the node in the order
                in which the rule needs to be set.

        See Also
        --------
        get_rule_for_parents
        get_rule_for_states
        set_rule_for_states
        """
        if self._is_node_parents_equal_parents_list(node, parents):
            new_order = []
            for user_given_parent in parents:
                for index in range(len(self.node[node]['_parents'])):
                    if self.node[node]['_parents'][index] == user_given_parent:
                        new_order.append(index)
                        break

            self.node[node]['_rule_for_parents'] = new_order

    def get_parents(self, node):
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
        return iter(self.get_rule_for_parents(node))

    def number_of_parents(self, node):
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
        return len(self.node[node]['_parents'])

    def _get_parent_objects(self, node):
        """
        Returns a list of those node objects which are parents of
        the argument node.
        """
        return (self.node[parent] for parent in self.get_parents(node))

    def set_cpd(self, node, cpd):
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

        #diff:       easy                 hard
        #intel: dumb   avg   smart    dumb  avg   smart
        #gradeA: 0.1    0.1    0.1     0.1  0.1    0.1
        #gradeB: 0.1    0.1    0.1     0.1  0.1    0.1
        #gradeC: 0.8    0.8    0.8     0.8  0.8    0.8
        """
        evidence = self.get_rule_for_parents(node)[::-1]
        evidence_card = [self.number_of_states(parent) for parent in evidence]
        self.node[node]['_cpd'] = CPD.TabularCPD(node,
                                                 self.number_of_states(node),
                                                 cpd, evidence, evidence_card)

    def get_cpd(self, node, beautify=False):
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
        if beautify:
            #TODO: ASCII art table
            pass
        else:
            return self.node[node]['_cpd'].get_cpd()

    def _change_state_observed(self, node, state, reset):
        """
        Changes observed status of state of a node

        Parameters
        ----------
        node: Graph node

        state: string (node state)

        reset: bool
            changes observed_status to False if reset is True
            changes observed_status to True if reset is False
        """
        found = 0
        for _state in self.node[node]['_states']:
            if _state['name'] == state:
                _state['observed_status'] = not reset
                found = 1
        if found:
            self._update_node_observed_status(node)
        else:
            raise Exceptions.StateError("State: ", state,
                                        " not found for node: ", node)

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
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        ...                       ('grade', 'reco'))])
        >>> G.set_states({'diff': ['easy', 'hard']})
        >>> G.set_observations({'diff': ['easy', 'hard']})
        """
        for node, states in observations.items():
            if isinstance(states, (list, tuple, set)):
                for state in states:
                    self._change_state_observed(node, state, reset=False)
            else:
                self._change_state_observed(node, states, reset=False)

    def reset_observations(self, nodes=None):
        """
        Resets observed-status of given nodes. Will not change a particular
        state. For that use, add_observations with reset=True.

        If no arguments are given, all states of all nodes are reset.

        Parameters
        ----------
        node : Graph Node, dict
            list of nodes or node whose observed_status is to be reset
            dict of {node: [states...]} when observed_status of states
            is to be reset.

        Examples
        --------
        >>> from pgmpy import BayesianModel as bm
        >>> G = bm.BayesianModel([('diff', 'intel'), ('diff', 'grade')])
        >>> G.set_states({'diff': ['easy', 'hard'], 'grade': ['C', 'A']})
        >>> G.get_rule_for_states('diff')
        >>> G.get_rule_for_states('grade')
        >>> G.set_observations({'grade': ['A']})
        >>> G.reset_observations({'grade': 'A'})

        See Also
        --------
        set_observations
        is_observed
        """
        if isinstance(nodes, dict):
            for node, states in nodes.items():
                if isinstance(states, (list, tuple, set)):
                    for state in states:
                        self._change_state_observed(node, state, reset=True)
                else:
                    self._change_state_observed(node, states, reset=True)
        else:
            if nodes is None:
                nodes = self.nodes()
            if not isinstance(nodes, (list, tuple)):
                nodes = [nodes]
            try:
                for node in nodes:
                    for state in self.node[node]['_states']:
                        state['observed_status'] = False
                    self._update_node_observed_status(node)
            except KeyError:
                raise Exceptions.NodeNotFoundError("Node not found", node)

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
        >>> from pgmpy import BayesianModel as bm
        >>> student = bm.BayesianModel()
        >>> student.add_node('grades')
        >>> student.set_states({'grades': ['A', 'B']})
        >>> student.set_observations({'grades': 'A'})
        >>> student.is_observed('grades')
        True
        """
        return self.node[node]['_observed']

    def _get_ancestors_of(self, obs_nodes_list):
        """
        Returns a list of all ancestors of all the observed nodes.

        Parameters
        ----------
        obs_nodes_list: string, list-type
            name of all the observed nodes
        """
        if not isinstance(obs_nodes_list, (list, tuple)):
            obs_nodes_list = [obs_nodes_list]

        ancestors_list = set()
        nodes_list = set(obs_nodes_list)
        while nodes_list:
            node = nodes_list.pop()
            if node not in ancestors_list:
                nodes_list.update(self.predecessors(node))
            ancestors_list.add(node)
        return ancestors_list

    def _get_observed_list(self):
        """
        Returns a list of all observed nodes
        """
        return [node for node in self.nodes()
                if self.node[node]['_observed']]

    def active_trail_nodes(self, start):
        """
        Returns all the nodes reachable from start via an active trail

        Parameters
        ----------
        start: Graph node

        Examples
        -------
        >>> from pgmpy import BayesianModel as bm
        >>> student = bm.BayesianModel()
        >>> student.add_nodes_from(['diff', 'intel', 'grades'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades')])
        >>> student.set_states({'diff': ['easy', 'hard'],
        ...                     'intel': ['dumb', 'smart'],
        ...                     'grades': ['A', 'B', 'C']})
        >>> student.set_observations({'grades': 'A'})
        >>> student.active_trail_nodes('diff')
        ['diff', 'intel']

        See Also
        --------
        is_active_trail(start, end)

        -------------------------------------------------------------------
        Details of algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        -------------------------------------------------------------------

        """
        observed_list = self._get_observed_list()
        ancestors_list = self._get_ancestors_of(observed_list)

        # Direction of flow of information
        # up ->  from parent to child
        # down -> from child to parent

        visit_list = set()
        visit_list.add((start, 'up'))
        traversed_list = set()
        active_nodes = set()
        while visit_list:
            node, direction = visit_list.pop()
            if (node, direction) not in traversed_list:
                if node not in observed_list:
                    active_nodes.add(node)
                traversed_list.add((node, direction))
                if direction == 'up' and node not in observed_list:
                    for parent in self.predecessors(node):
                        visit_list.add((parent, 'up'))
                    for child in self.successors(node):
                        visit_list.add((child, 'down'))
                elif direction == 'down':
                    if node not in observed_list:
                        for child in self.successors(node):
                            visit_list.add((child, 'down'))
                    if node in ancestors_list:
                        for parent in self.predecessors(node):
                            visit_list.add((parent, 'up'))
        return active_nodes

    def is_active_trail(self, start, end):
        """
        Returns True if there is any active trail between start and end node

        Parameters
        ----------
        start : Graph Node

        end : Graph Node

        Examples
        --------
        >>> from pgmpy import BayesianModel as bm
        >>> student = bm.BayesianModel()
        >>> student.add_nodes_from(['diff', 'intel', 'grades'])
        >>> student.add_edges_from([('diff', 'grades'), ('intel', 'grades')])
        >>> student.set_states({'diff': ['easy', 'hard'],
        ...                     'intel': ['dumb', 'smart'],
        ...                     'grades': ['A', 'B', 'C']})
        >>> student.set_observations({'grades': 'A'})
        >>> student.is_active_trail('diff', 'intel')
        True

        See Also
        --------
        active_trail_nodes('start')
        """
        if end in self.active_trail_nodes(start):
            return True
        else:
            return False
