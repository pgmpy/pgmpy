#!/usr/bin/env python3

import networkx as nx
import numpy as np
import Exceptions
from BayesianModel import CPDs
import itertools
from scipy import sparse


class BayesianModel(nx.DiGraph):
    """
    Public Methods
    --------------
    add_nodes('node1', 'node2', ...)
    add_edges(('node1', 'node2', ...), ('node3', 'node4', ...))
    add_states('node1', ('state1', 'state2', ...))
    get_states('node1')
    add_rule_for_states('node1', ('state2', 'state1', ...))
    add_rule_for_parents('node1', ('parent1', 'parent2', ...))
    get_parents('node1')
    add_tablularcpd('node1', cpd1)
    get_cpd('node1')
    add_observations(observations, reset=False)
    reset_observed_nodes('node1', ...)
    is_observed('node1')
    active_trail_nodes('node1')
    is_active_trail('node1', 'node2')
    marginal_probability('node')
    """
    #__init__ is inherited
    def _string_to_tuple(self, string):
        """Converts a single string into a tuple with one string element."""
        return (string,)

    def add_node(self, node):
        """
        Add a single node to the Graph.

        Parameters
        ----------
        node: node
              A node can only be a string.

        See Also
        --------
        add_nodes_from

        Examples
        --------
        >>> G = bm.BayesianModel()
        >>> G.add_node('difficulty')
        """
        if not isinstance(node, str):
            raise TypeError("Name of nodes must be strings")

        nx.DiGraph.add_node(self, node)

    def add_nodes_from(self, nodes):
        """
        Add multiple nodes to the Graph.

        Parameters
        ----------
        nodes: iterable container
               A container of nodes (list, dict, set, etc.).

        See Also
        --------
        add_node

        Examples
        --------
        >>> G = bm.BayesianModel()
        >>> G.add_nodes_from(['diff', 'intel', 'grade'])
        """
        if not all([isinstance(elem, str) for elem in nodes]):
            raise TypeError("Name of nodes must be strings")

        nx.DiGraph.add_nodes_from(self, nodes)

    def add_edges(self, *args):
        """Takes two tuples of nodes as input. All nodes in 'tail' are
        joint to all nodes in 'head' with the direction of each edge being
        from a node in 'tail' to a node in 'head'.

        EXAMPLE
        -------
        >>> bayesian_model.add_edges([('difficulty', 'grade'), ('intelligence', 'grades')])
        """
        #Converting string arguments into tuple arguments
        # if isinstance(tail, str):
        #     tail = self._string_to_tuple(tail)
        # if isinstance(head, str):
        #     head = self._string_to_tuple(head)
        #
        # for head_node in head:
        #     for tail_node in tail:
        #         self.add_edge(tail_node, head_node)
        self.add_edges_from(args)
        for node in self.nodes():
            self.node[node]['_parents'] = self.predecessors(node)
            self.node[node]['_rule_for_parents'] = [
                index for index in range(len(self.neighbors(node)))]
        # self.node[head_node]['_parents'] = self.predecessors(head_node)
        # self.node[head_node]['_rule_for_parents'] = [
        #         index for index in range(len(tail))]
        #TODO _rule_for_parents needs to made into a generator
            #TODO after each call to add_edges or call update node _parents and update _rule_for_parents

    def add_states(self, node, states):
        """Adds the names of states from the tuple 'states' to given 'node'."""
        self.node[node]['_states'] = [
            {'name': state, 'observed_status': False} for state in states]
        self.node[node]['_rule_for_states'] = [
            n for n in range(len(states))]
        self._update_node_observed_status(node)
        #TODO _rule_for_states needs to made into a generator

    def _update_node_observed_status(self, node):
        """
        Updates '_observed' in the node-dictionary.

        If any of the states of a node are observed, node.['_observed']
        is made True. Otherwise, it is False.
        """
        for state in self.node[node]['_states']:
            if state['observed_status']:
                self.node[node]['_observed'] = True
                break
        else:
            self.node[node]['_observed'] = False

    def _no_missing_states(self, node, states):
        """"Returns True if all the states of the node are present in the
        argument states.

        EXAMPLE
        -------
        >>> bayesian_model._no_missing_states('difficulty', ('hard', 'easy'))
        True
        """
        _all_states = set()
        for state in self.node[node]['_states']:
            _all_states.add(state['name'])
        missing_states = _all_states - set(states)

        if missing_states:
            raise Exceptions.MissingStatesError(missing_states)
        else:
            return True

    def _no_extra_states(self, node, states):
        """"Returns True if the argument states contains only the states
         present in Node.

        EXAMPLE
        -------
        >>> bayesian_model._no_extra_states('difficulty', ('hard', 'easy'))
        True
        """
        _all_states = set()
        for state in self.node[node]['_states']:
            _all_states.add(state['name'])
        extra_states = set(states) - _all_states

        if extra_states:
            raise Exceptions.ExtraStatesError(extra_states)
        else:
            return True

    def add_rule_for_states(self, node, states):
        """Sets new rule for order of states"""
        if self._no_extra_states(node, states) and \
                self._no_missing_states(node, states):
            _order = list()
            for user_given_state, state in itertools.product(
                    states, self.node[node]['_states']):
                if state['name'] == user_given_state:
                    _order.append(self.node[node]
                                  ['_states'].index(state))
                    break
            self.node[node]['_rule_for_states'] = _order
            #TODO _rule_for_states needs to made into a generator

    def get_states(self, node):
        """Returns tuple with states in user-defined order"""
        _list_states = list()
        for index in self.node[node]['_rule_for_states']:
            _list_states.append(self.node[node]['_states'][index]['name'])
        return _list_states
    #TODO get_states() needs to made into a generator

    def _no_extra_parents(self, node, parents):
        """"Returns True if set(states) is exactly equal to the set of states
        of the Node.

        EXAMPLE
        -------
        >>> bayesian_model._no_extra_parents('grades', ('difficutly',
        ...                                                 'intelligence'))
        True
        """
        _all_parents = set(self.node[node]['_parents'])
        extra_parents = set(parents) - _all_parents
        if extra_parents:
            raise Exceptions.ExtraParentsError(extra_parents)
        else:
            return True

    def _no_missing_parents(self, node, parents):
        """"Returns True if all parents of node are present in the
        argument parents.

        EXAMPLE
        -------
        >>> bayesian_model._no_missing_parents('grades', ('difficutly',
        ...                                                 'intelligence'))
        True
        """
        _all_parents = set(self.node[node]['_parents'])
        missing_parents = _all_parents - set(parents)
        if missing_parents:
            raise Exceptions.MissingParentsError(missing_parents)
        else:
            return True

    def add_rule_for_parents(self, node, parents):
        if self._no_missing_parents(node, parents) and \
                self._no_extra_parents(node, parents):
            _order = list()
            for user_given_parent, parent in itertools.product(
                    parents, self.node[node]['_parents']):
                if parent == user_given_parent:
                    _order.append(
                        self.node[node]['_parents'].index(parent))
                    break
        self.node[node]['_rule_for_parents'] = _order
        #TODO _rule_for_parents needs to made into a generator

    def get_parents(self, node):
        """Returns tuple with name of parents in order"""
        _str_parent_list = list()
        for index in self.node[node]['_rule_for_parents']:
            _str_parent_list.append(self.node[node]['_parents'][index])
        return _str_parent_list
    #TODO get_parents() needs to made into a generator

    def _get_parent_objects(self, node):
        """Returns tuple with parent-objects in order"""
        _obj_parent_list = list()
        for _parent in self.get_parents(node):
            _obj_parent_list.append(self.node[_parent])
        return _obj_parent_list
    #TODO _get_parent_objects() needs to made into a generator

    def add_tablularcpd(self, node, cpd):
        """Adds given CPD to node as numpy.array

        It is expected that CPD given will be a 2D array such that
        the order of probabilities in the array will be in according
        to the rules specified for parent and states.
        An example is shown below.

        EXAMPLE
        -------
        >>> student.add_states('grades', ('A','C','B'))
        >>> student.add_rule_for_parents('grades', ('diff', 'intel'))
        >>> student.add_rule_for_states('grades', ('A', 'B', 'C'))
        >>> student.add_tabularcpd('grades',
        ...             [[0.1,0
        .1,0.1,0.1,0.1,0.1],
        ...             [0.1,0.1,0.1,0.1,0.1,0.1],
        ...             [0.8,0.8,0.8,0.8,0.8,0.8]]
        ...             )

        #diff:       easy                 hard
        #intel: dumb   avg   smart    dumb  avg   smart
        #gradeA: 0.1    0.1    0.1     0.1  0.1    0.1
        #gradeB: 0.1    0.1    0.1     0.1  0.1    0.1
        #gradeC: 0.8    0.8    0.8     0.8  0.8    0.8
        """
        self.node[node]['_cpd'] = CPDs.TabularCPD(cpd)

    def get_cpd(self, node):
        return self.node[node]['_cpd'].table

    def add_observations(self, observations, reset=False):
        """
        Sets states of nodes as observed.

        observations: dictionary with key as node and value as the
                      states that is observed
        reset: if reset is True, the observation status is reset
        """
        #TODO check if multiple states of same node can be observed
        #TODO if above then, change code accordingly
        for _node in observations:
            for user_given_state in observations[_node]:
                if self._no_extra_states(_node, (user_given_state,)):
                    for state in self.node[_node]['_states']:
                        if state['name'] == user_given_state:
                            state['observed_status'] = True if not reset \
                                else False
                            break
                        self._update_node_observed_status(_node)

    def reset_observed_nodes(self, nodes=None):
        """Resets observed-status of given nodes.

        Will not change a particular state. For that use,
        add_observations with reset=True.

        If no arguments are given, all states of all nodes are reset.
        """
        if nodes is None:
            _to_reset = self.nodes()
        elif isinstance(nodes, str):
            _to_reset = self._string_to_tuple(nodes)
        else:
            _to_reset = nodes
        for node in _to_reset:
            for state in self.node[node]['_states']:
                state['observed_status'] = False
            self._update_node_observed_status(node)

    def is_observed(self, node):
        return self.node[node]['_observed']

    def _get_ancestors_observation(self, observation):
        """Returns ancestors of all the observed elements"""
        obs_list = [obs for obs in observation]
        ancestors_list = []
        while obs_list:
            node = obs_list.pop()
            if node not in ancestors_list:
                obs_list += self.predecessors(node)
            ancestors_list += [node]
        return ancestors_list

    def _get_observed_list(self):
        """Returns a list of all observed nodes"""
        return [_node for _node in self.nodes()
                if self.node[_node]['_observed']]

    def active_trail_nodes(self, start):
        """Returns all the nodes reachable from start via an active trail

        -------------------------------------------------------------------
        Details of algorithm can be found in 'Probabilistic Graphical Model
        Principles and Techniques' - Koller and Friedman
        Page 75 Algorithm 3.1
        -------------------------------------------------------------------

        EXAMPLE
        -------
        >>> student.add_nodes('diff', 'intel', 'grades')
        >>> student.add_edges(('diff', 'intel'), ('grades',))
        >>> student.add_states('diff', ('hard', 'easy'))
        >>> student.add_states('intel', ('smart', 'avg', 'dumb'))
        >>> student.add_states('grades', ('A', 'B', 'C'))
        >>> student.add_observations({'grades': 'A'})
        >>> student.active_trail_nodes('diff')
        ['diff', 'intel']
        """
        observed_list = self._get_observed_list()
        ancestors_list = self._get_ancestors_observation(observed_list)
        """
        Direction of flow of information
        up ->  from parent to child
        down -> from child to parent
        """
        visit_list = [(start, 'up')]
        traversed_list = []
        active_nodes = []
        while visit_list:
            _node, direction = visit_list.pop()
            if (_node, direction) not in traversed_list:
                if _node not in observed_list:
                    active_nodes += [_node]
                traversed_list += (_node, direction),
                if direction == 'up' and _node not in observed_list:
                    for parent in self.predecessors(_node):
                        visit_list += (parent, 'up'),
                    for child in self.successors(_node):
                        visit_list += (child, 'down'),
                elif direction == 'down':
                    if _node not in observed_list:
                        for child in self.successors(_node):
                            visit_list += (child, 'down'),
                    if _node in ancestors_list:
                        for parent in self.predecessors(_node):
                            visit_list += (parent, 'up'),
        active_nodes = list(set(active_nodes))
        return active_nodes

    def is_active_trail(self, start, end):
        """Returns True if there is any active trail
        between start and end node"""
        if end in self.predecessors(start) or end in self.successors(start)\
                or end in self.active_trail_nodes(start):
            return True
        else:
            return False

    def marginal_probability(self, node):
        """
        Returns marginalized probability distribution of a node.

        EXAMPLE
        -------
        >>> student.add_nodes('diff', 'intel', 'grades')
        >>> student.add_edges(('diff', 'intel'), ('grades',))
        >>> student.add_states('diff', ('hard', 'easy'))
        >>> student.add_states('intel', ('smart', 'dumb'))
        >>> student.add_states('grades', ('good', 'bad'))
        >>> student.add_tabularcpd('grades',
        ...             [[0.7, 0.6, 0.6, 0.2],
        ...             [0.3, 0.4, 0.4, 0.8]],
        ...             )
        >>> student.add_tabularcpd('diff', [[0.7],[0.3]])
        >>> student.add_tabularcpd('intel', [[0.8], [0.2]])
        >>> student.marginal_probability('grades')
        array([[ 0.632],
               [ 0.368]])
        """
        parent_list = [parent for parent in self._get_parent_objects(node)]
        num_parents = len(parent_list)
        mar_dist = self.node[node]['_cpd'].table
        """
        Kronecker product of two arrays
        kron(a,b)[k0,k1,...,kN] = [[ a[0,0]*b,   a[0,1]*b,  ... , a[0,-1]*b  ],
                                   [  ...                              ...   ],
                                   [ a[-1,0]*b,  a[-1,1]*b, ... , a[-1,-1]*b ]]
        """
        for num, val in enumerate(reversed(parent_list)):
            #_mat is compressed sparse row matrix
            _mat = sparse.csr_matrix(np.kron(np.identity(num_parents-num),
                                             val['_cpd'].table))
            mar_dist = sparse.csr_matrix.dot(mar_dist, _mat)

        return mar_dist
