#!/usr/bin/env python3

import networkx as nx
import numpy as np
import Exceptions


class BayesianModel(nx.DiGraph):
    """ Public Methods
    --------------
    add_nodes('node1', 'node2', ...)
    add_edges(('node1', 'node2', ...), ('node3', 'node4', ...))
    set_states('node1', ('state1', 'state2', ...))
    get_states('node1')
    add_rule_for_states('node1', ('state2', 'state1', ...))
    add_rule_for_parents('node1', ('parent1', 'parent2', ...))
    get_parents('node1')
    set_cpd('node1', cpd1)
    get_cpd('node1')
    set_observed(observations, reset=False)
    reset_observed('node1', ...)
    reset_observed()
    is_observed('node1')
    active_trail_nodes('node1')
    is_active_trail('node1', 'node2')
    """
    #__init__ is inherited
    def _string_to_tuple(self, string):
        """Converts a single string into a tuple with one string element."""
        return (string,)

    def add_nodes(self, *args):
        """Adds nodes to graph with node-labels as provided.
        Node-labels have to be strings.

        EXAMPLE
        -------
        >>> bayesian_model.add_nodes("difficulty", "intelligence", "grades")
        """
        #TODO allow adding of nodes from tuple?
        for item in args:
            if not isinstance(item, str):
                raise TypeError("Name of nodes must be strings.")
        self.add_nodes_from(args)
        #add_nodes_from() is method of nx.Graph

    def add_edges(self, tail, head):
        """Takes two tuples of nodes as input. All nodes in 'tail' are
        joint to all nodes in 'head' with the direction of each edge being
        from a node in 'tail' to a node in 'head'.

        EXAMPLE
        -------
        >>> bayesian_model.add_edges(("difficulty", "intelligence"), "grades")
        """
        #Converting string arguments into tuple arguments
        if isinstance(tail, str):
            tail = self._string_to_tuple(tail)
        if isinstance(head, str):
            head = self._string_to_tuple(head)

        for end_node in head:
            for start_node in tail:
                self.add_edge(start_node, end_node)

            self.node[end_node]['_parents'] = self.predecessors(end_node)
            self.node[end_node]['_rule_for_parents'] = (
                index for index in range(len(head)))

    def add_states(self, node, states):
        """Adds the names of states from the tuple 'states' to given 'node'."""
        self.node[node]['_states'] = [
            {'name': state, 'observed_status': False} for state in states]
        self.node[node]['_rule_for_states'] = (
            n for n in range(len(states)))
        self._update_node_observed_status(node)

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

    def _all_states_mentioned(self, node, states):
        """"Returns True if set(states) is exactly equal to the set of states
        of the Node.

        EXAMPLE
        -------
        >>> bayesian_model._all_states_mentioned('difficulty', ('hard', 'easy'))
        True
        >>> bayesian_model._all_states_mentioned('difficulty', ('hard'))
        MissingStatesError: The following states are missing: 'easy'
        """
        _all_states = set()
        for state in self.node[node]['_states']:
            _all_states.add(state['name'])
        leftout_states = _all_states - set(states)
        extra_states = set(states) - _all_states

        if leftout_states:
            raise Exceptions.MissingStatesError(leftout_states)
        elif extra_states:
            raise Exceptions.ExtraStatesError(extra_states)
        else:
            return True

    def add_rule_for_states(self, node, states):
        """Sets new rule for order of states"""
        if self._all_states_mentioned(node, states):
            _order = list()
            for user_given_state in states:
                for state in self.node[node]['_states']:
                    if state['name'] == user_given_state:
                        _order.append(self.node[node]
                                      ['_states'].index(state))
                        break
            self.node[node]['_rule_for_states'] = tuple(_order)

    def get_states(self, node):
        """Returns tuple with states in user-defined order"""
        for index in self.node[node]['_rule_for_states']:
            yield self.node[node]['_states'][index]['name']

    def _all_parents_mentioned(self, node, parents):
        """"Returns True if set(states) is exactly equal to the set of states
        of the Node.

        EXAMPLE
        -------
        >>> bayesian_model._all_parents_mentioned('grades', ('difficutly',
        ...                                                 'intelligence'))
        True
        >>> bayesian_model._all_parents_mentioned('grades', ('difficulty'))
        MissingParentsError: The following parents are missing: 'intelligence'
        """
        _all_parents = set(self.node[node]['_parents'])
        leftout_parents = _all_parents - set(parents)
        extra_parents = set(parents) - _all_parents

        if leftout_parents:
            raise Exceptions.MissingParentsError(leftout_parents)
        elif extra_parents:
            raise Exceptions.ExtraParentsError(extra_parents)
        else:
            return True

    def add_rule_for_parents(self, node, parents):
        if self._all_parents_mentioned(node, parents):
            _order = list()
            for user_given_parent in parents:
                for parent in self.node[node]['_parents']:
                    if parent == user_given_parent:
                        _order.append(self.node[node]['_parents'].index(parent))
                        break
            self.node[node]['_rule_for_parents'] = tuple(_order)

    def get_parents(self, node):
        """Returns tuple with parents in order"""
        for index in self.node[node]['_rule_for_parents']:
            yield self.node[node]['_parents'][index]

    def set_cpd(self, node, cpd):
        """Adds given CPD to node as numpy.array

        It is expected that CPD given will be a 2D array such that
        the order of probabilities in the array will be in according
        to the rules specified for parent and states.
        An example is shown below.

        EXAMPLE
        -------
        >>> student.set_states('grades', ('A','C','B'))
        >>> student.add_rule_for_parents('grades', ('diff', 'intel'))
        >>> student.add_rule_for_states('grades', ('A', 'B', 'C'))
        >>> student.add_cpd('grade',
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
        self.node[node]['_cpd'] = np.array(cpd)

    def get_cpd(self, node):
        return self.node[node]['_cpd']

    def set_observed(self, observations, reset=False):
        """
        Sets states of nodes as observed.

        @param observations: dictionary with key as node and value as a tuple
                             of states that are observed
        @return:
        """
        #TODO check if multiple states of same node can be observed
        #TODO if not above then, put validation
        for _node in observations:
            for user_given_state in observations[_node]:
                for state in self.node[_node]['_states']:
                    if state[0] == user_given_state:
                        state[1] = True if not reset else False
                        break
            self._calc_observed(_node)

    def reset_observed(self, nodes=False):
        """Resets observed-status of given nodes.

        Will not change a particular state. For that use, set_observed
            with reset=True.

        If no arguments are given, all states of all nodes are reset.
        @param nodes:
        @return:
        """
        if nodes is False:
            _to_reset = self.nodes()
        elif isinstance(nodes, str):
            _to_reset = self._string_to_tuple(nodes)
        else:
            _to_reset = nodes
        for node in _to_reset:
            for state in self.node[node]['_states']:
                state[1] = False
            self._calc_observed(node)

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
        >>> student.set_states('diff', ('hard', 'easy'))
        >>> student.set_states('intel', ('smart', 'avg', 'dumb'))
        >>> student.set_states('grades', ('A', 'B', 'C'))
        >>> student.set_observation({'grades': 'A'})
        >>> student.active_trail_nodes('diff')
        ['diff', 'intel']
        """
        observed_list = self._get_observed_list()
        ancestors_list = self._get_ancestors_observation(observed_list)
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
        if end in self.predecessors(start) or end in self.successors(start):
            return True
        elif end in self.active_trail_nodes(start):
            return True
        else:
            return False
