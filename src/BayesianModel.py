#!/usr/bin/env python3

import networkx as nx
import numpy as np
import ExceptionsPgmPy as epp


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
    """
    #__init__ is inherited
    def add_nodes(self, *args):
        """Adds nodes to graph with node-labels as provided in function.
        Currently, only string-labels are allowed.
        """
        for item in args:
            if not isinstance(item, str):
                raise TypeError("Name of nodes must be strings.")
        self.add_nodes_from(args)

    def _string_to_tuple(self, string):
        """Converts a single string into a tuple with a string element."""
        return (string,)

    def add_edges(self, tail, head):
        """Takes two tuples of nodes as input. All nodes in 'tail' are
        joint to all nodes in 'head' with the direction of each edge is
        from a node in 'tail' to a node in 'head'.
        """
        if isinstance(tail, str):
            tail = self._string_to_tuple(tail)
        if isinstance(head, str):
            head = self._string_to_tuple(head)

        for end_node in head:
            for start_node in tail:
                self.add_edge(start_node, end_node)

            self.node[end_node]['_parents'] = sorted(
                self.predecessors(end_node))
            self.node[end_node]['_rule_for_parents'] = (
                index for index in range(len(head)))

    def set_states(self, node, states):
        """Adds state-name from 'state' tuple to 'node'."""
        self.node[node]['_states'] = [
            [state, False] for state in sorted(states)]
        self.node[node]['_rule_for_states'] = (
            n for n in range(len(states)))
        self._calc_observed(node)
        #internal storage = [['a',0],['b',0],['c',0],]
        #user-given order = ('c','a','b')
        #_rule_for_states = (2,0,1)
        #Rule will contain indices with which internal order should be
        #accessed

    def _calc_observed(self, node):
        """
        Return True if any of the states of the node are observed
        @param node:
        @return:
        """

        for state in self.node[node]['_states']:
            if state[1]:
                self.node[node]['_observed'] = True
                break
        else:
            self.node[node]['_observed'] = False

    def add_rule_for_states(self, node, states):
        """Sets new rule for order of states"""
        #TODO check whether all states are mentioned?
        _order = list()
        for user_given_state in states:
            for state in self.node[node]['_states']:
                if state[0] == user_given_state:
                    _order.append(self.node[node]['_states'].index(state))
                    break
        self.node[node]['_rule_for_states'] = tuple(_order)

    def get_states(self, node):
        """Returns tuple with states in user-defined order"""
        for index in self.node[node]['_rule_for_states']:
            yield self.node[node]['_states'][index][0]

    def add_rule_for_parents(self, node, parents):
        #check if all parents are mentioned and no extra parents are
        ##present
        #_extra = set(parents) - set(self.predecessors(node))
        #_missing = set(self.predecessors(node)) - set(parents)
        #if not len(_missing):
            #raise epp.MissingParentsError(_missing)
        #if not len(_extra):
            #raise epp.ExtraParentsError(_extra)
        _ord = list()
        for user_given_parent in parents:
            for parent in self.node[node]['_parents']:
                if parent == user_given_parent:
                    _ord.append(self.node[node]['_parents'].index(parent))
                    break
        self.node[node]['_rule_for_parents'] = tuple(_ord)

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
if __name__ == '__main__':
    student = BayesianModel()
    student.add_nodes('diff', 'intel', 'grades')
    student.add_edges(('diff', 'intel'), ('grades',))
    print(sorted(student.edges()))
    student.set_states('diff', ('hard', 'easy'))
    print([m for m in student.states('diff')])
    student.set_states('intel', ('smart', 'avg', 'dumb'))
    print([m for m in student.states('intel')])
    student.set_states('grades', ('A', 'B', 'C'))
    print([m for m in student.states('grades')])
    student.add_rule_for_parents('grades', ('intel', 'diff'))
    print([m for m in student.parents('grades')])
    student.add_rule_for_states('grades', ('C', 'A', 'B'))
    print([m for m in student.states('grades')])
