#!/usr/bin/env python3

import networkx as nx
import numpy as np
from Exceptions import Exceptions
from BayesianModel import CPD
import itertools
from scipy import sparse
import sys


class BayesianModel(nx.DiGraph):
    """
    Public Methods
    --------------
    add_nodes('node1', 'node2', ...)
    add_edges(('node1', 'node2', ...), ('node3', 'node4', ...))
    add_states('node1', ('state1', 'state2', ...))
    get_states('node1')
    set_rule_for_states('node1', ('state2', 'state1', ...))
    set_rule_for_parents('node1', ('parent1', 'parent2', ...))
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
    def __init__(self, ebunch=None):
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
        >>> G = bm.BayesianModel()
        >>> G.add_nodes_from(['diff', 'intel', 'grade'])
        """
        # if not all([isinstance(elem, str) for elem in nodes]):
        #     raise TypeError("Name of nodes must be strings")
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
        #TODO: _rule_for_parents needs to made into a generator
        #TODO: Right now I have no idea why this ^ TODO is needed.

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

    def add_states(self, node, states):
        # TODO: Add the functionality to accept states of multiple
        # nodes in a single call.
        """
        Adds states to the node.

        Parameters
        ----------
        node  :  Graph node
                Must be already present in the Model

        states :  Container of states
                Can be list or tuple of states.

        See Also
        --------
        get_states

        Examples
        --------
        >>> G = bm.BayesianModel([('diff', 'intel'), ('diff', 'grade'),
        >>>                       ('intel', 'sat')])
        >>> G.add_states('diff', ['easy', 'hard'])
        >>> G.add_states('intel', ['dumb', 'smart'])
        """
        try:
            self.node[node]['_states'].extend([{'name': state,
                                                'observed_status': False}
                                               for state in states])
        except KeyError:
            self.node[node]['_states'] = [
                {'name': state, 'observed_status': False} for state in states]

        self._update_rule_for_states(node, len(self.node[node]['_states']))
    ################# For Reference ########################
    #   self.node[node]['_rule_for_states'] = [            #
    #       n for n in range(len(states))]                 #
    ########################################################
        self._update_node_observed_status(node)

        #TODO _rule_for_states needs to made into a generator

    def get_states(self, node):
        """
        Returns a tuple with states in user-defined order

        Parameters
        ----------
        node  :   node
                Graph Node. Must be already present in the Model.

        See Also
        --------
        set_states

        Examples
        --------
        >>> G = bm.BayesianModel([('diff', 'intel'), ('diff', 'grade'),
        >>>                       ('intel', 'sat')])
        >>> G.add_states('diff', ['easy', 'hard'])
        >>> G.get_states('diff')
        """
        _list_states = []
        for index in self.node[node]['_rule_for_states']:
            _list_states.append(self.node[node]['_states'][index]['name'])
        return _list_states
    #TODO get_states() needs to made into a generator

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
        >>> G = bm.BayesianModel([('diff', 'grade'), ('intel', 'grade'), ('intel', 'SAT')])
        >>> G.add_states('grade', ['A', 'B', 'C'])
        >>> G.number_of_states('grade')
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
        # TODO: Feels that this function is wrong, if it is opposite of
        # _no_extra_states
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
        >>> G = bm.BayesianModel([('diff', 'intel'), ('diff', 'grade')])
        >>> G.add_states('diff', ['easy', 'hard'])
        >>> G.add_states('grade', ['C', 'A', 'C'])
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
        >>> G = bm.BayesianModel([('diff', 'grade'), ('diff', 'intel')])
        >>> G.add_states('diff', ['easy', 'hard'])
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
            #TODO: _rule_for_states needs to made into a generator

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
        #TODO: DocString needs to be improved
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
        >>> G = bm.BayesianModel([('diff', 'intel'), ('diff', 'grade')])
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
        #TODO _rule_for_parents needs to made into a generator

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
        >>> G = bm.BayesianModel([('diff', 'grade'), ('intel', 'grade'),
        >>>                       ('intel', 'SAT'), ('grade', 'reco')])
        >>> G.get_parents('grade')
        ['diff', 'intel']
        >>> G.set_rule_for_parents('grade', ['intel', 'diff'])
        >>> G.get_parents('grade')
        ['intel', 'diff']
        """
        return self.get_rule_for_parents(node)
    #TODO get_parents() needs to made into a generator

    def number_of_parents(self, node):
        """
        Returns the number of parents of node

        node  :  Graph node

        Example
        -------
        >>> G = bm.BayesianModel([('diff', 'grade'), ('intel', 'grade'), ('intel', 'sat')])
        >>> G.number_of_parents('grade')
        2
        """
        return len(self.node[node]['_parents'])

    def _get_parent_objects(self, node):
        """
        Returns a list of those node objects which are parents of
        the argument node.
        """
        return [self.node[parent] for parent in self.get_parents(node)]

    #TODO _get_parent_objects() needs to made into a generator

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
        >>> student = bm.BayesianModel([('diff', 'grade'), ('diff', 'intel')])
        >>> student.add_states('grades', ('A','C','B'))
        >>> student.set_rule_for_parents('grades', ('diff', 'intel'))
        >>> student.set_rule_for_states('grades', ('A', 'B', 'C'))
        >>> student.set_cpd('grades',
        ...             [[0.1,0,1,0.1,0.1,0.1,0.1],
        ...             [0.1,0.1,0.1,0.1,0.1,0.1],
        ...             [0.8,0.8,0.8,0.8,0.8,0.8]]
        ...             )

        #diff:       easy                 hard
        #intel: dumb   avg   smart    dumb  avg   smart
        #gradeA: 0.1    0.1    0.1     0.1  0.1    0.1
        #gradeB: 0.1    0.1    0.1     0.1  0.1    0.1
        #gradeC: 0.8    0.8    0.8     0.8  0.8    0.8
        """
        self.node[node]['_cpd'] = CPD.TabularCPD(cpd)

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

    def set_observations(self, observations, reset=False):
        #TODO: Complete examples
        """
        Sets state of node as observed.

        Parameters
        ----------
        observations  :  A dictionary of the form of {node : state}
                        containing all the observed nodes.

        reset :  if reset is True, the observation status is reset

        See Also
        --------
        reset_observations
        is_observed

        Examples
        --------
        >>> G = bm.BayesianModel([('diff', 'grade'), ('intel', 'grade'), ('grade', 'reco'))])
        >>> G.add_states('diff', ['easy', 'hard'])
        >>> G.add_cpd()
        >>> G.set_observations({'diff': 'easy'})
        """
        #TODO check if multiple states of same node can be observed
        #TODO if above then, change code accordingly
        for node, state in observations.items():
            found = 0
            for state in self.node[node]['_states']:
                if state['name'] == state:
                    state['observed_status'] = True if not reset else False
                    found = 1
                    break
            if found:
                self._update_node_observed_status(node)
            else:
                raise Exceptions.StateError("State: ", state, " not found for node: ", node)

    def reset_observations(self, nodes=None):
        """
        Resets observed-status of given nodes. Will not change a particular
        state. For that use, add_observations with reset=True.

        If no arguments are given, all states of all nodes are reset.

        Parameters
        ----------
        node  :  Graph Node

        See Also
        --------
        set_observations
        is_observed
        """
        if nodes is None:
            nodes = self.nodes()
        try:
            for node in nodes:
                for state in self.node[node]['_states']:
                    state['observed_status'] = False
                self._update_node_observed_status(node)
        except KeyError:
            raise Exceptions.NodeNotFoundError("Node not found", node)

    def is_observed(self, node):
        #TODO: Write an example
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
        """
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
        """
        Returns a list of all observed nodes
        """
        return [node for node in self.nodes()
                if self.node[node]['_observed']]

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
        >>> student.set_cpd('grades',
        ...             [[0.7, 0.6, 0.6, 0.2],
        ...             [0.3, 0.4, 0.4, 0.8]],
        ...             )
        >>> student.set_cpd('diff', [[0.7],[0.3]])
        >>> student.set_cpd('intel', [[0.8], [0.2]])
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
