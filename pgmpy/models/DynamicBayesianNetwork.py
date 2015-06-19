import logging
from copy import deepcopy
from collections import defaultdict
from itertools import chain

import numpy as np
import networkx as nx

from pgmpy.factors import TabularCPD, TreeCPD, RuleCPD
from pgmpy.base import DirectedGraph


class DynamicBayesianNetwork(DirectedGraph):
    def __init__(self, ebunch=None):
        """
        Base class for Dynamic Bayesian Network

        This model is a time variant of the static Bayesian model, where each
        time-slice has some static nodes and is then replicated over a certain
        time-slice.

        The nodes can be any hashable python objects.

        Parameters:
        ----------
        ebunch: Data to initialize graph.  If data=None (default) an empty
              graph is created.  The data can be an edge list, or any NetworkX
              graph object

        Examples:
        --------
        Create an empty Dynamic Bayesian Network with no nodes and no edges
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()

        Adding nodes and edges inside the dynamic bayesian network. A single
        node can be added using the method below. For adding edges we need to
        specify the time slice since edges can be across different time slices.

        >>> dbn.add_nodes_from(['D','G','I','S','L'])
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('G',0),('L',0))])

        >>> dbn.nodes()
        ['L', 'G', 'S', 'I', 'D']
        >>> dbn.edges()
        [(('D', 0), ('G', 0)), (('G', 0), ('L', 0)), (('I', 0), ('G', 0))]

        If any variable is not present in the network while adding an edge,
        pgmpy will automatically add that variable to the network.


        Public Methods:
        ---------------
        add_cpds
        add_edge
        add_edges_from
        add_node
        add_nodes_from
        initialize_initial_state
        inter_slice
        intra_slice
        """
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []
        self.cardinalities = defaultdict(int)
        self.check_nodes = set()

    def add_node(self, node, **attr):
        super().add_node((node, 0), **attr)

    def add_nodes_from(self, nodes, **attr):
        for node in nodes:
            self.add_node(node)

    def nodes(self):
        return list(set([node for node, slice in super().nodes()]))

    def add_edge(self, start, end, **kwargs):
        """
        Add an edge between two nodes.

        The nodes will be automatically added if they are not present in the network.

        Parameters
        ----------
        start, end: The start, end nodes should contain the (node_name, time_slice)
                    Here, node_name can be a hashable python object while the
                    time_slice is an integer value, which denotes the index of the
                    time_slice that the node belongs to.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> model = DBN()
        >>> model.add_nodes_from(['D', 'I'])
        >>> model.add_edge(('D',0), ('I',0))
        >>> model.edges()
        [(('D', 0), ('I', 0))]
        """
        try:
            if len(start) != 2 or len(end) !=2:
                raise ValueError('Nodes must be of type (node, time_slice).')
            elif start[1] == end[1] - 1:
                start = (start[0], 0)
                end = (end[0], 1)
            elif start[1] == end[1] + 1:
                start = (start[0], 1)
                end = (end[0], 0)
            elif start == end:
                raise ValueError('Self Loops are not allowed')
            elif start[1] != end[1]:
                raise ValueError("Edges over multiple time slices is not currently supported")
        except TypeError:
            raise ValueError('Nodes must be of type (node, time_slice).')
        #
        # check_nodes = set(chain(*self.edges()))
        #
        # if start in check_nodes and end in check_nodes and nx.has_path(self, end, start):
        #     raise ValueError('Loops are not allowed. Adding the edge from ({node1}->{node2}) forms a loop.'.format(
        #         node1=start, node2=end))

        super().add_edge(start, end, **kwargs)
        #
        # if start[1] == end[1]:
        #     super().add_edge((start[0], 1 - start[1]), (end[0], 1 - end[1]))
        #
        # self._correct_nodes()

    def add_edges_from(self, ebunch, **kwargs):
        for edge in ebunch:
            self.add_edge(edge[0], edge[1])

    # def _correct_nodes(self):
    #     """
    #     This method automatically adjusts the nodes inside the
    #     Bayesian Network automatically and is used
    #     inside the add_edge() method
    #     EXAMPLE
    #     -------
    #     >>> from pgmpy.models import DynamicBayesianNetwork as DBN
    #     >>> dbn = DBN()
    #     >>> dbn.add_nodes_from(['D', 'I'])
    #     >>> dbn.add_edge(('D',0), ('I',0))
    #     >>> dbn.edges()
    #     [(('D', 0), ('I', 0))]
    #     >>> dbn.nodes()
    #     ['I', 'D']
    #     """
    #     temp_dict = deepcopy(self.node)
    #     for node in temp_dict:
    #         if isinstance(node, (tuple, list)):
    #             temp_value = self.node[node]
    #             self.node.pop(node)
    #             if node[0] not in self.node:
    #                 self.node[node[0]] = temp_value

    def get_intra_edges(self):
        """
        returns the intra slice edges present in the 2-TBN.
        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D','G','I','S','L'])
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('G',0),('L',0)),(('D',0),('D',1))])
        >>> dbn.get_intra_edges()
        [(('D', 0), ('G', 0)), (('G', 0), ('L', 0)), (('I', 0), ('G', 0))
        """
        return [edge for edge in self.edges() if edge[0][1] == edge[1][1]]

    def get_inter_edges(self):
        """
        returns the inter-slice edges present in the 2-TBN
        EXAMPLE:
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D','G','I','S','L'])
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1)))])
        >>> dbn.get_inter_edges()
        [(('D', 0), ('D', 1)), (('I', 0), ('I', 1))]
        """
        return [edge for edge in self.edges() if edge[0][1] != edge[1][1]]

    def add_cpds(self, *cpds):
        """
        This method adds the cpds to the dynamic bayesian network.
        Note that while adding variables and the evidence in cpd,
        they have to be of the following form
        (node_name, time_slice)
        Here, node_name is the node that is inserted
        while the time_slice is an integer value, which denotes
        the index of the time_slice that the node belongs to.

        Parameter
        ---------
        cpds  :  list, set, tuple (array-like)
            List of cpds (TabularCPD, TreeCPD, RuleCPD, Factor)
            which will be associated with the model

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors import TabularCPD
        >>> dbn = DBN()
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
        >>> grade_cpd = TabularCPD(('G',0), 3, [[0.3,0.05,0.9,0.5],
        ...                                     [0.4,0.25,0.8,0.03],
        ...                                     [0.3,0.7,0.02,0.2]], [('I', 0),('D', 0)],[2,2])
        >>> d_i_cpd = TabularCPD(('D',1),2,[[0.6,0.3],[0.4,0.7]],[('D',0)],2)
        >>> diff_cpd = TabularCPD(('D',0),2,[[0.6,0.4]])
        >>> intel_cpd = TabularCPD(('I',0),2,[[0.7,0.3]])
        >>> i_i_cpd = TabularCPD(('I',1),2,[[0.5,0.4],[0.5,0.6]],[('I',0)],2)
        >>> dbn.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
        >>> dbn.cpds
        """
        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, TreeCPD, RuleCPD)):
                raise ValueError('cpds should be an instances of TabularCPD, TreeCPD or RuleCPD')
            # check_nodes = []
            #
            # if all(map(lambda x: isinstance(x, (tuple, list)) and len(x) == 2 and x[1] in (0,1), iter(cpd.variables))):
            #     check_nodes.extend([x[0] for x in cpd.variables])
            # else:
            #     raise ValueError('CPD should have variables along with the time slices', cpd)

            if set(cpd.variables) - set(cpd.variables).intersection(set(super().nodes())):
                raise ValueError('CPD defined on variable not in the model', cpd)

            # self.check_nodes = self.check_nodes.union(cpd.variables)
            #
            # for prev_cpd_index in range(len(self.cpds)):
            #     if self.cpds[prev_cpd_index].variable == cpd.variable:
            #         logging.warning("Replacing existing CPD for {var}".format(var=cpd.variable))
            #         self.cpds[prev_cpd_index] = cpd
            #         break
            # else:
            self.cpds.append(cpd)

    def get_cpds(self, node=None):
        """
        Returns the cpds that have been added till now to the graph

        Parameter
        ---------
        node: The node should be be of the following form
        (node_name, time_slice)
        Here, node_name is the node that is inserted
        while the time_slice is an integer value, which denotes
        the index of the time_slice that the node belongs to.


         EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors import TabularCPD
        >>> dbn = DBN()
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
        >>> grade_cpd =  TabularCPD(('G',0), 3, [[0.3,0.05,0.9,0.5],
        ...                                      [0.4,0.25,0.8,0.03],
        ...                                      [0.3,0.7,0.02,0.2]], [('I', 0),('D', 0)],[2,2])
        >>> dbn.add_cpds(grade_cpd)
        >>> dbn.get_cpds()
        """
        if node:
            if node not in super().nodes():
                raise ValueError('Node not present in the model.')
            else:
                for cpd in self.cpds:
                    if cpd.variable == node:
                        return cpd
        else:
            return self.cpds
    #
    # def check_model(self):
    #     """
    #     Check the model for various errors. This method checks for the following
    #     errors.
    #
    #     * Checks if the sum of the probabilities for each state is equal to 1 (tol=0.01).
    #     * Checks if the CPDs associated with nodes are consistent with their parents.
    #
    #     Returns
    #     -------
    #     check: boolean
    #         True if all the checks are passed
    #     """
    #     for node in self.check_nodes:
    #         cpd = self.get_cpds(node=node)
    #         if isinstance(cpd, TabularCPD):
    #             evidence = cpd.evidence
    #             parents = self.get_parents(node)
    #             if set(evidence if evidence else []) != set(parents if parents else []):
    #                 raise ValueError("CPD associated with %s doesn't have "
    #                                  "proper parents associated with it." % node)
    #             if not np.allclose(cpd.marginalize(node, inplace=False).values,
    #                                np.ones(np.product(cpd.evidence_card)),
    #                                atol=0.01):
    #                 raise ValueError('Sum of probabilites of states for node %s'
    #                                  ' is not equal to 1.' % node)
    #     return True
    #
    # def initialize_initial_state(self):
    #     """
    #     This method will automatically re-adjust the cpds and the edges added to the bayesian network.
    #     If an edge that is added as an intra time slice edge in the 0th timeslice, this method will automatically
    #     add it in the 1st timeslice. It will also add the cpds.
    #     However, to call this method, one needs to add cpds as well as the edges in the
    #     bayesian network of the whole skeleton including the 0th and the 1st timeslice,.
    #     EXAMPLE
    #     -------
    #     >>> from pgmpy.models import DynamicBayesianNetwork as DBN
    #     >>> from pgmpy.factors import TabularCPD
    #     >>> student = DBN()
    #     >>> student.add_nodes_from(['D','G','I','S','L'])
    #     >>> student.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
    #     >>> grade_cpd = TabularCPD(('G',0), 3, [[0.3,0.05,0.9,0.5],
    #     ...                                                 [0.4,0.25,0.8,0.03],
    #     ...                                                 [0.3,0.7,0.02,0.2]], [('I', 0),('D', 0)],[2,2])
    #     >>> d_i_cpd = TabularCPD(('D',1),2,[[0.6,0.3],[0.4,0.7]],[('D',0)],2)
    #     >>> diff_cpd = TabularCPD(('D',0),2,[[0.6,0.4]])
    #     >>> intel_cpd = TabularCPD(('I',0),2,[[0.7,0.3]])
    #     >>> i_i_cpd = TabularCPD(('I',1),2,[[0.5,0.4],[0.5,0.6]],[('I',0)],2)
    #     >>> student.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
    #     >>> student.initialize_initial_state()
    #     """
    #
    #     for cpd in self.cpds:
    #         temp_var = (cpd.variable[0], 1 - cpd.variable[1])
    #         parents = self.get_parents(temp_var)
    #         if not any(x.variable == temp_var for x in self.cpds):
    #             if all(x[1] == parents[0][1] for x in parents):
    #                 if parents:
    #                     new_cpd = TabularCPD(temp_var, cpd.variable_card, np.split(cpd.values, cpd.variable_card), parents,
    #                      cpd.evidence_card)
    #                 else:
    #                     new_cpd = TabularCPD(temp_var, cpd.variable_card, np.split(cpd.values, cpd.variable_card))
    #                 self.add_cpds(new_cpd)
    #     self.check_model()
