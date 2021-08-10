from itertools import combinations, chain
from collections import defaultdict
from dataclasses import dataclass
import typing

import numpy as np
import pandas as pd
import networkx as nx

from pgmpy.factors.discrete import TabularCPD
from pgmpy.base import DAG


@dataclass(eq=True, frozen=True)
class DynamicNode:
    """
    Class for representing the nodes of Dynamic Bayesian Networks.
    """

    node: str
    time_slice: int

    def __getitem__(self, idx: int) -> typing.Union[str, int]:
        if idx == 0:
            return self.node
        elif idx == 1:
            return self.time_slice
        else:
            raise IndexError(f"Index {idx} out of bounds.")

    def __str__(self) -> str:
        return f"({self.node}, {self.time_slice})"

    def __repr__(self) -> str:
        return f"<DiscreteNode({self.node}, {self.time_slice}) at {hex(id(self))}>"

    def __lt__(self, other) -> bool:
        if self.node < other.node:
            return True
        elif self.node > other.node:
            return False
        else:
            return self.time_slice < other.time_slice

    def __le__(self, other) -> bool:
        if self.node <= other.node:
            return True
        elif self.node > other.node:
            return False
        else:
            return self.time_slice <= other.time_slice

    def __eq__(self, other) -> bool:
        if isinstance(other, DynamicNode):
            return (self.node, self.time_slice) == (other.node, other.time_slice)
        elif isinstance(other, (list, tuple)):
            return (self.node, self.time_slice) == tuple(other)
        else:
            return False

    def to_tuple(self) -> tuple:
        """
        Returns a tuple representation as (node, time_slice) for DynamicNode object.
        """
        return (self.node, self.time_slice)


class DynamicBayesianNetwork(DAG):
    def __init__(self, ebunch=None):
        """
        Base class for Dynamic Bayesian Network

        This is a time variant model of the static Bayesian model, where each
        time-slice has some static nodes and is then replicated over a certain
        time period.

        The nodes can be any hashable python objects.

        Parameters
        ----------
        ebunch: Data to initialize graph.  If data=None (default) an empty
              graph is created.  The data can be an edge list, or any NetworkX
              graph object

        Examples
        --------
        Create an empty Dynamic Bayesian Network with no nodes and no edges:
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()

        Adding nodes and edges inside the dynamic bayesian network. A single
        node can be added using the method below. For adding edges we need to
        specify the time slice since edges can be across different time slices.

        For example for a network as [image](http://s8.postimg.org/aaybw4x2t/Blank_Flowchart_New_Page_1.png),
        we will need to add all the edges in the 2-TBN as:

        >>> dbn.add_edges_from([(('D', 0), ('G', 0)), (('I', 0), ('G', 0)),
        ...                     (('G', 0), ('L', 0)), (('D', 0), ('D', 1)),
        ...                     (('I', 0), ('I', 1)), (('G', 0), ('G', 1)),
        ...                     (('G', 0), ('L', 1)), (('L', 0), ('L', 1))])

        We can query the edges and nodes in the network as:
        >>> dbn.nodes()
        ['G', 'D', 'I', 'L']
        >>> dbn.edges()
        [(('D', 1), ('G', 1)), (('I', 0), ('G', 0)), (('I', 0), ('I', 1)),
         (('I', 1), ('G', 1)), (('G', 0), ('L', 0)), (('G', 0), ('G', 1)),
         (('G', 0), ('L', 1)), (('D', 0), ('G', 0)), (('D', 0), ('D', 1)),
         (('L', 0), ('L', 1)), (('G', 1), ('L', 1))]

        If any variable is not present in the network while adding an edge,
        pgmpy will automatically add that variable to the network.

        But for adding nodes to the model we don't need to specify the time
        slice as it is common in all the time slices. And therefore pgmpy
        automatically replicated it all the time slices. For example, for
        adding a new variable `S` in the above network we can simply do:
        >>> dbn.add_node('S')
        >>> dbn.nodes()
        ['S', 'G', 'D', 'I', 'L']

        Public Methods
        --------------
        add_node
        add_nodes_from
        add_edges
        add_edges_from
        add_cpds
        initialize_initial_state
        inter_slice
        intra_slice
        copy
        """
        super(DynamicBayesianNetwork, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.cpds = []
        self.cardinalities = defaultdict(int)

    def add_node(self, node, **attr):
        """
        Adds a single node to the Network

        Parameters
        ----------
        node: node
            A node can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_node('A')
        ['A']
        """
        super(DynamicBayesianNetwork, self).add_node(DynamicNode(node, 0), **attr)

    def add_nodes_from(self, nodes, **attr):
        """
        Add multiple nodes to the Network.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, etc.).

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['A', 'B', 'C'])
        """
        for node in nodes:
            self.add_node(node)

    def _nodes(self):
        """
        Returns the list of nodes present in the network

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['A', 'B', 'C'])
        >>> sorted(dbn._nodes())
        ['B', 'A', 'C']
        """
        return list(
            set(
                [
                    node
                    for node, timeslice in super(DynamicBayesianNetwork, self).nodes()
                ]
            )
        )

    def _timeslices(self):
        return list(
            set(
                [
                    timeslice
                    for node, timeslice in super(DynamicBayesianNetwork, self).nodes()
                ]
            )
        )

    def add_edge(self, start, end, **kwargs):
        """
        Add an edge between two nodes.

        The nodes will be automatically added if they are not present in the network.

        Parameters
        ----------
        start: tuple
               Both the start and end nodes should specify the time slice as
               (node_name, time_slice). Here, node_name can be any hashable
               python object while the time_slice is an integer value,
               which denotes the time slice that the node belongs to.

        end: tuple
               Both the start and end nodes should specify the time slice as
               (node_name, time_slice). Here, node_name can be any hashable
               python object while the time_slice is an integer value,
               which denotes the time slice that the node belongs to.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> model = DBN()
        >>> model.add_nodes_from(['D', 'I'])
        >>> model.add_edge(('D',0), ('I',0))
        >>> sorted(model.edges())
        [(('D', 0), ('I', 0)), (('D', 1), ('I', 1))]
        """
        try:
            if len(start) != 2 or len(end) != 2:
                raise ValueError("Nodes must be of type (node, time_slice).")
            elif not isinstance(start[1], int) or not isinstance(end[1], int):
                raise ValueError("Nodes must be of type (node, time_slice).")
            elif start[1] == end[1]:
                start = (start[0], 0)
                end = (end[0], 0)
            elif start[1] == end[1] - 1:
                start = (start[0], 0)
                end = (end[0], 1)
            elif start[1] > end[1]:
                raise NotImplementedError(
                    "Edges in backward direction are not allowed."
                )
            elif start[1] != end[1]:
                raise ValueError(
                    "Edges over multiple time slices is not currently supported"
                )
        except TypeError:
            raise ValueError("Nodes must be of type (node, time_slice).")

        start = DynamicNode(*start)
        end = DynamicNode(*end)

        if start == end:
            raise ValueError("Self Loops are not allowed")
        elif (
            start in super(DynamicBayesianNetwork, self).nodes()
            and end in super(DynamicBayesianNetwork, self).nodes()
            and nx.has_path(self, end, start)
        ):
            raise ValueError(
                f"Loops are not allowed. Adding the edge from ({str(start)} --> {str(end)}) forms a loop."
            )

        super(DynamicBayesianNetwork, self).add_edge(start, end, **kwargs)

        if start[1] == end[1]:
            super(DynamicBayesianNetwork, self).add_edge(
                DynamicNode(start[0], 1 - start[1]), DynamicNode(end[0], 1 - end[1])
            )
        else:
            super(DynamicBayesianNetwork, self).add_node(
                DynamicNode(end[0], 1 - end[1])
            )

    def add_edges_from(self, ebunch, **kwargs):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python object.

        Parameters
        ----------
        ebunch : list, array-like
                List of edges to add. Each edge must be of the form of
                ((start, time_slice), (end, time_slice)).

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_edges_from([(('D', 0), ('G', 0)), (('I', 0), ('G', 0))])
        >>> dbn.nodes()
        ['G', 'I', 'D']
        >>> dbn.edges()
        [(('D', 1), ('G', 1)),
         (('I', 1), ('G', 1)),
         (('D', 0), ('G', 0)),
         (('I', 0), ('G', 0))]
        """
        for edge in ebunch:
            self.add_edge(edge[0], edge[1])

    def get_intra_edges(self, time_slice=0):
        """
        Returns the intra slice edges present in the 2-TBN.

        Parameters
        ----------
        time_slice: int (whole number)
                The time slice for which to get intra edges. The timeslice
                should be a positive value or zero.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D', 'G', 'I', 'S', 'L'])
        >>> dbn.add_edges_from([(('D', 0), ('G', 0)), (('I', 0), ('G', 0)),
        ...                     (('G', 0), ('L', 0)), (('D', 0), ('D', 1)),
        ...                     (('I', 0), ('I', 1)), (('G', 0), ('G', 1)),
        ...                     (('G', 0), ('L', 1)), (('L', 0), ('L', 1))])
        >>> dbn.get_intra_edges()
        [(('D', 0), ('G', 0)), (('G', 0), ('L', 0)), (('I', 0), ('G', 0))]
        """
        if not isinstance(time_slice, int) or time_slice < 0:
            raise ValueError(
                "The timeslice should be a positive value greater than or equal to zero"
            )

        return [
            tuple(DynamicNode(x[0], time_slice) for x in edge)
            for edge in self.edges()
            if edge[0][1] == edge[1][1] == 0
        ]

    def get_inter_edges(self):
        """
        Returns the inter-slice edges present in the 2-TBN.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_edges_from([(('D', 0), ('G', 0)), (('I', 0), ('G', 0)),
        ...                     (('G', 0), ('L', 0)), (('D', 0), ('D', 1)),
        ...                     (('I', 0), ('I', 1)), (('G', 0), ('G', 1)),
        ...                     (('G', 0), ('L', 1)), (('L', 0), ('L', 1))])
        >>> dbn.get_inter_edges()
        [(('D', 0), ('D', 1)),
         (('G', 0), ('G', 1)),
         (('G', 0), ('L', 1)),
         (('I', 0), ('I', 1)),
         (('L', 0), ('L', 1))]
        """
        return [edge for edge in self.edges() if edge[0][1] != edge[1][1]]

    def get_interface_nodes(self, time_slice=0):
        """
        Returns the nodes in the first timeslice whose children are present in the first timeslice.

        Parameters
        ----------
        time_slice:int
                The timeslice should be a positive value greater than or equal to zero

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D', 'G', 'I', 'S', 'L'])
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('G',0),('L',0)),(('D',0),('D',1))])
        >>> dbn.get_interface_nodes()
        [('D', 0)]
        """
        if not isinstance(time_slice, int) or time_slice < 0:
            raise ValueError(
                f"The timeslice should be a positive integer greater than or equal to zero: ({type(time_slice)}, value: {time_slice})"
            )

        return [
            DynamicNode(edge[time_slice][0], edge[time_slice][1])
            for edge in self.get_inter_edges()
        ]

    def get_slice_nodes(self, time_slice=0):
        """
        Returns the nodes present in a particular timeslice

        Parameters
        ----------
        time_slice:int
                The timeslice should be a positive value greater than or equal to zero

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D', 'G', 'I', 'S', 'L'])
        >>> dbn.add_edges_from([(('D', 0),('G', 0)),(('I', 0),('G', 0)),(('G', 0),('L', 0)),(('D', 0),('D', 1))])
        >>> dbn.get_slice_nodes()
        """
        if not isinstance(time_slice, int) or time_slice < 0:
            raise ValueError(
                "The timeslice should be a positive value greater than or equal to zero"
            )

        return [DynamicNode(node, time_slice) for node in self._nodes()]

    def add_cpds(self, *cpds):
        """
        This method adds the cpds to the dynamic bayesian network.
        Note that while adding variables and the evidence in cpd,
        they have to be of the following form
        (node_name, time_slice)
        Here, node_name is the node that is inserted
        while the time_slice is an integer value, which denotes
        the index of the time_slice that the node belongs to.

        Parameters
        ----------
        cpds : list, set, tuple (array-like)
            List of CPDs which are to be associated with the model. Each CPD
            should be an instance of `TabularCPD`.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> dbn = DBN()
        >>> dbn.add_edges_from([(('D', 0),('G', 0)),(('I', 0),('G', 0)),(('D', 0),('D', 1)),(('I', 0),('I', 1))])
        >>> grade_cpd = TabularCPD(('G', 0), 3, [[0.3, 0.05, 0.9, 0.5],
        ...                                      [0.4, 0.25, 0.8, 0.03],
        ...                                      [0.3, 0.7, 0.02, 0.2]],
        ...                        evidence=[('I', 0),('D', 0)],
        ...                        evidence_card=[2, 2])
        >>> d_i_cpd = TabularCPD(('D',1), 2, [[0.6, 0.3],
        ...                                   [0.4, 0.7]],
        ...                      evidence=[('D',0)],
        ...                      evidence_card=[2])
        >>> diff_cpd = TabularCPD(('D', 0), 2, [[0.6, 0.4]])
        >>> intel_cpd = TabularCPD(('I', 0), 2, [[0.7, 0.3]])
        >>> i_i_cpd = TabularCPD(('I', 1), 2, [[0.5, 0.4],
        ...                                    [0.5, 0.6]],
        ...                      evidence=[('I', 0)],
        ...                      evidence_card=[2])
        >>> dbn.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
        >>> dbn.get_cpds()
        [<TabularCPD representing P(('G', 0):3 | ('I', 0):2, ('D', 0):2) at 0x7ff7f27b0cf8>,
         <TabularCPD representing P(('D', 1):2 | ('D', 0):2) at 0x7ff810b9c2e8>,
         <TabularCPD representing P(('D', 0):2) at 0x7ff7f27e6f98>,
         <TabularCPD representing P(('I', 0):2) at 0x7ff7f27e6ba8>,
         <TabularCPD representing P(('I', 1):2 | ('I', 0):2) at 0x7ff7f27e6668>]
        """
        for cpd in cpds:
            if not isinstance(cpd, TabularCPD):
                raise ValueError("cpd should be an instance of TabularCPD")

            if set(cpd.variables) - set(cpd.variables).intersection(
                set(super(DynamicBayesianNetwork, self).nodes())
            ):
                raise ValueError("CPD defined on variable not in the model", cpd)

        self.cpds.extend(cpds)

    def get_cpds(self, node=None, time_slice=None):
        """
        Returns the CPDs that have been associated with the network.

        Parameters
        ----------
        node: tuple (node_name, time_slice)
            The node should be in the following form (node_name, time_slice).
            Here, node_name is the node that is inserted while the time_slice is
            an integer value, which denotes the index of the time_slice that the
            node belongs to.

        time_slice: int
            The time_slice should be a positive integer greater than or equal to zero.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> dbn = DBN()
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
        >>> grade_cpd =  TabularCPD(('G',0), 3, [[0.3,0.05,0.9,0.5],
        ...                                      [0.4,0.25,0.8,0.03],
        ...                                      [0.3,0.7,0.02,0.2]], [('I', 0),('D', 0)],[2,2])
        >>> dbn.add_cpds(grade_cpd)
        >>> dbn.get_cpds()
        """

        if time_slice is None:
            time_slices = self._timeslices()
        elif isinstance(time_slice, int) and time_slice >= 0:
            time_slices = [time_slice]
        elif isinstance(time_slice, typing.Iterable):
            if all(isinstance(n, int) for n in time_slice):
                time_slices = time_slice
            else:
                raise ValueError(
                    "At least one element inside time_slice interable is not positive and/or integer"
                )
        else:
            raise ValueError(
                "Time slice is not a positive integer neither a interable of integers"
            )

        if node:
            if node not in super(DynamicBayesianNetwork, self).nodes():
                raise ValueError("Node not present in the model.")
            else:
                for cpd in self.cpds:
                    if cpd.variable == node:
                        return cpd
        else:
            return_cpds = []
            for time_slice in time_slices:
                for var in self.get_slice_nodes(time_slice=time_slice):
                    cpd = self.get_cpds(node=var)
                    if cpd:
                        return_cpds.append(cpd)
            return return_cpds

    def remove_cpds(self, *cpds):
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds : list, set, tuple (array-like)
            List of CPDs which are to be associated with the model. Each CPD
            should be an instance of `TabularCPD`.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> dbn = DBN()
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
        >>> grade_cpd =  TabularCPD(('G',0), 3, [[0.3,0.05,0.9,0.5],
        ...                                      [0.4,0.25,0.8,0.03],
        ...                                      [0.3,0.7,0.02,0.2]], [('I', 0),('D', 0)],[2,2])
        >>> dbn.add_cpds(grade_cpd)
        >>> dbn.get_cpds()
        [<TabularCPD representing P(('G', 0):3 | ('I', 0):2, ('D', 0):2) at 0x3348ab0>]
        >>> dbn.remove_cpds(grade_cpd)
        >>> dbn.get_cpds()
        []
        """
        for cpd in cpds:
            if isinstance(cpd, (tuple, list)):
                cpd = self.get_cpds(cpd)
            self.cpds.remove(cpd)

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if the sum of the probabilities in each associated CPD for each
            state is equal to 1 (tol=0.01).
        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        boolean: True if everything seems to be order. Otherwise raises error
            according to the problem.
        """
        for node in super(DynamicBayesianNetwork, self).nodes():
            cpd = self.get_cpds(node=node)
            if isinstance(cpd, TabularCPD):
                evidence = cpd.variables[:0:-1]
                evidence_card = cpd.cardinality[:0:-1]
                parents = self.get_parents(node)
                if set(evidence) != set(parents if parents else []):
                    raise ValueError(
                        f"CPD associated with {node} doesn't have proper parents associated with it."
                    )
                if not np.allclose(
                    cpd.to_factor()
                    .marginalize([node], inplace=False)
                    .values.flatten("C"),
                    np.ones(np.product(evidence_card)),
                    atol=0.01,
                ):
                    raise ValueError(
                        f"Sum of probabilities of states for node {node} is not equal to 1"
                    )
        return True

    def initialize_initial_state(self):
        """
        This method will automatically re-adjust the cpds and the edges added to the bayesian network.
        If an edge that is added as an intra time slice edge in the 0th timeslice, this method will
        automatically add it in the 1st timeslice. It will also add the cpds. However, to call this
        method, one needs to add cpds as well as the edges in the bayesian network of the whole
        skeleton including the 0th and the 1st timeslice,.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> student = DBN()
        >>> student.add_nodes_from(['D', 'G', 'I', 'S', 'L'])
        >>> student.add_edges_from([(('D', 0),('G', 0)),(('I', 0),('G', 0)),(('D', 0),('D', 1)),(('I', 0),('I', 1))])
        >>> grade_cpd = TabularCPD(('G', 0), 3, [[0.3, 0.05, 0.9, 0.5],
        ...                                      [0.4, 0.25, 0.08, 0.3],
        ...                                      [0.3, 0.7, 0.02, 0.2]],
        ...                        evidence=[('I', 0),('D', 0)],
        ...                        evidence_card=[2, 2])
        >>> d_i_cpd = TabularCPD(('D', 1), 2, [[0.6, 0.3],
        ...                                    [0.4, 0.7]],
        ...                      evidence=[('D', 0)],
        ...                      evidence_card=[2])
        >>> diff_cpd = TabularCPD(('D', 0), 2, [[0.6, 0.4]])
        >>> intel_cpd = TabularCPD(('I',0), 2, [[0.7, 0.3]])
        >>> i_i_cpd = TabularCPD(('I', 1), 2, [[0.5, 0.4],
        ...                                    [0.5, 0.6]],
        ...                      evidence=[('I', 0)],
        ...                      evidence_card=[2])
        >>> student.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
        >>> student.initialize_initial_state()
        """
        for cpd in self.cpds:
            temp_var = DynamicNode(cpd.variable[0], 1 - cpd.variable[1])
            parents = self.get_parents(temp_var)
            if not any(x.variable == temp_var for x in self.cpds):
                if all(x[1] == parents[0][1] for x in parents):
                    if parents:
                        evidence_card = cpd.cardinality[:0:-1]
                        new_cpd = TabularCPD(
                            temp_var,
                            cpd.variable_card,
                            cpd.values.reshape(
                                cpd.variable_card, np.prod(evidence_card)
                            ),
                            parents,
                            evidence_card,
                        )
                    else:
                        if cpd.get_evidence():
                            initial_cpd = cpd.marginalize(
                                cpd.get_evidence(), inplace=False
                            )
                            new_cpd = TabularCPD(
                                temp_var,
                                cpd.variable_card,
                                np.reshape(initial_cpd.values, (2, -1)),
                            )
                        else:
                            new_cpd = TabularCPD(
                                temp_var,
                                cpd.variable_card,
                                np.reshape(cpd.values, (2, -1)),
                            )
                    self.add_cpds(new_cpd)
            self.check_model()

    def moralize(self):
        """
        Removes all the immoralities in the Network and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN([(('D',0), ('G',0)), (('I',0), ('G',0))])
        >>> moral_graph = dbn.moralize()
        >>> moral_graph.edges()
        EdgeView([(('G', 0), ('I', 0)),
                  (('G', 0), ('D', 0)),
                  (('D', 1), ('I', 1)),
                  (('D', 1), ('G', 1)),
                  (('I', 0), ('D', 0)),
                  (('G', 1), ('I', 1))])
        """
        moral_graph = self.to_undirected()

        for node in super(DynamicBayesianNetwork, self).nodes():
            moral_graph.add_edges_from(combinations(self.get_parents(node), 2))

        return moral_graph

    def copy(self):
        """
        Returns a copy of the dynamic bayesian network.

        Returns
        -------
        DynamicBayesianNetwork: copy of the dynamic bayesian network

        Examples
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> dbn = DBN()
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
        >>> grade_cpd =  TabularCPD(('G',0), 3, [[0.3, 0.05, 0.9,  0.5 ],
        ...                                      [0.4, 0.25, 0.8,  0.03],
        ...                                      [0.3,  0.7, 0.02, 0.2 ]],
        ...                         [('I', 0), ('D', 0)],[2,2])
        >>> dbn.add_cpds(grade_cpd)
        >>> dbn_copy = dbn.copy()
        >>> dbn_copy.nodes()
        ['Z', 'G', 'I', 'D']
        >>> dbn_copy.edges()
        [(('I', 1), ('G', 1)),
        (('I', 0), ('I', 1)),
        (('I', 0), ('G', 0)),
        (('D', 1), ('G', 1)),
        (('D', 0), ('G', 0)),
        (('D', 0), ('D', 1))]
        >>> dbn_copy.get_cpds()
        [<TabularCPD representing P(('G', 0):3 | ('I', 0):2, ('D', 0):2) at 0x7f13961a3320>]
        """
        dbn = DynamicBayesianNetwork()
        dbn.add_nodes_from(self._nodes())
        edges = [(u.to_tuple(), v.to_tuple()) for (u, v) in self.edges()]
        dbn.add_edges_from(edges)
        cpd_copy = [cpd.copy() for cpd in self.get_cpds()]
        dbn.add_cpds(*cpd_copy)
        return dbn

    def get_markov_blanket(self, node):
        # Wrap node into DynamicNode
        if not isinstance(node, DynamicNode):
            node = DynamicNode(*node)
        # Get standard Markov blanket
        markov_blanket = set(
            super(DynamicBayesianNetwork, self).get_markov_blanket(node)
        )

        # Augment Markov blanket:
        # if node is in the last time slice, unroll and add children nodes from next time slice
        max_ts = max([n.time_slice for n in self.nodes()])
        if node.time_slice == max_ts:
            # Move node to previous time slice and get children
            temp_children = self.get_children(
                DynamicNode(node.node, node.time_slice - 1)
            )
            # Move children to next time slice
            next_children = {
                DynamicNode(child.node, child.time_slice + 1) for child in temp_children
            }
            # Get children parents
            next_parents = set(
                chain(*[self.get_parents(child) for child in temp_children])
            )
            # Get children's parents
            temp_parents = {
                parent for child in temp_children for parent in self.get_parents(child)
            }
            # Move children's parents to next time slice
            next_parents = {
                DynamicNode(parent.node, parent.time_slice + 1)
                for parent in temp_parents
            }
            # Add them to Markov blanket
            markov_blanket = markov_blanket | next_children
            markov_blanket = markov_blanket | next_parents

        return sorted(markov_blanket)

    def get_constant_bn(self):
        """
        Returns a normal bayesian network object which has nodes from the first two
        time slices and all the edges in the first time slice and edges going from
        first to second time slice. The returned bayesian network bascially represents
        the part of the DBN which remains constant.
        """
        from pgmpy.models import BayesianNetwork

        return BayesianNetwork(
            ebunch=[(f"{u[0]}_{u[1]}", f"{v[0]}_{v[1]}") for (u, v) in self.edges()]
        )

    def fit(self, data, estimator="MLE"):
        """
        Learns the CPD of the model from data.

        Since the assumption is that the 2-TBN stays constant throughtout the model,
        the algorithm iterates over every 2 consecutive time slices in the data and
        updates the CPDs based on it.

        Parameters
        ----------
        data: pandas.DataFrame instance
            The column names must be of the form (variable, time_slice). The
            time-slices must start from 0.

        estimator: str
            Currently only Maximum Likelihood Estimator is supported.

        Returns
        -------
        None: The CPDs are added to the model instance.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> model = DBN(
        >>>     [
        >>>         (("A", 0), ("B", 0)),
        >>>         (("A", 0), ("C", 0)),
        >>>         (("B", 0), ("D", 0)),
        >>>         (("C", 0), ("D", 0)),
        >>>         (("A", 0), ("A", 1)),
        >>>         (("B", 0), ("B", 1)),
        >>>         (("C", 0), ("C", 1)),
        >>>         (("D", 0), ("D", 1)),
        >>>     ]
        >>> )
        >>> data = np.random.randint(low=0, high=2, size=(1000, 20))
        >>> colnames = []
        >>> for t in range(5):
        ...     colnames.extend([("A", t), ("B", t), ("C", t), ("D", t)])
        >>> df = pd.DataFrame(data, columns=colnames)
        >>> model.fit(df)
        """
        if not isinstance(data, pd.DataFrame):
            raise ValueError(f"data must be a pandas dataframe. Got: {type(data)}")

        if min(data.columns, key=lambda t: t[1])[1] != 0:
            raise ValueError("data column names must start from time slice 0.")

        if estimator not in {"MLE", "mle"}:
            raise ValueError("Only Maximum Likelihood Estimator is supported currently")

        n_samples = data.shape[0]
        const_bn = self.get_constant_bn()
        n_time_slices = max(data.columns, key=lambda t: t[1])[1]

        for t_slice in range(n_time_slices):
            colnames = [(node, t_slice) for node in self._nodes()]
            colnames.extend([(node, t_slice + 1) for node in self._nodes()])

            df_slice = data.loc[:, colnames]
            new_colnames = [f"{node}_{t - t_slice}" for (node, t) in df_slice.columns]
            df_slice.columns = new_colnames
            if t_slice == 0:
                const_bn.fit(df_slice)
            else:
                const_bn.fit_update(df_slice, n_prev_samples=t_slice * n_samples)

        cpds = []
        for cpd in const_bn.cpds:
            variables = [var.split("_") for var in cpd.variables]
            variables = [(v, int(t)) for v, t in variables]
            cpds.append(
                TabularCPD(
                    variable=variables[0],
                    variable_card=cpd.variable_card,
                    values=cpd.get_values(),
                    evidence=variables[1:],
                    evidence_card=cpd.cardinality[1:],
                )
            )

        self.add_cpds(*cpds)

    def active_trail_nodes(self, variables, observed=None, include_latents=False):
        if not isinstance(variables, DynamicNode):
            # Wrap variables in DynamicNode objects
            if len(variables) == 2 and isinstance(variables[1], int):
                variables = DynamicNode(*variables)
            else:
                variables = [DynamicNode(*v) for v in variables]
        if (
            observed is not None
            and len(observed) > 0
            and any([not isinstance(o, DynamicNode) for o in observed])
        ):
            # Wrap observed in DynamicNode objects
            if len(observed) == 2 and isinstance(observed[1], int):
                observed = DynamicNode(*observed)
            else:
                observed = [DynamicNode(*o) for o in observed]
        # Call super method
        return super(DynamicBayesianNetwork, self).active_trail_nodes(
            variables, observed, include_latents
        )
