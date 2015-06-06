
import numpy as np
from pgmpy.models import BayesianModel
from pgmpy.factors import TabularCPD, TreeCPD, RuleCPD
from collections import OrderedDict, namedtuple


edge_type = namedtuple('edge_type', ('start', 'end'))


class DynamicBayesianNetwork(BayesianModel):
    def __init__(self, ebunch=None):
        """
        Base class for Dynamic Bayesian Network

        This model is a time variant of the static Bayesian model, where each timeslice has
        some static nodes and is then replicated over a certain time slice.

        The nodes can be hashable python objects.

        However, the hidden nodes will compulsory have the following form.
        (node_name, time_slice)

        Here, node_name is the node that is inserted
        while the time_slice is an integer value, which denotes
        the index of the time_slice that the node belongs to.

        Edges are represented as links between the nodes.

        Parameters:
        ----------
        data: Data to initialize graph.  If data=None (default) an empty
              graph is created.  The data can be an edge list, or any
              NetworkX graph object

        Examples:
        --------
        Create an empty Dynamic Bayesian Network with no nodes and no edges
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        adding nodes and edges inside the dynamic bayesian network. A single
        node can be added using the method below.
        >>> dbn.add_nodes_from(['D','G','I','S','L'])
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('G',0),('L',0))])

        most of the methods will be imported from Bayesian Model
        However there will be a few changes
        >>> dbn.list_of_nodes
        ['D','G','I','S','L']
        >>> dbn.edges()
        [(('D',0),('G',0)),(('I',0),('G',0)),(('G',0),('L',0))]
        If some edges connect nodes not yet in the model, the nodes
        are added automatically.  There are no errors when adding
        nodes or edges that already exist.

        Methods:
        -------
        add_nodes_from
        add_edge
        add_edges_from
        intra_slice
        inter_slice
        add_cpds
        get_cpds
        compute_initial_state
        unroll
        remove_timeslice
        """
        super().__init__()
        self.inter_edges = []
        self.intra_edges = []
        self.dbn_cpds = OrderedDict()
        self.list_of_nodes = []
        if ebunch:
            self.add_nodes_from(ebunch)
        self.timestate = 1
        self.computed = False

    def add_nodes_from(self, nodes):
        """
        adding nodes in the static Bayesian Network.

        Parameters:
        ----------
        nodes: nodes
               Nodes can be any hashable python object.

        EXAMPLE
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D','G','I','S','L'])
        """
        for node in nodes:
            self.list_of_nodes.extend(node)

    def add_edge(self, start, end, **kwargs):
        """
        Add an edge between two nodes.

        The nodes will be automatically added if they are
        not already in the initial static Bayesian Network.

        Parameters
        ----------
        start, end: The start, end nodes should contain the
                    (node_name, time_slice)
                    Here, node_name can be a hashable python object
                    while the time_slice is an integer value, which
                    denotes the index of the time_slice that the node
                    belongs to.

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['d', 'i'])
        >>> dbn.add_edge((('d',0), ('i',0)))
        """
        edge = (start, end)
        if not any(map(lambda x: isinstance(x, tuple), iter(edge))):
            raise AttributeError('The edge is not of a standard form')

        if not any(map(lambda x: isinstance(x[1], int), iter(edge))):
            raise AttributeError('The edge does not contain the timeslice in integer form')

        if not any(map(lambda x: x[0] in self.list_of_nodes, iter(edge))):
            new_nodes = list(filter(lambda x: x[0] not in self.list_of_nodes, iter(edge)))
            self.add_nodes_from(set(map(lambda x: x[0], iter(new_nodes))))

        if edge[0][1] == edge[1][1]:
            if edge[0][1] in (0, 1):
                new_edge = map(lambda x: (x[0], 1 - edge[0][1]), edge)
                super().add_edge(*edge_type(*new_edge))
                self.intra_edges.extend([edge_type(*edge), edge_type(*map(lambda x: (x[0], 1 - edge[0][1]), edge))])
            else:
                self.intra_edges.extend([edge_type(*edge)])
        else:
            self.inter_edges.append(edge_type(*edge))

        super().add_edge(*edge_type(*edge))

    def add_edges_from(self, ebunch, **kwargs):
        """
        Adding multiple edges in a single list
        Extension of the add_edge method

        Parameters:
        ----------
        List of edges: Each edge should consist of two nodes
                       where each node is of the form
                       (node_name, time_slice)
                       Here, node_name can be a hashable 
                       python object while the time_slice is an
                       integer value, which denotes the index of
                       the time_slice that the node belongs to.

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D','G','I','S','L'])
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('G',0),('L',0))])
        """
        for edge in ebunch:
            self.add_edge(edge[0], edge[1])

    def intra_slice(self, timeslice):
        """
        returns the intra slice edges present in a single time slice.

        Parameters:
        ----------
        timeslice: integer value. should be ranging from 0
                   to the number of the slices present in
                   the bayesian network.

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D','G','I','S','L'])
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('G',0),('L',0))])
        >>> dbn.intra_slice(0)
        [edge_type(start=('D', 0), end=('G', 0)),
        edge_type(start=('I', 0), end=('G', 0)),
        edge_type(start=('G', 0), end=('L', 0))]
        """
        if not isinstance(timeslice, int):
            raise ValueError('the timeslice is not an integer value')
        if timeslice < 0:
            raise ValueError('negative values for a timeslice are not permissible')
        if timeslice > self.timestate:
            raise ValueError('the bayesian network does not contain the intra slice edges of this timeslice')
        return list(filter(lambda x: x.start[1] == x.end[1] == timeslice, self.intra_edges))

    def inter_slice(self, timeslice):
        """
        returns the inter-slice edges in which the edges end in
        the given time slice

        Parameters:
        ----------
        timeslice: integer value. should be ranging from 0
                   to the number of the slices present in
                   the bayesian network.

        EXAMPLE:
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D','G','I','S','L'])
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1)))])
        >>> dbn.inter_slice(1)
        [edge_type(start=('D', 0), end=('D', 1)),
        edge_type(start=('I', 0), end=('I', 1))]
        """
        if not isinstance(timeslice, int):
            raise ValueError('the timeslice is not an integer value')
        if timeslice < 0:
            raise ValueError('negative values for a timeslice are not permissible')
        if timeslice > self.timestate:
            raise ValueError('the bayesian network does not contain the inter slice edges of this timeslice')
        return list(filter(lambda x: x.end[1] == timeslice, self.inter_edges))

    def add_dbn_cpds(self, timeslice, *cpds):
        """
        Add CPD (Conditional Probability Distribution) to the Bayesian Model.

        Parameters
        ----------
        cpds  :  list, set, tuple (array-like)
            List of cpds (TabularCPD, TreeCPD, RuleCPD, Factor)
            which will be associated with the model

        timeslice :integer value. should be ranging from 0
                   to the number of the slices present in
                   the bayesian network.

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors.CPD import TabularCPD
        >>> student = DBN('D','G','I','S','L')
        >>> grades_cpd = TabularCPD('G', 3, [[0.1,0.1,0.1,0.1,0.1,0.1],
        ...                                  [0.1,0.1,0.1,0.1,0.1,0.1],
        ...                                  [0.8,0.8,0.8,0.8,0.8,0.8]],
        ...                         evidence=['D', 'I'], evidence_card=[2, 3])
        >>> student.add_dbn_cpds(0, grades_cpd)

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
        for cpd in cpds:
            if not isinstance(cpd, (TabularCPD, TreeCPD, RuleCPD)):
                raise ValueError('Only TabularCPD, TreeCPD or RuleCPD can be'
                                 ' added.')

            if (cpd.variable, timeslice) not in self.dbn_cpds:
                self.dbn_cpds[(cpd.variable, timeslice)] = cpd

    def get_dbn_cpds(self, node=None, timeslice=None):
        """
        Returns the cpds that have been added till now to the graph

        Parameters
        ---------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        timeslice: integer value.
                   should be ranging from 0 to the number of the
                   slices present in the bayesian network.
    
        EXAMPLE
        --------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors import TabularCPD
        >>> student = DBN('D','G','I','S','L')
        >>> grade_cpd = TabularCPD('G', 3, [[0.3, 0.05, 0.9, 0.5],[0.4, 0.25, 0.08, 0.3],[0.3, 0.7, 0.02, 0.2]], ['I','D'],[2,2])
        >>> student.add_dbn_cpds(0, grade_cpd)
        >>> student.get_dbn_cpds('G',0)
        ╒═════╤═════╤══════╤══════╤═════╕
        │ D   │ D_0 │ D_0  │ D_1  │ D_1 │
        ├─────┼─────┼──────┼──────┼─────┤
        │ I   │ I_0 │ I_1  │ I_0  │ I_1 │
        ├─────┼─────┼──────┼──────┼─────┤
        │ G_0 │ 0.3 │ 0.05 │ 0.9  │ 0.5 │
        ├─────┼─────┼──────┼──────┼─────┤
        │ G_1 │ 0.4 │ 0.25 │ 0.08 │ 0.3 │
        ├─────┼─────┼──────┼──────┼─────┤
        │ G_2 │ 0.3 │ 0.7  │ 0.02 │ 0.2 │
        ╘═════╧═════╧══════╧══════╧═════╛
        """
        if node:
            if timeslice:
                if (node,timeslice) in self.dbn_cpds:
                    return self.dbn_cpds[(node, timeslice)]
            else:
                list_of_cpds = []
                for time_state in range(self.timestate + 1):
                    list_of_cpds.append(self.dbn_cpds[(node, time_state)])
                return list_of_cpds
        else:
            return self.dbn_cpds

    def compute_initial_state(self):
        """
        This method will automatically re-adjust the cpds and the edges added to the bayesian network.
        If an edge that is added as an intra time slice edge in the 0th timeslice, this method will automatically
        add it in the 1st timeslice. It will also add the cpds.
        However, to call this method, one needs to add cpds as well as the edges in the
        bayesian network of the whole skeleton including the 0th and the 1st timeslice,.

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors import TabularCPD
        >>> student = DBN()
        >>> student.add_nodes_from(['D','G','I','S','L'])
        >>> student.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
        >>> grade_cpd = TabularCPD('G', 3, [[0.3, 0.05, 0.9, 0.5],
        ...                                 [0.4, 0.25, 0.08, 0.3],
        ...                                 [0.3, 0.7, 0.02, 0.2]], ['I','D'],[2,2])
        >>> intel_cpd = TabularCPD('I', 2, [[0.9, 0.4],[0.1, 0.6]],['I'],[2])
        >>> diff_cpd = TabularCPD('D', 2, [[0.3, 0.2],[0.7, 0.8]], ['D'],[2])
        >>> d_cpd = TabularCPD('D', 2, [[0.6, 0.4]])
        >>> i_cpd = TabularCPD('I', 2, [[0.7, 0.3]])
        >>> student.add_dbn_cpds(0, grade_cpd, d_cpd, i_cpd)
        >>> student.add_dbn_cpds(1, diff_cpd, intel_cpd)
        >>> student.compute_initial_state()
        """

        if not self.computed:
            for node in self.dbn_cpds:
                list_of_parents = super().get_parents(node)
                if all(x[1] == list_of_parents[0][1] for x in list_of_parents) and list_of_parents != [] and (
                        node[0], 1 - node[1]) not in self.dbn_cpds:
                    list_of_new_parents = super().get_parents((node[0], 1 - node[1]))
                    if all(x[1] == list_of_new_parents[0][1] for x in
                           list_of_new_parents) and list_of_new_parents != []:
                        self.dbn_cpds[(node[0], 1 - node[1])] = self.dbn_cpds[node]
                cpd = self.dbn_cpds[node]
                if list_of_parents:
                    self.dbn_cpds[node] = TabularCPD(node, cpd.variable_card, np.split(cpd.values, cpd.variable_card),
                                                list_of_parents, cpd.evidence_card)
                else:
                    self.dbn_cpds[node] = TabularCPD(node, cpd.variable_card, np.split(cpd.values, cpd.variable_card))
                super().add_cpds(self.dbn_cpds[node])

            super().check_model()

            if len(self.dbn_cpds) != len(super().nodes()):
                raise ValueError('all the cpds have not been added yet')
        self.computed = True

    def unroll(self, number_of_time_slices):
        """
        This means that the Bayesian Network will have the added cpds
        and the edges
        The assumption over here is that the number_of_time_slices will be the time slices added to the Bayesian Network

        Parameters:
        -----------
        number_of_time_slices: int value
                               number of time slices that will
                               be added to the bayesian network

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors import TabularCPD
        >>> student = DBN()
        >>> student.add_nodes_from(['D','G','I','S','L'])
        >>> student.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
        >>> grade_cpd = TabularCPD('G', 3, [[0.3, 0.05, 0.9, 0.5],
        ...                                 [0.4, 0.25, 0.08, 0.3],
        ...                                 [0.3, 0.7, 0.02, 0.2]], ['I','D'],[2,2])
        >>> intel_cpd = TabularCPD('I', 2, [[0.9, 0.4],[0.1, 0.6]],['I'],[2])
        >>> diff_cpd = TabularCPD('D', 2, [[0.3, 0.2],[0.7, 0.8]], ['D'],[2])
        >>> d_cpd = TabularCPD('D', 2, [[0.6, 0.4]])
        >>> i_cpd = TabularCPD('I', 2, [[0.7, 0.3]])
        >>> student.add_dbn_cpds(0, grade_cpd, d_cpd, i_cpd)
        >>> student.add_dbn_cpds(1, diff_cpd, intel_cpd)
        >>> student.unroll(2)
        >>> student.edges()
        [(('I', 0), ('I', 1)),
        (('I', 0), ('G', 0)),
        (('D', 3), ('G', 3)),
        (('D', 2), ('D', 3)),
        (('D', 2), ('G', 2)),
        (('I', 3), ('G', 3)),
        (('D', 1), ('D', 2)),
        (('D', 1), ('G', 1)),
        (('I', 1), ('I', 2)),
        (('I', 1), ('G', 1)),
        (('D', 0), ('D', 1)),
        (('D', 0), ('G', 0)),
        (('I', 2), ('I', 3)),
        (('I', 2), ('G', 2))]
         """

        self.compute_initial_state()
        start = 1
        while start <= number_of_time_slices:
            new_edges = [edge_type(*((a, b + start) for a, b in group)) for group in
                         self.intra_slice(self.timestate) + self.inter_slice(self.timestate)]
            self.add_edges_from(new_edges)
            for node in self.dbn_cpds:
                if node[1] == 1:
                    new_cpd_variable = (node[0], start + self.timestate)
                    new_cpd_evidence = super().get_parents(new_cpd_variable)
                    new_cpd = TabularCPD(new_cpd_variable, self.dbn_cpds[node].variable_card,
                                         np.split(self.dbn_cpds[node].values, self.dbn_cpds[node].variable_card),
                                         new_cpd_evidence,
                                         self.dbn_cpds[node].evidence_card)
                    super().add_cpds(new_cpd)
                    self.dbn_cpds[new_cpd_variable] = new_cpd
            start += 1
        self.timestate += number_of_time_slices

    def remove_timeslice(self, number_of_time_slices):
        """
        removes the time slices that have been previously added to the bayesian network.

        Parameters
        ----------
        number_of_time_slices: int value
                               The number of time slices that will be removed
                               from the bayesian network.

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> from pgmpy.factors import TabularCPD
        >>> student = DBN()
        >>> student.add_nodes_from(['D','G','I','S','L'])
        >>> student.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
        >>> grade_cpd = TabularCPD('G', 3, [[0.3, 0.05, 0.9, 0.5],
        ...                                 [0.4, 0.25, 0.08, 0.3],
        ...                                 [0.3, 0.7, 0.02, 0.2]], ['I', 'D'],[2,2])
        >>> intel_cpd = TabularCPD('I', 2, [[0.9,0.4],[0.1,0.6]],['I'],[2])
        >>> diff_cpd = TabularCPD('D', 2, [[0.3,0.2],[0.7, 0.8]], ['D'],[2])
        >>> d_cpd = TabularCPD('D', 2, [[0.6,0.4]])
        >>> i_cpd = TabularCPD('I', 2, [[0.7,0.3]])
        >>> student.add_dbn_cpds(0, grade_cpd, d_cpd, i_cpd)
        >>> student.add_dbn_cpds(1, diff_cpd, intel_cpd)
        >>> student.unroll(3)
        >>> student.edges()
        [(('I', 0), ('I', 1)),
        (('I', 0), ('G', 0)),
        (('I', 4), ('G', 4)),
        (('D', 4), ('G', 4)),
        (('D', 3), ('G', 3)),
        (('D', 3), ('D', 4)),
        (('D', 2), ('D', 3)),
        (('D', 2), ('G', 2)),
        (('I', 3), ('G', 3)),
        (('I', 3), ('I', 4)),
        (('D', 1), ('D', 2)),
        (('D', 1), ('G', 1)),
        (('I', 1), ('I', 2)),
        (('I', 1), ('G', 1)),
        (('D', 0), ('D', 1)),
        (('D', 0), ('G', 0)),
        (('I', 2), ('I', 3)),
        (('I', 2), ('G', 2))]
        >>> student.remove_timeslice(2)
        >>> student.edges()
        [(('I', 0), ('I', 1)),
        (('I', 0), ('G', 0)),
        (('D', 2), ('G', 2)),
        (('D', 1), ('D', 2)),
        (('D', 1), ('G', 1)),
        (('I', 1), ('I', 2)),
        (('I', 1), ('G', 1)),
        (('D', 0), ('D', 1)),
        (('D', 0), ('G', 0)),
        (('I', 2), ('G', 2))]
        """
        start = 1
        if not self.timestate - number_of_time_slices:
            raise ValueError('complete removal of bayesian network is not allowed')
        while start <= number_of_time_slices:
            super().remove_edges_from(self.intra_slice(self.timestate) + self.inter_slice(self.timestate))
            list_of_nodes = list(filter(lambda x: x[1] == self.timestate, super().nodes()))
            list_of_cpds = [self.dbn_cpds[node] for node in list_of_nodes]
            super().remove_cpds(*list_of_cpds)
            for node in list_of_nodes:
                del self.dbn_cpds[node]
            super().remove_nodes_from(list_of_nodes)
            start += 1
            self.timestate -= 1
        if self.timestate == 0:
            self.computed = False
