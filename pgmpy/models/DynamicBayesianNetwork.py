from copy import deepcopy

import numpy as np

from pgmpy.models import BayesianModel


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

        Most of the methods will be imported from Bayesian Model

        >>> dbn.nodes()
        ['L', 'G', 'S', 'I', 'D']
        >>> dbn.edges()
        [(('D', 0), ('G', 0)), (('G', 0), ('L', 0)), (('I', 0), ('G', 0))]

        If some edges connect nodes not yet in the model, the nodes
        are added automatically. There are no errors when adding
        nodes or edges that already exist.

        Methods:
        -------
        add_nodes_from
        add_edge
        add_edges_from
        intra_slice
        inter_slice
        add_cpds
        initialize_initial_state
        """
        super().__init__()
        if ebunch:
            self.add_nodes_from(ebunch)

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
        >>> dbn.add_nodes_from(['D', 'I'])
        >>> dbn.add_edge(('D',0), ('I',0))
        >>> dbn.edges()
        [(('D', 0), ('I', 0))]
        """
        if not (isinstance(start, (tuple, list)) and  isinstance(end, (tuple, list))): 
            raise ValueError('the nodes inside the edge must be enclosed in a list or a tuple')

        if not (start[1] in (0,1) and end[1] in (0,1)):
            raise ValueError('the timeslices inside the node must belong to 0 or 1')

        super().add_edge(start, end)
        self.correct_nodes()

    def correct_nodes(self):
        """
        This method automatically adjusts the nodes inside the 
        Bayesian Network automatically and is used 
        inside the add_edge() method

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN()
        >>> dbn.add_nodes_from(['D', 'I'])
        >>> dbn.add_edge(('D',0), ('I',0))
        >>> dbn.edges()
        [(('D', 0), ('I', 0))]
        >>> dbn.nodes()
        ['I', 'D']
        """
        temp_dict = deepcopy(self.node)
        for node in temp_dict:
            if isinstance(node, (tuple, list)):
                temp_value = self.node[node]
                self.node.pop(node)
                if node[0] not in self.node:
                    self.node[node[0]] = temp_value

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
        This method adds the cpds inside the dynamic bayesian network.
        Note that while adding variables and the evidence in cpd,
        they have to be of the following form
        (node_name, time_slice)

        Here, node_name is the node that is inserted
        while the time_slice is an integer value, which denotes
        the index of the time_slice that the node belongs to.

        EXAMPLE
        -------
        >>> from pgmpy.models import DynamicBayesianNetwork as DBN
        >>> dbn = DBN(['D','G','I','S','L'])
        >>> dbn.add_edges_from([(('D',0),('G',0)),(('I',0),('G',0)),(('D',0),('D',1)),(('I',0),('I',1))])
        >>> grade_cpd = grade_cpd = TabularCPD(('G',0), 3, [[0.3,0.05,0.9,0.5],
        ...                                                 [0.4,0.25,0.8,0.03],
        ...                                                 [0.3,0.7,0.02,0.2]], [('I', 0),('D', 0)],[2,2])
        >>> d_i_cpd = TabularCPD(('D',1),2,[[0.6,0.3],[0.4,0.7]],[('D',0)],2)
        >>> diff_cpd = TabularCPD(('D',0),2,[[0.6,0.4]])
        >>> intel_cpd = TabularCPD(('I',0),2,[[0.7,0.3]])
        >>> i_i_cpd = TabularCPD(('I',1),2,[[0.5,0.4],[0.5,0.6]],[('I',0)],2)
        >>> dbn.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
        >>> dbn.cpds
        [<TabularCPD representing P(('G', 0):3 | ('I', 0):2, ('D', 0):2) at 0xb9b5b30>,
        <TabularCPD representing P(('D', 0):2) at 0xb9728b0>,
        <TabularCPD representing P(('I', 0):2) at 0xb972ff0>,
        <TabularCPD representing P(('D', 1):2 | ('D', 0):2) at 0xb972fd0>,
        <TabularCPD representing P(('I', 1):2 | ('I', 0):2) at 0xb972b50>]
        """
        super().add_cpds(*cpds)

    def initialize_initial_state(self):
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
        >>> grade_cpd = TabularCPD(('G',0), 3, [[0.3,0.05,0.9,0.5],
        ...                                                 [0.4,0.25,0.8,0.03],
        ...                                                 [0.3,0.7,0.02,0.2]], [('I', 0),('D', 0)],[2,2])
        >>> d_i_cpd = TabularCPD(('D',1),2,[[0.6,0.3],[0.4,0.7]],[('D',0)],2)
        >>> diff_cpd = TabularCPD(('D',0),2,[[0.6,0.4]])
        >>> intel_cpd = TabularCPD(('I',0),2,[[0.7,0.3]])
        >>> i_i_cpd = TabularCPD(('I',1),2,[[0.5,0.4],[0.5,0.6]],[('I',0)],2)
        >>> student.add_cpds(grade_cpd, d_i_cpd, diff_cpd, intel_cpd, i_i_cpd)
        >>> student.initialize_initial_state()
        """

        self.add_edges_from([list((a, 1-b) for a,b in group) for group in self.get_intra_edges()])
        for cpd in self.cpds:
            temp_var = (cpd.variable[0], 1 - cpd.variable[1])
            parents = self.get_parents(temp_var)
            if not any(x.variable == temp_var for x in self.cpds):
                if all(x[1] == parents[0][1] for x in parents):
                    if parents:
                        new_cpd = TabularCPD(temp_var, cpd.variable_card, np.split(cpd.values, cpd.variable_card), parents,
                         cpd.evidence_card) 
                    else:
                        new_cpd = TabularCPD(temp_var, cpd.variable_card, np.split(cpd.values, cpd.variable_card)) 
                self.cpds.append(new_cpd)
        self.check_model()
