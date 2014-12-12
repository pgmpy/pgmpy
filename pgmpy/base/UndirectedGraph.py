#!/usr/bin/env python3

from collections import defaultdict
from pgmpy.exceptions import CardinalityError
import abc
import itertools
import networkx as nx
import numpy as np


class UndirectedGraph(nx.Graph):
    """
    Base class for all the Undirected Graphical models.

    UndirectedGraph assumes that all the nodes in graph are either random
    variables, factors or cliques of random variables and edges in the graphs
    are interactions between these random variables, factors or clusters.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data can be an edge list or any Networkx graph object.

    Examples
    --------
    Create an empty UndirectedGraph with no nodes and no edges

    >>> from pgmpy.base import UndirectedGraph
    >>> G = UndirectedGraph()

    G can be grown in several ways

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
    """

    def __init__(self, ebunch=None):
        super(UndirectedGraph, self).__init__(ebunch)
        self.factors = []
        self.cardinalities = defaultdict(int)

    def add_node(self, node, **kwargs):
        """
        Add a single node to the Graph.

        Parameters
        ----------
        node: node
            A node can be any hashable Python object.

        See Also
        --------
        add_nodes_from : add a collection of nodes

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_node('A')
        """
        super(UndirectedGraph, self).add_node(node, **kwargs)

    def add_nodes_from(self, nodes, **kwargs):
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
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(['A', 'B', 'C'])
        """
        for node in nodes:
            self.add_node(node, **kwargs)

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u,v : nodes
            Nodes can be any hashable Python object.

        See Also
        --------
        add_edges_from : add a collection of edges

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edge('Alice', 'Bob')
        """
        super(UndirectedGraph, self).add_edge(u, v, **kwargs)

    def add_edges_from(self, ebunch, **kwargs):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        See Also
        --------
        add_edge : Add a single edge

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from([('Alice', 'Bob'), ('Bob', 'Charles')])
        """
        for edge in ebunch:
            self.add_edge(*edge, **kwargs)

    def _update_cardinalities(self, factors):
        """
        Update the cardinalaties of all the random variables from factors.
        If cardinality of variables doesn't match across all the factors, then
        it will throw CardinalityError
        """
        for factor in factors:
            for variable, cardinality in zip(factor.scope(), factor.cardinality):
                if ((self.cardinalities[variable]) and
                        (self.cardinalities[variable] != cardinality)):
                    raise CardinalityError(
                        'Cardinality of variable %s not matching among factors' % variable)
                else:
                    self.cardinalities[variable] = cardinality

    def add_factors(self, *factors):
        """
        Associate a factor to the graph.
        See factors class for the order of potential values

        Parameters
        ----------
        *factor: pgmpy.factors.Factor object
            A factor object on any subset of the variables of the model which
            is to be associated with the model.

        Returns
        -------
        None

        See Also
        --------
        get_factors

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> from pgmpy.factors import Factor
        >>> student = UndirectedGraph([('Alice', 'Bob'), ('Bob', 'Charles'),
        ...                            ('Charles', 'Debbie'),
        ...                            ('Debbie', 'Alice')])
        >>> factor = Factor(['Alice', 'Bob'], [3, 2], np.random.rand(6))
        >>> student.add_factors(factor)
        """
        self.factors.extend(factors)

    def get_factors(self):
        """
        Returns the factors that have been added till now to the graph

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> from pgmpy.factors import Factor
        >>> student = UndirectedGraph([('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> factor = Factor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                 np.random.rand(6))
        >>> student.add_factors(factor)
        >>> student.get_factors()
        """
        return self.factors

    def remove_factors(self, *factors):
        """
        Removes the given factors from the added factors.

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> from pgmpy.factors import Factor
        >>> student = UndirectedGraph([('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> factor = Factor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                 np.random.rand(6))
        >>> student.add_factors(factor)
        >>> student.remove_factors(factor)
        """
        for factor in factors:
            self.factors.remove(factor)

    def check_clique(self, nodes):
        """
        Check if the given nodes form a clique.

        Parameters
        ----------
        nodes: list of nodes.
            Nodes to check if they are a part of any clique.
        """
        for node1, node2 in itertools.combinations(nodes, 2):
            if not self.has_edge(node1, node2):
                return False
        return True

    def is_triangulated(self):
        """
        Checks whether the undirected graph is triangulated or not.

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_edges_from([('x1', 'x2'), ('x1', 'x3'), ('x1', 'x4'),
        ...                   ('x2', 'x4'), ('x3', 'x4')])
        >>> G.is_triangulated()
        True
        """
        return nx.is_chordal(self)

    def triangulate(self, heuristic='H6', order=None, inplace=False):
        """
        Triangulate the graph.

        If order of deletion is given heuristic algorithm will not be used.

        Parameters
        ----------
        heuristic: H1 | H2 | H3 | H4 | H5 | H6
            The heuristic algorithm to use to decide the deletion order of
            the variables to compute the triangulated graph.
            Let X be the set of variables and X(i) denotes the i-th variable.
                * S(i) - The size of the clique created by deleting the variable.
                * E(i) - Cardinality of variable X(i).
                * M(i) - The maximum size of the cliques of the subgraph given by X(i) and its adjacent nodes.
                * C(i) - The sum of the size of cliques of the subgraph given by X(i)
            and its adjacent nodes.
            The heuristic algorithm decide the deletion order if this way:
                * H1 - Delete the variable with minimal S(i).
                * H2 - Delete the variable with minimal S(i)/E(i).
                * H3 - Delete the variable with minimal S(i) - M(i).
                * H4 - Delete the variable with minimal S(i) - C(i).
                * H5 - Delete the variable with minimal S(i)/M(i).
                * H6 - Delete the variable with minimal S(i)/C(i).

        order: list, tuple (array-like)
            The order of deletion of the variables to compute the triagulated
            graph. If order is given heuristic algorithm will not be used.

        inplace: True | False
            if inplace is true then adds the edges to the object from
            which it is called else returns a new object.

        Reference
        ---------
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.3607

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> from pgmpy.factors import Factor
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        >>> G.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
        ...                   ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
        ...                   ('x4', 'x7'), ('x5', 'x7')])
        >>> phi = [Factor(edge, [2, 2], np.random.rand(4)) for edge in G.edges()]
        >>> G.add_factors(*phi)
        >>> G_chordal = G.triangulate()
        """
        self._update_cardinalities(self.factors)

        if self.is_triangulated():
            if inplace:
                return
            else:
                return self

        graph_copy = nx.Graph(self.edges())
        edge_set = set()

        def _find_common_cliques(cliques_list):
            """
            Finds the common cliques among the given set of cliques for
            corresponding node.
            """
            common = set([tuple(x) for x in cliques_list[0]])
            for i in range(1, len(cliques_list)):
                common = common & set([tuple(x) for x in cliques_list[i]])
            return list(common)

        def _find_size_of_clique(clique, cardinalities):
            """
            Computes the size of a clique.

            Size of a clique is defined as product of cardinalities of all the
            nodes present in the clique.
            """
            return list(map(lambda x: np.prod([cardinalities[node] for node in x]),
                            clique))

        def _get_cliques_dict(node):
            """
            Returns a dictionary in the form of {node: cliques_formed} of the
            node along with its neighboring nodes.

            clique_dict_removed would be containing the cliques created
            after deletion of the node
            clique_dict_node would be containing the cliques created before
            deletion of the node
            """
            graph_working_copy = nx.Graph(graph_copy.edges())
            neighbors = graph_working_copy.neighbors(node)
            graph_working_copy.add_edges_from(itertools.combinations(neighbors, 2))
            clique_dict = nx.cliques_containing_node(graph_working_copy,
                                                     nodes=([node] + neighbors))
            graph_working_copy.remove_node(node)
            clique_dict_removed = nx.cliques_containing_node(graph_working_copy,
                                                             nodes=neighbors)
            return clique_dict, clique_dict_removed

        if not order:
            order = []

            for index in range(self.number_of_nodes()):
                # S represents the size of clique created by deleting the
                # node from the graph
                S = {}
                # M represents the size of maximum size of cliques given by
                # the node and its adjacent node
                M = {}
                # C represents the sum of size of the cliques created by the
                # node and its adjacent node
                C = {}
                for node in set(graph_copy.nodes()) - set(order):
                    clique_dict, clique_dict_removed = _get_cliques_dict(node)
                    S[node] = _find_size_of_clique(
                        _find_common_cliques(list(clique_dict_removed.values())),
                        self.cardinalities
                    )[0]
                    common_clique_size = _find_size_of_clique(
                        _find_common_cliques(list(clique_dict.values())),
                        self.cardinalities
                    )
                    M[node] = np.max(common_clique_size)
                    C[node] = np.sum(common_clique_size)

                if heuristic == 'H1':
                    node_to_delete = min(S, key=S.get)

                elif heuristic == 'H2':
                    S_by_E = {key: S[key] / self.cardinalities[key] for key in S}
                    node_to_delete = min(S_by_E, key=S_by_E.get)

                elif heuristic == 'H3':
                    S_minus_M = {key: S[key] - M[key] for key in S}
                    node_to_delete = min(S_minus_M, key=S_minus_M.get)

                elif heuristic == 'H4':
                    S_minus_C = {key: S[key] - C[key] for key in S}
                    node_to_delete = min(S_minus_C, key=S_minus_C.get)

                elif heuristic == 'H5':
                    S_by_M = {key: S[key] / M[key] for key in S}
                    node_to_delete = min(S_by_M, key=S_by_M.get)

                else:
                    S_by_C = {key: S[key] / C[key] for key in S}
                    node_to_delete = min(S_by_C, key=S_by_C.get)

                order.append(node_to_delete)

        graph_copy = nx.Graph(self.edges())
        for node in order:
            for edge in itertools.combinations(graph_copy.neighbors(node), 2):
                graph_copy.add_edge(edge[0], edge[1])
                edge_set.add(edge)
            graph_copy.remove_node(node)

        if inplace:
            for edge in edge_set:
                self.add_edge(edge[0], edge[1])
            return self

        else:
            graph_copy = UndirectedGraph(self.edges())
            for edge in edge_set:
                graph_copy.add_edge(edge[0], edge[1])
            return graph_copy

    @abc.abstractmethod
    def to_junction_tree(self):
        """
        Creates a junction tree for a given undirected graph.

        For a given undirected graph (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of the
        edge to other.
        """
        pass

    def get_partition_function(self):
        """
        Returns the partition function for a given undirected graph.

        A partition function is defined as

        .. math:: \sum_{X}(\prod_{i=1}^{m} \phi_i)

        where m is the number of factors present in the graph
        and X are all the random variables present.

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> from pgmpy.factors import Factor
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        >>> G.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
        ...                   ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
        ...                   ('x4', 'x7'), ('x5', 'x7')])
        >>> phi = [Factor(edge, [2, 2], np.random.rand(4)) for edge in G.edges()]
        >>> G.add_factors(*phi)
        >>> G.get_partition_function()
        """
        factor = self.factors[0]
        factor = factor.product(*[self.factors[i] for i in
                                  range(1, len(self.factors))])
        if set(factor.scope()) != set(self.nodes()):
            raise ValueError('Factor for all the random variables not defined.')

        return np.sum(factor.values)
