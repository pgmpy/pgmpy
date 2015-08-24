#!/usr/bin/env python3

import itertools

import networkx as nx


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

    Many common graph features allow python syntax for speed reporting.

    >>> 'a' in G     # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3
    """

    def __init__(self, ebunch=None):
        super().__init__(ebunch)

    def add_node(self, node, **kwargs):
        """
        Add a single node to the Graph.

        Parameters
        ----------
        node: node
            A node can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_node('A')
        """
        super().add_node(node, **kwargs)

    def add_nodes_from(self, nodes, **kwargs):
        """
        Add multiple nodes to the Graph.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, etc.).

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

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edge('Alice', 'Bob')
        """
        super().add_edge(u, v, **kwargs)

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

        Examples
        --------
        >>> from pgmpy.base import UndirectedGraph
        >>> G = UndirectedGraph()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from([('Alice', 'Bob'), ('Bob', 'Charles')])
        """
        for edge in ebunch:
            self.add_edge(*edge, **kwargs)

    def check_clique(self, nodes):
        """
        Check if the given nodes form a clique.

        Parameters
        ----------
        nodes: list, array-like
            List of nodes to check if they are a part of any clique.
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
