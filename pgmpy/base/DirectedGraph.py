#!/usr/bin/env python3

import itertools

import networkx as nx

from pgmpy.base import UndirectedGraph


class DirectedGraph(nx.DiGraph):
    """
    Base class for directed graphs.

    Each node in the graph can represent either a random variable, `Factor`,
    or a cluster of random variables. Edges in the graph represent the
    dependencies between these.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data can be an edge list or any Networkx graph object.

    Examples
    --------
    Create an empty DirectedGraph with no nodes and no edges

    >>> from pgmpy.base import DirectedGraph
    >>> G = DirectedGraph()

    G can be grown in several ways:

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
    are added automatically. There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> 'a' in G     # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3
    """

    def __init__(self, ebunch=None):
        super(DirectedGraph, self).__init__(ebunch)

    def add_node(self, node):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.

        Examples
        --------
        >>> from pgmpy.base import DirectedGraph
        >>> G = DirectedGraph()
        >>> G.add_node('A')
        >>> G.nodes()
        ['A']
        """
        super(DirectedGraph, self).add_node(node)

    def add_nodes_from(self, nodes):
        """
        Add multiple nodes to the Graph.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, or any hashable python
            object).

        Examples
        --------
        >>> from pgmpy.base import DirectedGraph
        >>> G = DirectedGraph()
        >>> G.add_nodes_from(['A', 'B', 'C'])
        >>> G.nodes()
        ['A', 'B', 'C']
        """
        for node in nodes:
            self.add_node(node)

    def add_edge(self, u, v):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.base import DirectedGraph
        >>> G = DirectedGraph()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edge('Alice', 'Bob')
        >>> G.nodes()
        ['Alice', 'Bob', 'Charles']
        >>> G.edges()
        [('Alice', 'Bob')]

        When the node is not already present in the graph:
        >>> G.add_edge('Alice', 'Ankur')
        >>> G.nodes()
        ['Alice', 'Ankur', 'Bob', 'Charles']
        >>> G.edges()
        [('Alice', 'Bob'), ('Alice', 'Ankur')]
        """
        super(DirectedGraph, self).add_edge(u, v) 

    def add_edges_from(self, ebunch):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python
        objects.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        Examples
        --------
        >>> from pgmpy.base import DirectedGraph
        >>> G = DirectedGraph()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from([('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> G.nodes()
        ['Alice', 'Bob', 'Charles']
        >>> G.edges()
        [('Alice', 'Bob'), ('Bob', 'Charles')]

        When the node is not already in the model.
        >>> G.add_edges_from([('Alice', 'Ankur')])
        >>> G.nodes()
        ['Alice', 'Bob', 'Charles', 'Ankur']
        >>> G.edges()
        [('Alice', 'Bob'), ('Bob', 'Charles'), ('Alice', 'Ankur')]
        """
        for edge in ebunch:
            self.add_edge(*edge)

    def get_parents(self, node):
        """
        Returns a list of parents of node.

        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose parents would be returned.

        Examples
        --------
        >>> from pgmpy.base import DirectedGraph
        >>> G = DirectedGraph([('diff', 'grade'), ('intel', 'grade')])
        >>> G.parents('grade')
        ['diff', 'intel']
        """
        return self.predecessors(node)

    def moralize(self):
        """
        Removes all the immoralities in the DirectedGraph and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Examples
        --------
        >>> from pgmpy.base import DirectedGraph
        >>> G = DirectedGraph([('diff', 'grade'), ('intel', 'grade')])
        >>> moral_graph = G.moralize()
        >>> moral_graph.edges()
        [('intel', 'grade'), ('intel', 'diff'), ('grade', 'diff')]
        """
        moral_graph = UndirectedGraph(self.to_undirected().edges())

        for node in self.nodes():
            moral_graph.add_edges_from(
                itertools.combinations(self.get_parents(node), 2))

        return moral_graph
