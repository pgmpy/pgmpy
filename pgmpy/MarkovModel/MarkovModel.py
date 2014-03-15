#!/usr/bin/env python3

import networkx as nx
import numpy as np
import itertools
from pgmpy import Exceptions
from pgmpy.Factor import Factor


class MarkovModel(nx.Graph):
    """
        Base class for Markov Model.

        A MarkovModel stores nodes and edges with factors over cliques in the
        graph and other attributes.

        MarkovModel holds undirected edges. Self loops are not allowed neither
        multiple (parallel) edges.

        Nodes should be strings.

        Edges are represented as links between nodes.

        Parameters
        ----------
        data : input graph
            Data to initialize graph.  If data=None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX Graph object.

        Examples
        --------
        Create an empty Markov model with no nodes and no edges.

        >>> from pgmpy import MarkovModel as mm
        >>> H = mm.MarkovModel()

        H can be grown in several ways.

        **Nodes:**

        Add one node at a time:

        >>> H.add_node('a')

        Add the nodes from any container (a list, set or tuple or the nodes
        from another graph).

        >>> H.add_nodes_from(['a', 'b'])

        **Edges:**

        H can also be grown by adding edges.

        Add one edge,

        >>> H.add_edge('a', 'b')

        a list of edges,

        >>> H.add_edges_from([('a', 'b'), ('b', 'c')])

        If some edges connect nodes not yet in the model, the nodes
        are added automatically.  There are no errors when adding
        nodes or edges that already exist.

        **Shortcuts:**

        Many common graph features allow python syntax to speed reporting.

        >>> 'a' in H     # check if node in graph
        True
        >>> len(H)  # number of nodes in graph
        3

        Public Methods
        --------------
        add_node('node1')
        add_nodes_from(['node1', 'node2', ...])
        add_edge('node1')
        add_edges_from([('node1', 'node2'),('node3', 'node4')])
    """
    def __init__(self, ebunch=None):
        if ebunch is not None:
            self._check_node_string(set(itertools.chain(*ebunch)))

        nx.Graph.__init__(self, ebunch)

        if ebunch is not None:
            new_nodes = set(itertools.chain(*ebunch))
            self._check_graph(delete_graph=True)
            self._update_node_neighbors(new_nodes)

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
        >>> from pgmpy import MarkovModel as mm
        >>> H = mm.MarkovModel()
        >>> H.add_node('A')
        """
        self._check_node_string([node])
        nx.Graph.add_node(self, node)

        self._update_node_neighbors([node])

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
        >>> from pgmpy import MarkovModel as mm
        >>> H = mm.MarkovModel()
        >>> H.add_nodes_from(['A', 'B', 'C'])
        """
        self._check_node_string(nodes)
        nx.Graph.add_nodes_from(self, nodes)

        self._update_node_neighbors(nodes)

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
        >>> from pgmpy import MarkovModel as mm
        >>> H = mm.MarkovModel()
        >>> H.add_nodes_from(['A', 'B'])
        >>> H.add_edge('A', 'B')
        """
        # string check required because if nodes not present networkx
        # automatically adds those nodes
        self._check_node_string([u, v])
        if self._check_graph([(u, v)], delete_graph=False):
            nx.Graph.add_edge(self, u, v)

        self._update_node_neighbors([u, v])

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
        >>> from pgmpy import MarkovModel as mm
        >>> H = mm.MarkovModel()
        >>> H.add_nodes_from(['A', 'B', 'C'])
        >>> H.add_edges_from([('A', 'B'), ('C', 'B')])
        """
        if ebunch is not None:
            self._check_node_string(set(itertools.chain(*ebunch)))

        if self._check_graph(ebunch, delete_graph=False):
            nx.Graph.add_edges_from(self, ebunch)

        new_nodes = set(itertools.chain(*ebunch))
        self._update_node_neighbors(new_nodes)

    def _update_node_neighbors(self, node_list):
        """
        Function to update each node's _neighbors attribute.
        This function is called when new node or new edge is added.
        """
        for node in node_list:
            self.node[node]['_neighbors'] = sorted(self.neighbors(node))

    def _check_graph(self, ebunch=None, delete_graph=False):
        """
        Checks for self loops in the graph. If finds any, reverts the graph
        to previous state or in case when called from __init__ deletes the
        graph.
        """
        if delete_graph:
            if ebunch is not None:
                for edge in ebunch:
                    if edge[0] == edge[1]:
                        del self
                        raise Exceptions.SelfLoopError("Self Loops are not "
                                                       "allowed", edge)
        else:
            for edge in ebunch:
                if edge[0] == edge[1]:
                    raise Exceptions.SelfLoopError("Self loops are not "
                                                   "allowed", edge)
            return True

    @staticmethod
    def _check_node_string(node_list):
        """
        Checks if all the newly added node are strings.
        Called from __init__, add_node, add_nodes_from, add_edge and
        add_edges_from.
        """
        for node in node_list:
            if not (isinstance(node, str)):
                raise TypeError("Node names must be strings")
