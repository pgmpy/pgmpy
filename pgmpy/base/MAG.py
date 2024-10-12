#!/usr/bin/env python3

import networkx as nx

from pgmpy.base import DAG


class MixedGraph(nx.MultiDiGraph):
    """
    Class representing a Mixed Graph. A mixed graph can contain both a directed
    edge and a bi-directed edge. Bi-directed edges are represented using two
    edges between the same nodes in opposite directions. All the operations are
    although done on a canonical representation of the Mixed Graph which is a
    DAG. The canonical representation replaces each bidirected edge with a
    latent variable.  For example: A <-> B is converted to A <- _e_AB -> B.
    """

    def __init__(self, directed_edges=None, bidirected_edges=None, latents=set()):
        """
        Initialzies a Mixed Graph.

        Parameters
        ----------
        directed_edges: list
            Data to initialize graph.  If data=None (default) an empty
            graph is created.  The data can be an edge list, or any
            NetworkX graph object. For bidirected edges e.g. X <-> Y, two
            edges need to be specified: X -> Y and Y <- X.

        latents: set, array-like
            List of variables which are latent (i.e. unobserved) in the model.
        """
        # TODO: Check why init is not taking the arguments directly.
        super(MixedGraph, self).__init__()

        # These get filled in the `add_edge` method.
        self.directed_edges = set()
        self.bidirected_edges = set()

        self.add_edges_from(
            directed_edges=directed_edges, bidirected_edges=bidirected_edges
        )
        self.latents = set(latents)

    def copy(self):
        """
        Returns a copy of the current object.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph([("A", "B"), ("A", "B"), ("B", "A")])
        >>> G_copy = G.copy()
        """
        return MixedGraph(
            directed_edges=self.directed_edges,
            bidirected_edges=self.bidirected_edges,
            latents=self.latents,
        )

    def to_canonical(self):
        """
        Converts a mixed graph into it's canonical representation.
        For each bi-directed edge, a latent variable is added as.
        For example: A <-> B is converted to A <- _e_AB -> B.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph([("A", "B"), ("B", "C"), ("A", "C"), ("C", "A")])
        >>> G_canonical = G.to_canonical()
        >>> G_canonical.nodes()
        >>> G_canonical.edges()
        """
        new_latents = set()

        edges = list(self.directed_edges)
        for u, v in self.bidirected_edges:
            latent_node = f"_e_{str(u)}{str(v)}"
            new_latents.add(latent_node)

            edges.extend([(latent_node, u), (latent_node, v)])

        return DAG(ebunch=edges, latents=self.latents.union(new_latents))

    def add_node(self, node):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph()
        >>> G.add_node(node='A')
        >>> sorted(G.nodes())
        ['A']
        """
        super(MixedGraph, self).add_node(node)

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
        >>> from pgmpy.base import DAG
        >>> G = MixedGraph()
        >>> G.add_nodes_from(nodes=['A', 'B', 'C'])
        >>> G.nodes()
        NodeView(('A', 'B', 'C'))
        """
        nodes = list(nodes)

        for node in nodes:
            self.add_node(node=node)

    def add_edge(self, u, v, directed=True):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edge(u='Alice', v='Bob')
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob')])

        When the node is not already present in the graph:
        >>> G.add_edge(u='Alice', v='Ankur')
        >>> G.nodes()
        NodeView(('Alice', 'Ankur', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Alice', 'Ankur')])
        """
        if directed:
            super(MixedGraph, self).add_edge(u, v)
            self.directed_edges.add((u, v))
        else:
            super(MixedGraph, self).add_edge(u, v)
            super(MixedGraph, self).add_edge(v, u)
            self.bidirected_edges.add((u, v))

    def add_edges_from(self, directed_edges=None, bidirected_edges=None):
        """
        Add all the edges in directed_edges and bidirected_edges.

        If nodes referred in the edges are not already present, they
        will be automatically added. Node names can be any hashable python
        object.

        Parameters
        ----------
        edges: container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).


        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> G = MixedGraph()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from(ebunch=[('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles')])

        When the node is not already in the model:
        >>> G.add_edges_from(ebunch=[('Alice', 'Ankur')])
        >>> G.nodes()
        NodeView(('Alice', 'Bob', 'Charles', 'Ankur'))
        >>> G.edges()
        OutEdgeView([('Alice', 'Bob'), ('Bob', 'Charles'), ('Alice', 'Ankur')])
        """
        directed_edges = [] if directed_edges is None else list(directed_edges)
        bidirected_edges = [] if bidirected_edges is None else list(bidirected_edges)

        for edge in directed_edges:
            self.add_edge(edge[0], edge[1], directed=True)
        for edge in bidirected_edges:
            self.add_edge(edge[0], edge[1], directed=False)

    def get_spouse(self, node):
        """
        Returns the spouse of `node`. Spouse in mixed graph is defined as
        the nodes connected to `node` through a bi-directed edge.

        Parameters
        ----------
        node: any hashable python object
            The node whose spouses are needed.

        Returns
        -------
        list: List of spouses of `node`.

        Examples
        --------
        >>> from pgmpy.base import MixedGraph
        >>> g = MixedGraph([('X', 'Y'), ('Y', 'Z'), ('X', 'Z'), ('Z', 'X')])
        >>> g.get_spouse('X')
        ['Z']
        >>> g.get_spouse('Y')
        []
        """
        spouses = []
        for neigh in self.neighbors(node):
            if node in self.neighbors(neigh):
                spouses.append(neigh)

        return spouses


class MAG(MixedGraph):
    """
    Class representing Maximal Ancestral Graph (MAG)[1].

    References
    ----------
    [1] Zhang, Jiji. "Causal reasoning with ancestral graphs." Journal of Machine Learning Research 9 (2008): 1437-1474.
    """

    def __init__(self, ebunch=None, latents=set()):
        super(MAG, self).__init__(ebunch=directed_ebunch, latents=latents)


class PAG(MixedGraph):
    """
    Class representing Partial Ancestral Graph (PAG)[1].

    References
    ----------
    [1] Zhang, Jiji. "Causal reasoning with ancestral graphs." Journal of Machine Learning Research 9 (2008): 1437-1474.
    """

    def __init__(self, directed_ebunch=None, undirected_ebunch=None, latents=set()):
        super(PAG, self).__init__(ebunch=directed_ebunch, latents=latents)
        self.undirected_edges = set(undirected_ebunch)
