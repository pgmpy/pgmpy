#!/usr/bin/env python3

from pgmpy.base import UndirectedGraph


class JunctionTree(UndirectedGraph):
    """
    Class for representing Junction Tree.

    Junction tree is undirected graph where each node represents a clique
    (list, tuple or set of nodes) and edges represent sepset between two cliques.
    Each sepset in G separates the variables strictly on one side of edge to
    other.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data is an edge list.

    Examples
    --------
    Create an empty JunctionTree with no nodes and no edges

    >>> from pgmpy.models import JunctionTree
    >>> G = JunctionTree()

    G can be grown by adding clique nodes.

    **Nodes:**

    Add a tuple (or list or set) of nodes as single clique node.

    >>> G.add_node(('a', 'b', 'c'))
    >>> G.add_nodes_from([('a', 'b'), ('a', 'b', 'c')])

    **Edges:**

    G can also be grown by adding edges.

    >>> G.add_edge(('a', 'b', 'c'), ('a', 'b'))

    or a list of edges

    >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
    ...                   (('a', 'b', 'c'), ('a', 'c'))])
    """

    def __init__(self, ebunch=None):
        super(JunctionTree, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)

    def add_node(self, node, **kwargs):
        """
        Add a single node to the junction tree.

        Parameters
        ----------
        node: node
            A node should be a collection of nodes forming a clique. It can be
            a list, set or tuple of nodes

        See Also
        --------
        add_nodes_from: add a collection of nodes

        Examples
        --------
        >>> from pgmpy.models import JunctionTree
        >>> G = JunctionTree()
        >>> G.add_node(('a', 'b', 'c'))
        """
        if not isinstance(node, (list, set, tuple)):
            raise TypeError('Node can only be a list, set or tuple of nodes forming a clique')

        node = tuple(node)
        super(JunctionTree, self).add_node(node, **kwargs)

    def add_nodes_from(self, nodes, **kwargs):
        """
        Add multiple nodes to the junction tree.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, etc.).

        See Also
        --------
        add_node: add a single node

        Examples
        --------
        >>> from pgmpy.models import JunctionTree
        >>> G = JunctionTree()
        >>> G.add_nodes_from([('a', 'b'), ('a', 'b', 'c')])
        """
        for node in nodes:
            self.add_node(node, **kwargs)

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between two clique nodes.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any list or set or tuple of nodes forming a clique.

        Examples
        --------
        >>> from pgmpy.models import JunctionTree
        >>> G = JunctionTree()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        """
        set_u = set(u)
        set_v = set(v)
        if not set_u.intersection(set_v):
            raise ValueError('No sepset found between these two edges.')

        super(JunctionTree, self).add_edge(u, v)

    def get_factors(self, node=None):
        """
        Return the factors that have been added till now to the graph.

        If node is not None, it would return the factor corresponding to the
        given node.

        Examples
        --------
        >>> from pgmpy.models import JunctionTree
        >>> from pgmpy.factors import Factor
        >>> G = JunctionTree()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        >>> phi1 = Factor(['a', 'b', 'c'], [2, 2, 2], np.random.rand(8))
        >>> phi2 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi3 = Factor(['a', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2, phi3)
        >>> G.get_factors()
        >>> G.get_factors(node=('a', 'b', 'c'))
        """
        if node is None:
            return super(JunctionTree, self).get_factors()
        else:
            if node not in self.nodes():
                raise ValueError('Node not present in Junction Tree')
            factors = list(filter(lambda x: set(x.scope()) == set(node), self.factors))
            return factors[0]
