#!/usr/bin/env python3

from pgmpy.models import ClusterGraph


class JunctionTree(ClusterGraph):
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
        import networkx as nx

        if u in self.nodes() and v in self.nodes() and nx.has_path(self, u, v):
            raise ValueError('Addition of edge between {u} and {v} forms a cycle breaking the '
                             'properties of Junction Tree'.format(u=str(u), v=str(v)))

        super(JunctionTree, self).add_edge(u, v, **kwargs)

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors. In the same time also updates the cardinalities of all the random
        variables.

        * Checks if clique potentials are defined for all the cliques or not.
        * Check for running intersection property is not done explicitly over
        here as it done in the add_edges method.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        import networkx as nx

        if not nx.is_connected(self):
            raise ValueError('The Junction Tree defined is not fully connected.')

        return super(JunctionTree, self).check_model()
