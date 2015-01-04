#!/usr/bin/env python3

from pgmpy.base import UndirectedGraph
from collections import defaultdict
import numpy as np


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
        self.factors = []
        self.cardinalities = defaultdict(int)

    def add_node(self, node, **kwargs):
        """
        Add a single node to the junction tree.

        Parameters
        ----------
        node: node
            A node should be a collection of nodes forming a clique. It can be
            a list, set or tuple of nodes

        Examples
        --------
        >>> from pgmpy.models import JunctionTree
        >>> G = JunctionTree()
        >>> G.add_node(('a', 'b', 'c'))
        """
        if not isinstance(node, (list, set, tuple)):
            raise TypeError('Node can only be a list, set or tuple of nodes'
                            'forming a clique')

        node = tuple(node)
        super(JunctionTree, self).add_node(node, **kwargs)

    def add_nodes_from(self, nodes, **kwargs):
        """
        Add multiple nodes to the junction tree.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, etc.).

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

    def add_factors(self, *factors):
        """
        Associate a factor to the graph.
        See factors class for the order of potential values

        Parameters
        ----------
        *factor: pgmpy.factors.factors object
            A factor object on any subset of the variables of the model which
            is to be associated with the model.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.models import JunctionTree
        >>> from pgmpy.factors import Factor
        >>> student = JunctionTree()
        >>> student.add_node(('Alice', 'Bob'))
        >>> factor = Factor(['Alice', 'Bob'], cardinality=[3, 2], np.random.rand(6))
        >>> student.add_factors(factor)
        """
        for factor in factors:
            factor_scope = set(factor.scope())
            nodes = [set(node) for node in self.nodes()]
            if factor_scope not in nodes:
                raise ValueError('Factors defined on clusters of variable not'
                                 'present in model')

            self.factors.append(factor)

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
            return self.factors
        else:
            nodes = [set(node) for node in self.nodes()]
            if set(node) not in nodes:
                raise ValueError('Node not present in Junction Tree')
            factors = list(filter(lambda x: set(x.scope()) == set(node),
                                  self.factors))
            return factors[0]

    def remove_factors(self, *factors):
        """
        Removes the given factors from the added factors.

        Examples
        --------
        >>> from pgmpy.models import JunctionTree
        >>> from pgmpy.factors import Factor
        >>> student = JunctionTree()
        >>> factor = Factor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                 np.random.rand(6))
        >>> student.add_factors(factor)
        >>> student.remove_factors(factor)
        """
        for factor in factors:
            self.factors.remove(factor)

    def get_partition_function(self):
        """
        Returns the partition function for a given undirected graph.

        A partition function is defined as

        .. math:: \sum_{X}(\prod_{i=1}^{m} \phi_i)

        where m is the number of factors present in the graph
        and X are all the random variables present.

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
        >>> G.get_partition_function()
        """
        if self.check_model():
            factor = self.factors[0]
            factor = factor.product(*[self.factors[i] for i in
                                      range(1, len(self.factors))])
            return np.sum(factor.values)

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if clique potentials are defined for all the cliques or not.
        * Check for running intersection property is not done explicitly over
        here as it done in the add_edges method.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        for clique in self.nodes():
            if self.get_factors(clique):
                pass
            else:
                raise ValueError('Factors for all the cliques or clusters not'
                                 'defined.')

        if len(self.factors) != len(self.nodes()):
            raise ValueError('One to one mapping of factor to clique or cluster'
                             'is not there.')

        return True
