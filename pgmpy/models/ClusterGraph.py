#!/usr/bin/env python3

from collections import defaultdict

import numpy as np

from pgmpy.base import UndirectedGraph
from pgmpy.factors import FactorDict, factor_product
from pgmpy.utils import compat_fns


class ClusterGraph(UndirectedGraph):
    r"""
    Base class for representing Cluster Graph.

    Cluster graph is an undirected graph which is associated with a subset of variables. The graph contains undirected
    edges that connects clusters whose scopes have a non-empty intersection.

    Formally, a cluster graph is  :math:`\mathcal{U}` for a set of factors :math:`\Phi` over :math:`\mathcal{X}` is an
    undirected graph, each of whose nodes :math:`i` is associated with a subset :math:`C_i \subseteq X`. A cluster
    graph must be family-preserving - each factor :math:`\phi \in \Phi` must be associated with a cluster C, denoted
    :math:`\alpha(\phi)`, such that :math:`Scope[\phi] \subseteq C_i`. Each edge between a pair of clusters :math:`C_i`
    and :math:`C_j` is associated with a sepset :math:`S_{i,j} \subseteq C_i \cap C_j`.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is created. The data is an edge list

    Examples
    --------
    Create an empty ClusterGraph with no nodes and no edges

    >>> from pgmpy.models import ClusterGraph
    >>> G = ClusterGraph()

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
        super(ClusterGraph, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.factors = []

    def add_node(self, node, **kwargs):
        """
        Add a single node to the cluster graph.

        Parameters
        ----------
        node: node
            A node should be a collection of nodes forming a clique. It can be
            a list, set or tuple of nodes

        Examples
        --------
        >>> from pgmpy.models import ClusterGraph
        >>> G = ClusterGraph()
        >>> G.add_node(('a', 'b', 'c'))
        """
        if not isinstance(node, (list, set, tuple)):
            raise TypeError(
                "Node can only be a list, set or tuple of nodes forming a clique"
            )

        node = tuple(node)
        super(ClusterGraph, self).add_node(node, **kwargs)

    def add_nodes_from(self, nodes, **kwargs):
        """
        Add multiple nodes to the cluster graph.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, etc.).

        Examples
        --------
        >>> from pgmpy.models import ClusterGraph
        >>> G = ClusterGraph()
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
        >>> from pgmpy.models import ClusterGraph
        >>> G = ClusterGraph()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        """
        set_u = set(u)
        set_v = set(v)
        if set_u.isdisjoint(set_v):
            raise ValueError("No sepset found between these two edges.")

        super(ClusterGraph, self).add_edge(u, v)

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
        >>> from pgmpy.models import ClusterGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = ClusterGraph()
        >>> student.add_node(('Alice', 'Bob'))
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[3, 2],
        ...                 values=np.random.rand(6))
        >>> student.add_factors(factor)
        """
        for factor in factors:
            factor_scope = set(factor.scope())
            nodes = [set(node) for node in self.nodes()]
            if factor_scope not in nodes:
                raise ValueError(
                    "Factors defined on clusters of variable not" "present in model"
                )

            self.factors.append(factor)

    def get_factors(self, node=None):
        """
        Return the factors that have been added till now to the graph.

        If node is not None, it would return the factor corresponding to the
        given node.

        Examples
        --------
        >>> from pgmpy.models import ClusterGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = ClusterGraph()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        >>> phi1 = DiscreteFactor(['a', 'b', 'c'], [2, 2, 2], np.random.rand(8))
        >>> phi2 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi3 = DiscreteFactor(['a', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2, phi3)
        >>> G.get_factors()
        >>> G.get_factors(node=('a', 'b', 'c'))
        """
        if node is None:
            return self.factors
        else:
            nodes = [set(n) for n in self.nodes()]

            if set(node) not in nodes:
                raise ValueError("Node not present in Cluster Graph")

            factors = filter(lambda x: set(x.scope()) == set(node), self.factors)
            return next(factors)

    def remove_factors(self, *factors):
        """
        Removes the given factors from the added factors.

        Examples
        --------
        >>> from pgmpy.models import ClusterGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = ClusterGraph()
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                         values=np.random.rand(4))
        >>> student.add_node(('Alice', 'Bob'))
        >>> student.add_factors(factor)
        >>> student.remove_factors(factor)
        """
        for factor in factors:
            self.factors.remove(factor)

    @property
    def clique_beliefs(self) -> FactorDict:
        """
        Return a mapping from the cliques to their factor representations.

        Returns
        -------
        FactorDict: mapping from cliques to factors

        Examples
        --------
        >>> from pgmpy.models import ClusterGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = ClusterGraph()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        >>> phi1 = DiscreteFactor(['a', 'b', 'c'], [2, 2, 2], np.random.rand(8))
        >>> phi2 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi3 = DiscreteFactor(['a', 'c'], [2, 2], np.random.rand(4))
        >>> G.clique_beliefs
        """
        return FactorDict({clique: self.get_factors(clique) for clique in self.nodes()})

    @clique_beliefs.setter
    def clique_beliefs(self, clique_beliefs: FactorDict) -> None:
        self.remove_factors(*self.get_factors())
        self.add_factors(*clique_beliefs.values())

    def get_cardinality(self, node=None):
        """
        Returns the cardinality of the node

        Parameters
        ----------
        node: any hashable python object (optional)
            The node whose cardinality we want. If node is not specified returns a
            dictionary with the given variable as keys and their respective cardinality
            as values.

        Returns
        -------
        int or dict : If node is specified returns the cardinality of the node.
                      If node is not specified returns a dictionary with the given
                      variable as keys and their respective cardinality as values.


        Examples
        --------
        >>> from pgmpy.models import ClusterGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = ClusterGraph()
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                 values=np.random.rand(4))
        >>> student.add_node(('Alice', 'Bob'))
        >>> student.add_factors(factor)
        >>> student.get_cardinality()
        defaultdict(<class 'int'>, {'Alice': 2, 'Bob': 2})

        >>> student.get_cardinality(node='Alice')
        2
        """
        if node:
            for factor in self.factors:
                for variable, cardinality in zip(factor.scope(), factor.cardinality):
                    if node == variable:
                        return cardinality

        else:
            cardinalities = defaultdict(int)
            for factor in self.factors:
                for variable, cardinality in zip(factor.scope(), factor.cardinality):
                    cardinalities[variable] = cardinality
            return cardinalities

    def get_partition_function(self):
        r"""
        Returns the partition function for a given undirected graph.

        A partition function is defined as

        .. math:: \sum_{X}(\prod_{i=1}^{m} \phi_i)

        where m is the number of factors present in the graph
        and X are all the random variables present.

        Examples
        --------
        >>> from pgmpy.models import ClusterGraph
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = ClusterGraph()
        >>> G.add_nodes_from([('a', 'b', 'c'), ('a', 'b'), ('a', 'c')])
        >>> G.add_edges_from([(('a', 'b', 'c'), ('a', 'b')),
        ...                   (('a', 'b', 'c'), ('a', 'c'))])
        >>> phi1 = DiscreteFactor(['a', 'b', 'c'], [2, 2, 2], np.random.rand(8))
        >>> phi2 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi3 = DiscreteFactor(['a', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2, phi3)
        >>> G.get_partition_function()
        """
        if self.check_model():
            factor = self.factors[0]
            factor = factor_product(
                factor, *[self.factors[i] for i in range(1, len(self.factors))]
            )
            return compat_fns.sum(factor.values)

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors.

        * Checks if factors are defined for all the cliques or not.
        * Check for running intersection property is not done explicitly over
          here as it done in the add_edges method.
        * Checks if cardinality information for all the variables is available or not. If
          not it raises an error.
        * Check if cardinality of random variable remains same across all the
          factors.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        for clique in self.nodes():
            factors = filter(lambda x: set(x.scope()) == set(clique), self.factors)
            if not any(factors):
                raise ValueError("Factors for all the cliques or clusters not defined.")

        cardinalities = self.get_cardinality()
        if len(set((x for clique in self.nodes() for x in clique))) != len(
            cardinalities
        ):
            raise ValueError("Factors for all the variables not defined.")

        for factor in self.factors:
            for variable, cardinality in zip(factor.scope(), factor.cardinality):
                if cardinalities[variable] != cardinality:
                    raise ValueError(
                        f"Cardinality of variable {variable} not matching among factors"
                    )

        return True

    def copy(self):
        """
        Returns a copy of ClusterGraph.

        Returns
        -------
        ClusterGraph: copy of ClusterGraph

        Examples
        --------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = ClusterGraph()
        >>> G.add_nodes_from([('a', 'b'), ('b', 'c')])
        >>> G.add_edge(('a', 'b'), ('b', 'c'))
        >>> phi1 = DiscreteFactor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = DiscreteFactor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> graph_copy = G.copy()
        >>> graph_copy.factors
        [<DiscreteFactor representing phi(a:2, b:2) at 0xb71b19cc>,
         <DiscreteFactor representing phi(b:2, c:2) at 0xb4eaf3ac>]
        >>> graph_copy.edges()
        [(('a', 'b'), ('b', 'c'))]
        >>> graph_copy.nodes()
        [('a', 'b'), ('b', 'c')]
        """
        copy = ClusterGraph(self.edges())
        if self.factors:
            factors_copy = [factor.copy() for factor in self.factors]
            copy.add_factors(*factors_copy)
        return copy
