#!/usr/bin/env python3

from pgmpy.base import UndirectedGraph
from networkx.algorithms import bipartite
import itertools


class FactorGraph(UndirectedGraph):
    """
    Class for representing factor graph.

    Factor graph is a bipartite graph representing factorization of a function.
    They allow efficient computation of marginal distributions through sum-product
    algorithm.

    A factor graph contains two types of nodes. One type corresponds to random
    variables whereas the second type corresponds to factors over these variables.
    The graph only contains edges between variables and factor nodes. Each factor
    node is associated with one factor whose scope is the set of variables that
    are its neighbors.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data is an edge list.

    Examples
    --------
    Create an empty FactorGraph with no nodes and no edges

    >>> from pgmpy.models import FactorGraph
    >>> G = FactorGraph()

    G can be grown by adding variable nodes as well as factor nodes

    **Nodes:**

    Add a node at a time or a list of nodes.

    >>> G.add_node('a')
    >>> G.add_nodes_from(['a', 'b'])

    **Edges:**

    G can also be grown by adding edges.

    >>> G.add_edge('a', 'phi1')

    or a list of edges

    >>> G.add_edges_from([('a', 'phi1'), ('b', 'phi1')])
    """

    def __init__(self, ebunch=None):
        super(FactorGraph, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between variable_node and factor_node.

        Parameters
        ----------
        u, v: nodes
            Nodes can be any hashable Python object.

        See Also
        --------
        add_edges_from: add a collection of edges

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> G.add_nodes_from(['phi1', 'phi2'])
        >>> G.add_edge('a', 'phi1')
        """
        if u != v:
            super(FactorGraph, self).add_edge(u, v, **kwargs)
        else:
            raise ValueError('Self loops are not allowed')

        if not bipartite.is_bipartite(self):
            self.remove_edge(u, v)
            raise ValueError('Edges can only be between variables and factors')

    def add_factors(self, *factors):
        """
        Associate a factor to the graph.
        See factors class for the order of potential values.

        Parameters
        ----------
        *factor: pgmpy.factors.Factor object
            A factor object on any subset of the variables of the model which
            is to be associated with the model.

        See Also
        --------
        get_factors

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors import Factor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> G.add_nodes_from(['phi1', 'phi2'])
        >>> G.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
        ...                   ('b', 'phi2'), ('c', 'phi2')])
        >>> phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1)
        """
        for factor in factors:
            if set(factor.variables) - set(factor.variables).intersection(
                    set(self.nodes())):
                raise ValueError("Factors defined on variable not in the model",
                                 factor)
            super(FactorGraph, self).add_factors(factor)

    def to_markov_model(self):
        """
        Converts the factor graph into markov model.

        A markov model contains nodes as random variables and edge between
        two nodes imply interaction between them.

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors import Factor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> G.add_nodes_from(['phi1', 'phi2'])
        >>> G.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
        ...                   ('b', 'phi2'), ('c', 'phi2')])
        >>> phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> mm = G.to_markov_model()
        """
        from pgmpy.models import MarkovModel
        mm = MarkovModel()

        variable_nodes = set([x for factor in self.factors for x in factor.scope()])
        if not bipartite.is_bipartite_node_set(self, variable_nodes):
            raise ValueError('Factors not associated for all the random variables.')

        mm.add_nodes_from(variable_nodes)
        for factor in self.factors:
            scope = factor.scope()
            mm.add_edges_from(itertools.combinations(scope, 2))
            mm.add_factors(factor)

        return mm

    def to_junction_tree(self):
        """
        Create a junction treeo (or clique tree) for a given factor graph.

        For a given factor graph (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of
        edge to other

        Examples
        --------
        >>> from pgmpy.models import FactorGraph
        >>> from pgmpy.factors import Factor
        >>> G = FactorGraph()
        >>> G.add_nodes_from(['a', 'b', 'c'])
        >>> G.add_nodes_from(['phi1', 'phi2'])
        >>> G.add_edges_from([('a', 'phi1'), ('b', 'phi1'),
        ...                   ('b', 'phi2'), ('c', 'phi2')])
        >>> phi1 = Factor(['a', 'b'], [2, 2], np.random.rand(4))
        >>> phi2 = Factor(['b', 'c'], [2, 2], np.random.rand(4))
        >>> G.add_factors(phi1, phi2)
        >>> junction_tree = G.to_junction_tree()
        """
        mm = self.to_markov_model()
        return mm.to_junction_tree()
