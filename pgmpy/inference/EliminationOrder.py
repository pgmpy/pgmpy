from abc import abstractmethod
from itertools import combinations
from math import prod

import networkx as nx
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.models import DiscreteBayesianNetwork, DiscreteMarkovNetwork


class BaseEliminationOrder:
    """
    Base class of the greedy elimination-order heuristics. Subclasses implement
    `cost`, the cost of eliminating a node from the current graph; the node with
    the smallest cost is eliminated first, its neighbours are connected (fill-in
    edges) and the costs are recomputed on the reduced graph.

    Parameters
    ----------
    model: DiscreteBayesianNetwork or DiscreteMarkovNetwork instance
        The model on which we want to compute the elimination orders. The costs are
        computed on the moral graph of a Bayesian network, or on the Markov network
        itself; the model is not modified.
    """

    def __init__(self, model):
        if isinstance(model, DiscreteBayesianNetwork):
            self.moralized_model = model.moralize()
        elif isinstance(model, DiscreteMarkovNetwork):
            self.moralized_model = nx.Graph()
            self.moralized_model.add_nodes_from(model.nodes())
            self.moralized_model.add_edges_from(model.edges())
        else:
            raise ValueError("Model should be a DiscreteBayesianNetwork or a DiscreteMarkovNetwork instance")
        self.model = model
        self.cardinality = model.get_cardinality()

    @abstractmethod
    def cost(self, node):
        """
        The cost function to compute the cost of elimination of each node.
        This method is just a dummy and returns 0 for all the nodes. Actual cost functions
        are implemented in the classes inheriting BaseEliminationOrder.

        Parameters
        ----------
        node: string, any hashable python object.
            The node whose cost is to be computed.
        """
        return 0

    def get_elimination_order(self, nodes=None, show_progress=True):
        """
        Returns the greedy elimination order based on the cost function: the node
        having the least cost in the current graph is eliminated first (ties are
        broken by the order of the nodes in the graph), its fill-in edges are added,
        and the costs are recomputed. `self.moralized_model` is modified in the
        process.

        Parameters
        ----------
        nodes: list, tuple, set (array-like)
            The variables which are to be eliminated. If None, all the variables
            of the model are eliminated.

        show_progress: boolean (default: True)
            Whether to show a progress bar.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.factors.discrete import TabularCPD
        >>> from pgmpy.inference.EliminationOrder import WeightedMinFill
        >>> rng = np.random.default_rng(42)
        >>> model = DiscreteBayesianNetwork(
        ...     [
        ...         ("c", "d"),
        ...         ("d", "g"),
        ...         ("i", "g"),
        ...         ("i", "s"),
        ...         ("s", "j"),
        ...         ("g", "l"),
        ...         ("l", "j"),
        ...         ("j", "h"),
        ...         ("g", "h"),
        ...     ]
        ... )
        >>> cpd_c = TabularCPD("c", 2, rng.random((2, 1)))
        >>> cpd_d = TabularCPD("d", 2, rng.random((2, 2)), ["c"], [2])
        >>> cpd_g = TabularCPD("g", 3, rng.random((3, 4)), ["d", "i"], [2, 2])
        >>> cpd_i = TabularCPD("i", 2, rng.random((2, 1)))
        >>> cpd_s = TabularCPD("s", 2, rng.random((2, 2)), ["i"], [2])
        >>> cpd_j = TabularCPD("j", 2, rng.random((2, 4)), ["l", "s"], [2, 2])
        >>> cpd_l = TabularCPD("l", 2, rng.random((2, 3)), ["g"], [3])
        >>> cpd_h = TabularCPD("h", 2, rng.random((2, 6)), ["g", "j"], [3, 2])
        >>> model.add_cpds(cpd_c, cpd_d, cpd_g, cpd_i, cpd_s, cpd_j, cpd_l, cpd_h)
        >>> WeightedMinFill(model).get_elimination_order(["c", "d", "g", "l", "s"])
        ['c', 'd', 'l', 's', 'g']
        """
        if nodes is None:
            remaining = list(self.moralized_model.nodes())
        else:
            nodes = set(nodes)
            if missing := nodes - set(self.moralized_model.nodes()):
                raise ValueError(f"Nodes not found in the model: {missing}")
            remaining = [node for node in self.moralized_model.nodes() if node in nodes]

        pbar = None
        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=len(remaining))
            pbar.set_description("Finding Elimination Order: ")

        ordering = []
        while remaining:
            node = min(remaining, key=self.cost)
            ordering.append(node)
            remaining.remove(node)
            self.moralized_model.add_edges_from(self.fill_in_edges(node))
            self.moralized_model.remove_node(node)
            if pbar is not None:
                pbar.update(1)
        if pbar is not None:
            pbar.close()
        return ordering

    def fill_in_edges(self, node):
        """
        Return edges needed to be added to the graph if a node is removed: the pairs
        of its neighbours which are not adjacent yet.

        Parameters
        ----------
        node: string (any hashable python object)
            Node to be removed from the graph.
        """
        graph = self.moralized_model
        return [(u, v) for u, v in combinations(graph.neighbors(node), 2) if not graph.has_edge(u, v)]

    def _weight(self, nodes):
        """Product of the cardinalities of `nodes`; a node without a known cardinality counts as 1."""
        return prod(self.cardinality.get(node, 1) for node in nodes)


class WeightedMinFill(BaseEliminationOrder):
    def cost(self, node):
        """
        Cost function for WeightedMinFill.
        The cost of eliminating a node is the sum of weights of the edges that need to
        be added to the graph due to its elimination, where a weight of an edge is the
        product of the weights, domain cardinality, of its constituent vertices.
        """
        return sum(self._weight(edge) for edge in self.fill_in_edges(node))


class MinNeighbors(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of eliminating a node is the number of neighbors it has in the
        current graph.
        """
        return self.moralized_model.degree(node)


class MinWeight(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of eliminating a node is the product of weights, domain cardinality,
        of its neighbors.
        """
        return self._weight(self.moralized_model.neighbors(node))


class MinFill(BaseEliminationOrder):
    def cost(self, node):
        """
        The cost of eliminating a node is the number of edges that need to be added
        (fill in edges) to the graph due to its elimination
        """
        return len(self.fill_in_edges(node))


class _Kjaerulff(BaseEliminationOrder):
    """
    Base class of Kjaerulff's triangulation heuristics H1-H6, defined for a node X(i)
    of the current graph in terms of

    * S(i) - the size (product of the cardinalities) of the clique created by deleting X(i),
      i.e. of its neighbours,
    * E(i) - the cardinality of X(i),
    * M(i) - the maximum size of the cliques containing X(i),
    * C(i) - the sum of the sizes of the cliques containing X(i).

    References
    ----------
    Kjaerulff, U. (1990). Triangulation of graphs - algorithms giving small total state space.
    """

    def _terms(self, node):
        graph = self.moralized_model
        S = self._weight(graph.neighbors(node))
        E = self.cardinality.get(node, 1)
        clique_sizes = [self._weight(clique) for clique in nx.find_cliques(graph, nodes=[node])]
        return S, E, max(clique_sizes), sum(clique_sizes)


class H1(_Kjaerulff):
    def cost(self, node):
        """Kjaerulff H1: S(i)."""
        S, E, M, C = self._terms(node)
        return S


class H2(_Kjaerulff):
    def cost(self, node):
        """Kjaerulff H2: S(i) / E(i)."""
        S, E, M, C = self._terms(node)
        return S / E


class H3(_Kjaerulff):
    def cost(self, node):
        """Kjaerulff H3: S(i) - M(i)."""
        S, E, M, C = self._terms(node)
        return S - M


class H4(_Kjaerulff):
    def cost(self, node):
        """Kjaerulff H4: S(i) - C(i)."""
        S, E, M, C = self._terms(node)
        return S - C


class H5(_Kjaerulff):
    def cost(self, node):
        """Kjaerulff H5: S(i) / M(i)."""
        S, E, M, C = self._terms(node)
        return S / M


class H6(_Kjaerulff):
    def cost(self, node):
        """Kjaerulff H6: S(i) / C(i)."""
        S, E, M, C = self._terms(node)
        return S / C


ELIMINATION_HEURISTICS = {
    "minfill": MinFill,
    "weightedminfill": WeightedMinFill,
    "minneighbors": MinNeighbors,
    "minweight": MinWeight,
    "h1": H1,
    "h2": H2,
    "h3": H3,
    "h4": H4,
    "h5": H5,
    "h6": H6,
}
