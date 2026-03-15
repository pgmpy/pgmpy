from itertools import combinations

import networkx as nx
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery, _ScoreMixin
from pgmpy.estimators.StructureScore import get_scoring_method
from pgmpy.global_vars import logger
from pgmpy.utils.mathext import powerset


class ExhaustiveSearch(_ScoreMixin, _BaseCausalDiscovery):
    """
    Causal discovery using exhaustive search over all possible DAGs.

    Searches through all possible DAG structures and returns the one
    with the highest structure score. Only feasible for datasets with
    6 or fewer variables due to exponential search space.

    Parameters
    ----------
    scoring_method : str or StructureScore instance, default=None
        The score to be optimized. If None, automatically selected
        based on data type.

    use_cache : bool, default=True
        If True, uses caching for faster score computation.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned causal graph with the highest score.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix of the learned causal graph.

    n_features_in_ : int
        Number of features in the input data.

    feature_names_in_ : np.ndarray
        Feature names in the input data.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.causal_discovery import ExhaustiveSearch
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(np.random.randint(0, 3, size=(500, 3)), columns=list("ABC"))
    >>> est = ExhaustiveSearch()
    >>> est.fit(data)
    >>> list(est.causal_graph_.edges())
    [('B', 'A')]
    """

    def __init__(self, scoring_method=None, use_cache=True):
        self.scoring_method = scoring_method
        self.use_cache = use_cache

    def _all_dags(self, nodes):
        """
        Generator that yields all possible DAGs for given nodes.

        Parameters
        ----------
        nodes : list
            List of node names.

        Yields
        ------
        nx.DiGraph
            A valid directed acyclic graph.
        """
        if len(nodes) > 6:
            logger.info("Generating all DAGs of n nodes likely not feasible for n>6!")
            logger.info(
                "Attempting to search through {n} graphs".format(
                    n=2 ** (len(nodes) * (len(nodes) - 1))
                )
            )

        edges = list(combinations(nodes, 2))
        edges.extend([(y, x) for x, y in edges])
        all_graphs = powerset(edges)

        for graph_edges in all_graphs:
            graph = nx.DiGraph(graph_edges)
            graph.add_nodes_from(nodes)
            if nx.is_directed_acyclic_graph(graph):
                yield graph

    def all_scores(self, X):
        """
        Computes all DAGs and their structure scores, ordered by score.

        Parameters
        ----------
        X : pd.DataFrame
            The data to score DAGs against.

        Returns
        -------
        list of (score, dag) tuples
            Ordered by score value.
        """
        nodes = sorted(X.columns)
        _, scoring_method = get_scoring_method(self.scoring_method, X, self.use_cache)
        scored_dags = sorted(
            [(scoring_method.score(dag), dag) for dag in self._all_dags(nodes)],
            key=lambda x: x[0],
        )
        return scored_dags

    def _fit(self, X: pd.DataFrame):
        """
        Finds the DAG with the highest structure score.

        Parameters
        ----------
        X : pd.DataFrame
            Data to learn causal structure from.

        Returns
        -------
        self : ExhaustiveSearch
        """
        nodes = sorted(X.columns)

        _, scoring_method = get_scoring_method(self.scoring_method, X, self.use_cache)

        best_dag = max(self._all_dags(nodes), key=scoring_method.score)

        best_model = DAG()
        best_model.add_nodes_from(sorted(best_dag.nodes()))
        best_model.add_edges_from(sorted(best_dag.edges()))

        self.causal_graph_ = best_model
        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, weight=1, dtype="int"
        )

        return self
