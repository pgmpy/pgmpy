#!/usr/bin/env python

from itertools import combinations
from typing import Optional, Union

import networkx as nx
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery, _ScoreMixin
from pgmpy.estimators.StructureScore import StructureScore, get_scoring_method
from pgmpy.global_vars import logger
from pgmpy.utils.mathext import powerset


class ExhaustiveSearch(_ScoreMixin, _BaseCausalDiscovery):
    """
    Score-based causal discovery using exhaustive search over all possible DAGs.

    Search class for exhaustive searches over all DAGs with a given set of variables.
    Takes a `StructureScore`-Instance as parameter and finds the model with maximal score.

    WARNING: This is computationally expensive! With n variables, there are
    2^(n*(n-1)) possible graphs to evaluate. Only feasible for n <= 6.

    Parameters
    ----------
    scoring_method : str or StructureScore instance, default=None
        The score to be optimized during structure estimation. Supported
        structure scores:

        - Discrete data: 'k2', 'bdeu', 'bds', 'bic-d', 'aic-d'
        - Continuous data: 'll-g', 'aic-g', 'bic-g'
        - Mixed data: 'll-cg', 'aic-cg', 'bic-cg'

        If None, the appropriate scoring method is automatically selected based
        on the data type. Also accepts a custom score instance that inherits
        from `StructureScore`.

    return_type : str, default='dag'
        The type of graph to return. Options are:
        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    use_cache : bool, default=True
        If True, uses caching of local scores for faster computation.
        Note: Caching only works for scoring methods which are decomposable.
        Can give incorrect results for custom non-decomposable scoring methods.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned causal graph with the maximum score.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.causal_discovery import ExhaustiveSearch
    >>> data = pd.DataFrame(
    ...     np.random.randint(low=0, high=2, size=(100, 3)),
    ...     columns=['A', 'B', 'C']
    ... )
    >>> data = data.astype('category')
    >>> est = ExhaustiveSearch(scoring_method='bic-d')
    >>> est.fit(data)
    >>> est.causal_graph_.edges()
    """

    def __init__(
        self,
        scoring_method: Optional[Union[str, StructureScore]] = None,
        return_type: str = "dag",
        use_cache: bool = True,
    ):
        self.scoring_method = scoring_method
        self.return_type = return_type
        self.use_cache = use_cache

    def _fit(self, X: pd.DataFrame):
        """
        Fits the ExhaustiveSearch algorithm to the data.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : ExhaustiveSearch
            Fitted estimator.
        """
        self.variables_ = list(X.columns)

        _, scoring_method = get_scoring_method(
            self.scoring_method, X, self.use_cache
        )

        best_dag = max(
            self._all_dags(nodes=self.variables_), key=scoring_method.score
        )

        best_model = DAG()
        best_model.add_nodes_from(sorted(best_dag.nodes()))
        best_model.add_edges_from(sorted(best_dag.edges()))

        if self.return_type.lower() == "dag":
            self.causal_graph_ = best_model
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = best_model.to_pdag()
        else:
            raise ValueError(
                f"return_type must be one of: dag or pdag. Got: {self.return_type}"
            )

        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, weight=1, dtype="int"
        )

        return self

    def _all_dags(self, nodes=None):
        """
        Generates all possible directed acyclic graphs with a given set of nodes,
        sparse ones first. `2**(n*(n-1))` graphs need to be searched, given `n` nodes,
        so this is likely not feasible for n>6. This is a generator.

        Parameters
        ----------
        nodes : list of nodes for the DAGs, optional
            A list of the node names that the generated DAGs should have.
            If not provided, nodes are taken from data.

        Yields
        ------
        graph : nx.DiGraph
            All acyclic nx.DiGraphs, ordered by number of edges. Empty DAG first.

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.causal_discovery import ExhaustiveSearch
        >>> data = pd.DataFrame(
        ...     data={
        ...         "Temperature": [23, 19],
        ...         "Weather": ["sunny", "cloudy"],
        ...         "Humidity": [65, 75],
        ...     }
        ... )
        >>> data = data.astype('category')
        >>> est = ExhaustiveSearch()
        >>> est.fit(data)
        >>> dags = list(est._all_dags())
        >>> [list(dag.edges()) for dag in dags[:3]]
        """
        if nodes is None:
            nodes = sorted(self.variables_)

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

    def all_scores(self):
        """
        Computes scores for all DAGs, ordered by their scores.

        Returns
        -------
        scored_dags : list
            A list of (score, dag)-tuples, where score is a float and dag is
            an acyclic nx.DiGraph. The list is ordered by score values.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.causal_discovery import ExhaustiveSearch
        >>> np.random.seed(42)
        >>> data = pd.DataFrame(
        ...     np.random.randint(low=0, high=2, size=(100, 3)),
        ...     columns=['A', 'B', 'C']
        ... )
        >>> data = data.astype('category')
        >>> est = ExhaustiveSearch(scoring_method='k2')
        >>> est.fit(data)
        >>> scores = est.all_scores()
        >>> for score, model in scores[:3]:
        ...     print("{0:.2f}\t{1}".format(score, model.edges()))
        """
        if not hasattr(self, "variables_"):
            raise ValueError(
                "Model must be fitted before calling all_scores(). Call fit(X) first."
            )

        _, scoring_method = get_scoring_method(
            self.scoring_method, None, self.use_cache
        )

        scored_dags = sorted(
            [(scoring_method.score(dag), dag) for dag in self._all_dags()],
            key=lambda x: x[0],
        )
        return scored_dags