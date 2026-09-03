from collections.abc import Generator
from itertools import combinations

import networkx as nx
import pandas as pd
from sklearn.utils.validation import check_is_fitted

from pgmpy import logger
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.structure_score import BaseStructureScore, get_scoring_method
from pgmpy.utils.mathext import powerset


class ExhaustiveSearch(BaseCausalDiscovery):
    """
    Score-based causal discovery using exhaustive search over all possible DAGs.

    Given a tabular dataset, the algorithm scores every acyclic graph on the
    given variables and returns the one with the maximal score. Since there are
    ``2**(n*(n-1))`` possible graphs for ``n`` variables, this is only feasible
    for small numbers of variables (n <= 6 or so); for larger problems, use a
    heuristic search such as :class:`~pgmpy.causal_discovery.HillClimbSearch`
    instead.

    Parameters
    ----------
    scoring_method : str or BaseStructureScore instance, default=None
        The score to be optimized during structure estimation. Please refer :doc:`/api/structure_score` for a list of
        available scoring methods.

        If ``None``, the appropriate scoring method is automatically selected based on the data type. If a string is
        provided, the corresponding scoring method is instantiated with default parameters. To customize score-specific
        parameters, please pass an instance of the scoring class.

    return_type : str, default='dag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG) where edges that
          could not be oriented are left undirected.

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned causal graph as a DAG (or PDAG) with maximal score.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/alarm")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the ExhaustiveSearch algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import ExhaustiveSearch
    >>> es = ExhaustiveSearch(scoring_method="bic-d")
    >>> es.fit(df)  # doctest: +SKIP
    ExhaustiveSearch(scoring_method='bic-d')
    >>> _ = es.causal_graph_.edges()  # doctest: +SKIP

    References
    ----------
    - :footcite:t:`koller_friedman_2009`
    """

    def __init__(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        return_type: str = "dag",
        show_progress: bool = True,
    ):
        self.scoring_method = scoring_method
        self.return_type = return_type
        self.show_progress = show_progress

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the ExhaustiveSearch algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.ExhaustiveSearch
            Returns the instance with the fitted attributes.
        """
        self.variables_ = list(X.columns)
        self.scoring_method_ = get_scoring_method(self.scoring_method, X)

        best_dag = max(self.all_dags(), key=self.scoring_method_.score)

        best_model = DAG()
        best_model.add_nodes_from(sorted(best_dag.nodes()))
        best_model.add_edges_from(sorted(best_dag.edges()))

        if self.return_type.lower() == "dag":
            self.causal_graph_ = best_model
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = best_model.to_pdag()
        else:
            raise ValueError(f"return_type must be one of: dag or pdag. Got: {self.return_type}")

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self

    def all_dags(self, nodes: list | None = None) -> Generator[nx.DiGraph]:
        """
        Computes all possible directed acyclic graphs with a given set of nodes,
        sparse ones first. `2**(n*(n-1))` graphs need to be searched, given `n` nodes,
        so this is likely not feasible for n>6. This is a generator.

        Parameters
        ----------
        nodes : list of nodes for the DAGs, optional
            A list of the node names that the generated DAGs should have.
            If not provided, the variables the estimator was fitted on are used.

        Returns
        -------
        dags : Generator object for nx.DiGraphs
            Generator that yields all acyclic nx.DiGraphs, ordered by number of edges. Empty DAG first.

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
        >>> est = ExhaustiveSearch()
        >>> list(est.all_dags(nodes=list(data.columns)))  # doctest: +ELLIPSIS
        [<networkx.classes.digraph.DiGraph object at 0x...>, <networkx.classes.digraph.DiGraph object at 0x...>, ...]
        """
        if nodes is None:
            check_is_fitted(self, "variables_")
            nodes = sorted(self.variables_)

        if len(nodes) > 6:
            logger.info("Generating all DAGs of n nodes likely not feasible for n>6!")
            logger.info(f"Attempting to search through {2 ** (len(nodes) * (len(nodes) - 1))} graphs")

        edges = list(combinations(nodes, 2))
        edges.extend([(y, x) for x, y in edges])
        all_graphs = powerset(edges)

        for graph_edges in all_graphs:
            graph = nx.DiGraph(graph_edges)
            graph.add_nodes_from(nodes)
            if nx.is_directed_acyclic_graph(graph):
                yield graph

    def all_scores(self) -> list[tuple[float, nx.DiGraph]]:
        """
        Computes a list of DAGs and their structure scores, ordered by score.

        Must be called after :meth:`fit`.

        Returns
        -------
        scored_dags : list of (score, dag) pairs
            A list of (score, dag)-tuples, where score is a float and dag is an
            acyclic nx.DiGraph. The list is ordered by score values.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.causal_discovery import ExhaustiveSearch
        >>> np.random.seed(42)
        >>> data = pd.DataFrame(np.random.randint(low=0, high=5, size=(5000, 2)), columns=list("AB"))
        >>> data["C"] = data["B"]
        >>> data = data.astype("category")
        >>> est = ExhaustiveSearch(scoring_method="k2")
        >>> est.fit(data)  # doctest: +SKIP
        >>> scores = est.all_scores()  # doctest: +SKIP
        """
        check_is_fitted(self, "variables_")

        scored_dags = sorted(
            [(self.scoring_method_.score(dag), dag) for dag in self.all_dags()],
            key=lambda x: x[0],
        )
        return scored_dags
