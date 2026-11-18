from collections.abc import Generator, Hashable
from itertools import combinations, islice
from math import comb

import networkx as nx
import pandas as pd
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.structure_score import BaseStructureScore, get_scoring_method
from pgmpy.utils._warnings import _warn_external
from pgmpy.utils.mathext import powerset


class ExhaustiveSearch(BaseCausalDiscovery):
    """
    Score-based causal discovery using an exhaustive search over all DAGs.

    Given a tabular dataset, the algorithm scores every Directed Acyclic Graph (DAG) over the variables in the data and
    returns the one with the highest score. Unlike heuristic searches such as HillClimbSearch or GES, the result is
    guaranteed to be a global optimum of the scoring method.

    The number of DAGs grows super-exponentially with the number of variables: there are 25 DAGs over 3 variables, 543
    over 4, 29,281 over 5, and 3,781,503 over 6. The search is therefore feasible only for a handful of variables.

    Parameters
    ----------
    scoring_method : str or BaseStructureScore instance, default=None
        The score to be optimized during structure estimation. Please refer :doc:`/api/structure_score` for a list of
        available scoring methods.

        If ``None``, the appropriate scoring method is automatically selected based on the data type. If a string is
        provided, the corresponding scoring method is instantiated with default parameters. To customize score-specific
        parameters, please pass an instance of the scoring class.

    return_type : str, default='pdag'
        The type of graph to return. Options are:

        - 'dag': Returns the highest scoring directed acyclic graph (DAG).
        - 'pdag': Returns the partially directed acyclic graph (PDAG) representing the Markov equivalence class of the
          highest scoring DAG.

    show_progress : bool, default=True
        If True, shows a progress bar while scoring the DAGs.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph. If multiple DAGs have the highest score, the one with the fewest edges is used.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    dag_scores_ : list of (float, tuple) pairs
        The score of every DAG over the variables, sorted from the highest to the lowest score. Each element is a
        ``(score, edges)`` pair, where ``edges`` is the sorted tuple of the ``(parent, child)`` edges of the DAG.

    scoring_method_ : BaseStructureScore
        The structure score used for learning, resolved from ``scoring_method``.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.example_models import load_model
    >>> model = load_model("bnlearn/cancer")
    >>> df = model.simulate(n_samples=1000, rng=42)[["Smoker", "Cancer", "Xray"]]

    Use the ExhaustiveSearch algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import ExhaustiveSearch
    >>> est = ExhaustiveSearch(scoring_method="bic-d", return_type="dag")
    >>> est.fit(df)
    ExhaustiveSearch(return_type='dag', scoring_method='bic-d')
    >>> est.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.DAG.DAG object at 0x...>

    The scores of all the DAGs are available after fitting, with the learned DAG first:

    >>> len(est.dag_scores_)
    25
    >>> best_score, best_edges = est.dag_scores_[0]
    >>> best_edges == tuple(sorted(est.causal_graph_.edges()))
    True

    References
    ----------
    - :footcite:t:`koller_friedman_2009`
    """

    _tags = {
        "name": "exhaustive_search",
        "data_types": ["discrete", "continuous", "mixed"],
        "identifiable_graph": "cpdag",
        "requires_target": False,
        "capability:multivariate": True,
        "capability:expert_knowledge": [],
        "assumption:causal_sufficiency": True,
        "assumption:acyclicity": True,
        "assumption:faithfulness": True,
        "assumption:linearity": False,
        "assumption:additive_noise": False,
        "assumption:gaussian_noise": False,
        "assumption:non_gaussian_noise": False,
        "assumption:low_noise": False,
    }

    def __init__(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        return_type: str = "pdag",
        show_progress: bool = True,
    ):
        self.scoring_method = scoring_method
        self.return_type = return_type
        self.show_progress = show_progress

    @staticmethod
    def _all_dags(nodes: list[Hashable]) -> Generator[nx.DiGraph]:
        """
        Generates all the DAGs over `nodes`, ordered by the number of edges, with the empty DAG first.

        `2**(n*(n-1))` candidate directed graphs are checked for `n` nodes, so this is likely not feasible for n>6.

        Parameters
        ----------
        nodes : list
            The nodes of the generated DAGs.

        Warns
        -----
        UserWarning
            If more than six nodes would make exhaustive enumeration impractical.

        Returns
        -------
        dags : Generator of nx.DiGraph
            All the DAGs over `nodes`.
        """
        if len(nodes) > 6:
            candidate_count = 2 ** (len(nodes) * (len(nodes) - 1))
            _warn_external(
                f"Exhaustive search over {len(nodes)} nodes examines {candidate_count:,} candidate directed graphs. "
                "Reduce the node set or use a non-exhaustive estimator.",
                UserWarning,
            )

        edges = list(combinations(nodes, 2))
        edges.extend([(y, x) for x, y in edges])
        all_graphs = powerset(edges)

        for graph_edges in all_graphs:
            graph = nx.DiGraph(graph_edges)
            graph.add_nodes_from(nodes)
            if nx.is_directed_acyclic_graph(graph):
                yield graph

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the ExhaustiveSearch algorithm.

        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            The data to learn the causal structure from. If a numpy array is
            passed, then the column names would be x0 to x(n_features-1).

        Returns
        -------
        self : pgmpy.causal_discovery.ExhaustiveSearch
            Returns the instance with the fitted attributes.
        """
        # Step 0: Check the arguments.
        if self.return_type.lower() not in ("dag", "pdag"):
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        # Step 1: Resolve the scoring method.
        self.scoring_method_ = get_scoring_method(self.scoring_method, X)

        # Step 2: Count the DAGs over the variables using Robinson's recurrence.
        nodes = sorted(X.columns)

        n_dags = [1]
        for n in range(1, len(nodes) + 1):
            n_dags.append(
                sum((-1) ** (k + 1) * comb(n, k) * 2 ** (k * (n - k)) * n_dags[n - k] for k in range(1, n + 1))
            )

        # Step 3: Score every DAG, stopping the enumeration after the last one.
        dag_scores = []
        for dag in tqdm(
            islice(self._all_dags(nodes), n_dags[-1]),
            total=n_dags[-1],
            desc="Scoring DAGs",
            disable=not (self.show_progress and config.SHOW_PROGRESS),
        ):
            dag_scores.append((self.scoring_method_.score(dag), tuple(sorted(dag.edges()))))

        # Step 4: Rank the DAGs by score, keeping the sparsest first among equal scores.
        self.dag_scores_ = sorted(dag_scores, key=lambda dag_score: dag_score[0], reverse=True)

        # Step 5: Store the results.
        best_model = DAG()
        best_model.add_nodes_from(nodes)
        best_model.add_edges_from(self.dag_scores_[0][1])

        if self.return_type.lower() == "dag":
            self.causal_graph_ = best_model
        else:
            self.causal_graph_ = best_model.to_pdag()

        self.adjacency_matrix_ = self.causal_graph_.to_adjacency(
            encoding="binary", nodelist=list(self.causal_graph_.nodes())
        )

        return self
