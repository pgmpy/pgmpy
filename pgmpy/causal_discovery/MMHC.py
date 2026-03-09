from typing import Optional, Union

import networkx as nx
import pandas as pd

from pgmpy.base import UndirectedGraph
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.causal_discovery.HillClimbSearch import HillClimbSearch
from pgmpy.estimators import ExpertKnowledge
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators.StructureScore import StructureScore
from pgmpy.utils.mathext import powerset


class MMHC(_BaseCausalDiscovery):
    """
    Hybrid causal discovery using Max-Min Hill-Climbing (MMHC).

    This class implements the MMHC algorithm [1]_ for causal discovery.
    Given a tabular dataset, the algorithm estimates the causal structure
    among the variables in the data as a Directed Acyclic Graph (DAG) or
    Partially Directed Acyclic Graph (PDAG).

    MMHC works in two phases:

    1. **MMPC phase** (constraint-based): Discovers an undirected skeleton of
       the causal graph using the Max-Min Parents and Children (MMPC) algorithm,
       which relies on conditional independence tests.
    2. **Hill-Climbing phase** (score-based): Orients the skeleton edges using
       the sklearn-compatible :class:`~pgmpy.causal_discovery.HillClimbSearch`
       estimator, restricted to edges present in the skeleton.

    Parameters
    ----------
    ci_test : str, default='chi_square'
        The conditional independence test to use during the MMPC skeleton
        discovery phase. Currently ``'chi_square'`` is supported for discrete
        data.

    significance_level : float, default=0.01
        The significance level used for conditional independence tests in the
        MMPC phase. A lower value produces a sparser skeleton.

    scoring_method : str or StructureScore instance, default=None
        The score to be optimised during the Hill-Climbing phase. Supported
        structure scores:

        - Discrete data: ``'k2'``, ``'bdeu'``, ``'bds'``, ``'bic-d'``, ``'aic-d'``
        - Continuous data: ``'ll-g'``, ``'aic-g'``, ``'bic-g'``
        - Mixed data: ``'ll-cg'``, ``'aic-cg'``, ``'bic-cg'``

        If ``None``, an appropriate scoring method is selected automatically
        based on data type.

    tabu_length : int, default=10
        The number of recent graph modifications stored in the tabu list during
        the Hill-Climbing phase. These modifications cannot be immediately
        reversed, which encourages broader search-space exploration.

    return_type : str, default='pdag'
        Type of graph to return after fitting:

        - ``'dag'``: Returns a Directed Acyclic Graph.
        - ``'pdag'``: Returns a Partially Directed Acyclic Graph (equivalence
          class representation).

    use_cache : bool, default=True
        If ``True``, caches local scores during the Hill-Climbing phase for
        faster computation. Caching only works correctly for decomposable
        scoring methods.

    show_progress : bool, default=True
        If ``True``, shows a progress bar during the Hill-Climbing phase.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph after calling :meth:`fit`.

    skeleton_ : pgmpy.base.UndirectedGraph
        The undirected skeleton discovered by the MMPC phase.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        Number of features seen during :meth:`fit`.

    feature_names_in_ : np.ndarray
        Feature names seen during :meth:`fit`.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(2500, 4)), columns=list("XYZW")
    ... )
    >>> data["sum"] = data.sum(axis=1)

    Use MMHC to learn the causal structure:

    >>> from pgmpy.causal_discovery import MMHC
    >>> est = MMHC(show_progress=False)
    >>> est.fit(data)
    >>> est.causal_graph_.edges()
    >>> est.skeleton_.edges()

    Inspect the intermediate MMPC skeleton and the final graph:

    >>> est.skeleton_.edges()   # undirected skeleton from MMPC phase
    >>> est.causal_graph_.edges()  # oriented edges from HC phase

    References
    ----------
    .. [1] Tsamardinos et al., The Max-Min Hill-Climbing Bayesian Network
           Structure Learning Algorithm (2005).
           http://www.dsl-lab.org/supplements/mmhc_paper/paper_online.pdf
    """

    def __init__(
        self,
        ci_test: str = "chi_square",
        significance_level: float = 0.01,
        scoring_method: Optional[Union[str, StructureScore]] = None,
        tabu_length: int = 10,
        return_type: str = "pdag",
        use_cache: bool = True,
        show_progress: bool = True,
    ):
        self.ci_test = ci_test
        self.significance_level = significance_level
        self.scoring_method = scoring_method
        self.tabu_length = tabu_length
        self.return_type = return_type
        self.use_cache = use_cache
        self.show_progress = show_progress

    def _mmpc(self, X: pd.DataFrame) -> UndirectedGraph:
        """
        Estimates an undirected skeleton using the MMPC algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The dataset to learn the skeleton from.

        Returns
        -------
        skeleton : pgmpy.base.UndirectedGraph
            The undirected skeleton of the Bayesian Network.

        References
        ----------
        Tsamardinos et al., The Max-Min Hill-Climbing Bayesian Network
        Structure Learning Algorithm (2005), Algorithms 1 & 2.
        http://www.dsl-lab.org/supplements/mmhc_paper/paper_online.pdf
        """
        nodes = list(X.columns)

        def assoc(var_x, var_y, conditioning_set):
            """Negative p-value of the CI test — higher means more associated."""
            return 1 - chi_square(
                var_x, var_y, conditioning_set, X, boolean=False
            )[1]

        def min_assoc(var_x, var_y, conditioning_vars):
            """Minimal association of var_x, var_y over all subsets of conditioning_vars."""
            return min(
                assoc(var_x, var_y, subset)
                for subset in powerset(conditioning_vars)
            )

        def max_min_heuristic(node, current_neighbors):
            """Return the variable that maximises min_assoc with node given current_neighbors."""
            max_min_assoc = 0
            best_candidate = None

            for candidate in set(nodes) - set(current_neighbors + [node]):
                candidate_assoc = min_assoc(node, candidate, current_neighbors)
                if candidate_assoc >= max_min_assoc:
                    best_candidate = candidate
                    max_min_assoc = candidate_assoc

            return best_candidate, max_min_assoc

        # Find candidate parents and children for each node
        neighbors = {node: [] for node in nodes}

        for node in nodes:
            # Forward Phase: greedily add variables that maximise min association
            while True:
                new_neighbor, new_neighbor_min_assoc = max_min_heuristic(
                    node, neighbors[node]
                )
                if new_neighbor_min_assoc > 0:
                    neighbors[node].append(new_neighbor)
                else:
                    break

            # Backward Phase: remove false positives via CI tests
            for neighbor in list(neighbors[node]):
                other_neighbors = [n for n in neighbors[node] if n != neighbor]
                for sep_set in powerset(other_neighbors):
                    if chi_square(
                        X=node,
                        Y=neighbor,
                        Z=sep_set,
                        data=X,
                        significance_level=self.significance_level,
                    ):
                        neighbors[node].remove(neighbor)
                        break

        # Symmetry correction: keep an edge only if both endpoints agree
        for node in nodes:
            for neighbor in list(neighbors[node]):
                if node not in neighbors[neighbor]:
                    neighbors[node].remove(neighbor)

        skeleton = UndirectedGraph()
        skeleton.add_nodes_from(nodes)
        for node in nodes:
            skeleton.add_edges_from([(node, nbr) for nbr in neighbors[node]])

        return skeleton

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the MMHC algorithm.

        Phase 1 runs MMPC to discover an undirected skeleton.
        Phase 2 runs HillClimbSearch (sklearn-compatible) to orient the edges,
        with all non-skeleton edges forbidden.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.MMHC
            The fitted estimator.
        """
        self.variables_ = list(X.columns)

        # --- Phase 1: MMPC skeleton discovery ---
        skeleton = self._mmpc(X)
        self.skeleton_ = skeleton

        # Build the set of forbidden edges:
        # any directed edge not present in the skeleton (in either direction) is forbidden.
        possible_edges = set(
            nx.complete_graph(n=self.variables_, create_using=nx.Graph).edges()
        )
        skeleton_edges = set(skeleton.to_directed().edges())
        forbidden_edges = list(possible_edges - skeleton_edges)

        expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)

        # --- Phase 2: Score-based orientation via HillClimbSearch ---
        hc = HillClimbSearch(
            scoring_method=self.scoring_method,
            tabu_length=self.tabu_length,
            expert_knowledge=expert_knowledge,
            return_type=self.return_type,
            use_cache=self.use_cache,
            show_progress=self.show_progress,
        )
        hc.fit(X)

        self.causal_graph_ = hc.causal_graph_
        self.adjacency_matrix_ = hc.adjacency_matrix_

        return self
