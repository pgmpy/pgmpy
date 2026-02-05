"""
MMHC (Max-Min Hill-Climbing) causal discovery algorithm.

Sklearn-compatible hybrid causal discovery: MMPC skeleton + HillClimbSearch orientation.
"""

from typing import Optional, Union

import pandas as pd

from pgmpy.base import UndirectedGraph
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.causal_discovery.HillClimbSearch import HillClimbSearch
from pgmpy.estimators import ExpertKnowledge
from pgmpy.estimators.CITests import chi_square
from pgmpy.estimators.StructureScore import StructureScore
from pgmpy.utils import get_dataset_type
from pgmpy.utils.mathext import powerset


class MMHC(_BaseCausalDiscovery):
    """
    Hybrid causal discovery using MMPC skeleton and HillClimbSearch orientation.

    This class implements the MMHC (Max-Min Hill-Climbing) algorithm [1]_ for
    causal discovery. Given a tabular dataset, the algorithm estimates the
    causal structure in two phases:

    1. **Skeleton (MMPC)**: Learns an undirected skeleton using the Max-Min
       Parents-and-Children algorithm with conditional independence tests.
    2. **Orientation (HillClimbSearch)**: Orients the skeleton using score-based
       local search (hill climbing), restricting the search to edges present
       in the skeleton.

    The algorithm is designed for discrete data and uses the chi-square
    conditional independence test in the MMPC phase.

    Parameters
    ----------
    scoring_method : str or StructureScore instance, default=None
        The score to be optimized during the orientation phase. Supported
        structure scores for discrete data: 'k2', 'bdeu', 'bds', 'bic-d', 'aic-d'.
        If None, defaults to 'bdeu' (with equivalent_sample_size=10) for
        discrete data.

    significance_level : float, default=0.01
        The significance level for conditional independence tests in the
        MMPC skeleton phase. Lower values yield sparser skeletons.

    tabu_length : int, default=10
        The number of recent graph modifications to store in the tabu list
        during HillClimbSearch. Serves to explore the search space better.

    max_indegree : int or None, default=None
        If provided, the orientation phase only considers DAGs where all nodes
        have at most `max_indegree` parents.

    return_type : str, default='dag'
        The type of graph to return. Options are:
        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    use_cache : bool, default=True
        If True, uses caching of local scores during orientation.
        Only safe for decomposable scoring methods.

    show_progress : bool, default=True
        If True, shows a progress bar during the orientation phase.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph.

    skeleton_ : UndirectedGraph
        The learned undirected skeleton from the MMPC phase.

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
    >>> from pgmpy.causal_discovery import MMHC
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(2500, 4)), columns=list("XYZW")
    ... )
    >>> data["sum"] = data.sum(axis=1)
    >>> data = data.astype("category")  # MMHC requires discrete (categorical) data
    >>> mmhc = MMHC(scoring_method="bdeu", significance_level=0.01)
    >>> mmhc.fit(data)
    >>> mmhc.causal_graph_.edges()

    References
    ----------
    .. [1] Tsamardinos et al., The max-min hill-climbing Bayesian network
           structure learning algorithm (2005),
           http://www.dsl-lab.org/supplements/mmhc_paper/paper_online.pdf
    """

    def __init__(
        self,
        scoring_method: Optional[Union[str, StructureScore]] = None,
        significance_level: float = 0.01,
        tabu_length: int = 10,
        max_indegree: Optional[int] = None,
        return_type: str = "dag",
        use_cache: bool = True,
        show_progress: bool = True,
    ):
        self.scoring_method = scoring_method
        self.significance_level = significance_level
        self.tabu_length = tabu_length
        self.max_indegree = max_indegree
        self.return_type = return_type
        self.use_cache = use_cache
        self.show_progress = show_progress

    def _mmpc(self, X: pd.DataFrame) -> UndirectedGraph:
        """Estimate graph skeleton using the MMPC (Max-Min Parents-and-Children) algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the skeleton from.

        Returns
        -------
        skeleton : UndirectedGraph
            An estimate for the undirected graph skeleton.
        """
        nodes = list(X.columns)

        def assoc(X_var, Y_var, Zs):
            """Measure for (conditional) association; use 1 - p-value of independence test."""
            return 1 - chi_square(
                X_var, Y_var, list(Zs), data=X, boolean=False
            )[1]

        def min_assoc(X_var, Y_var, Zs):
            """Minimal association of X, Y given any subset of Zs."""
            return min(
                assoc(X_var, Y_var, Zs_subset) for Zs_subset in powerset(Zs)
            )

        def max_min_heuristic(X_var, Zs):
            """Find variable that maximizes min_assoc with node relative to neighbors."""
            max_min_assoc = 0
            best_Y = None
            for Y in set(nodes) - set(Zs + [X_var]):
                min_assoc_val = min_assoc(X_var, Y, Zs)
                if min_assoc_val >= max_min_assoc:
                    best_Y = Y
                    max_min_assoc = min_assoc_val
            return (best_Y, max_min_assoc)

        neighbors = {}
        for node in nodes:
            neighbors[node] = []

            # Forward Phase
            while True:
                new_neighbor, new_neighbor_min_assoc = max_min_heuristic(
                    node, neighbors[node]
                )
                if new_neighbor_min_assoc > 0:
                    neighbors[node].append(new_neighbor)
                else:
                    break

            # Backward Phase
            for neigh in list(neighbors[node]):
                other_neighbors = [n for n in neighbors[node] if n != neigh]
                for sep_set in powerset(other_neighbors):
                    if chi_square(
                        X=node,
                        Y=neigh,
                        Z=list(sep_set),
                        data=X,
                        boolean=True,
                        significance_level=self.significance_level,
                    ):
                        neighbors[node].remove(neigh)
                        break

        # Symmetrize: keep edge only if both sides have it
        for node in nodes:
            for neigh in list(neighbors[node]):
                if node not in neighbors[neigh]:
                    neighbors[node].remove(neigh)

        skel = UndirectedGraph()
        skel.add_nodes_from(nodes)
        for node in nodes:
            skel.add_edges_from([(node, neigh) for neigh in neighbors[node]])

        return skel

    def _fit(self, X: pd.DataFrame) -> "MMHC":
        """
        Fit the MMHC algorithm: MMPC skeleton + HillClimbSearch orientation.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : MMHC
            The fitted estimator.
        """
        self.variables_ = list(X.columns)

        # MMHC uses chi-square CI tests (discrete only); require discrete data
        if get_dataset_type(X) != "discrete":
            raise ValueError(
                "MMHC only supports discrete data. Please use a DataFrame with "
                "categorical columns (e.g. df.astype('category'))."
            )

        # Phase 1: Learn skeleton with MMPC
        self.skeleton_ = self._mmpc(X)

        # Phase 2: Restrict orientation to skeleton edges only
        skeleton_edge_set = set(frozenset([u, v]) for u, v in self.skeleton_.edges())
        forbidden_edges = [
            (u, v)
            for u in self.variables_
            for v in self.variables_
            if u != v and frozenset([u, v]) not in skeleton_edge_set
        ]
        expert_knowledge = ExpertKnowledge(forbidden_edges=forbidden_edges)

        # Default scoring for discrete data (match old MmhcEstimator: BDeu)
        scoring_method = self.scoring_method
        if scoring_method is None:
            scoring_method = "bdeu"

        # Phase 3: Orient using HillClimbSearch
        hc = HillClimbSearch(
            scoring_method=scoring_method,
            tabu_length=self.tabu_length,
            max_indegree=self.max_indegree,
            expert_knowledge=expert_knowledge,
            return_type=self.return_type,
            use_cache=self.use_cache,
            show_progress=self.show_progress,
        )
        hc.fit(X)

        self.causal_graph_ = hc.causal_graph_
        self.adjacency_matrix_ = hc.adjacency_matrix_
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = X.columns

        return self
