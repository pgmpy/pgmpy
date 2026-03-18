from typing import List, Optional, Tuple, Union

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery, _ScoreMixin
from pgmpy.estimators.StructureScore import StructureScore, get_scoring_method


class BOSS(_ScoreMixin, _BaseCausalDiscovery):
    """
    Score-based causal discovery using Best Order Score Search (BOSS).

    This class implements the BOSS algorithm [1]_ for causal discovery. Given a
    tabular dataset, the algorithm estimates the causal structure among the
    variables in the data as a Directed Acyclic Graph (DAG) or Partially
    Directed Acyclic Graph (PDAG).

    BOSS is a permutation-based algorithm that greedily searches over orderings
    of variables. Unlike graph-space algorithms (e.g., HillClimbSearch, GES),
    BOSS works in the space of variable permutations and constructs DAGs from
    permutations using the Grow-Shrink (GS) procedure. The algorithm proceeds
    in three phases:

        1. **Permutation search**: Greedy optimization of the variable ordering
           using the best-move operator, which tries moving each variable to
           earlier positions in the permutation.
        2. **DAG construction**: The Grow-Shrink procedure builds a DAG from
           the optimized permutation by greedily selecting parents for each
           variable from its predecessors.
        3. **BES phase**: Backward Equivalence Search removes spurious edges
           to ensure asymptotic correctness.

    Parameters
    ----------
    scoring_method : str or StructureScore instance, default=None
        The score to be optimized during structure estimation. Supported
        structure scores:

        - Discrete data: 'k2', 'bdeu', 'bds', 'bic-d', 'aic-d'
        - Continuous data: 'll-g', 'aic-g', 'bic-g'
        - Mixed data: 'll-cg', 'aic-cg', 'bic-cg'

        If None, the appropriate scoring method is automatically selected based
        on the data type. BIC is recommended per the paper.

    return_type : str, default='pdag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    use_cache : bool, default=True
        If True, uses caching of local scores for faster computation.
        Note: Caching only works for scoring methods which are decomposable.

    random_state : int or None, default=None
        Seed for the random number generator used to create the initial
        permutation. If None, the initial permutation is non-deterministic.
        Uses ``np.random.default_rng(random_state)`` for modern seeding.

    max_iter : int, default=1000
        The maximum number of permutation search iterations. The algorithm
        terminates when no improving move is found or this limit is reached.

    Attributes
    ----------
    causal_graph_ : DAG or PDAG
        The learned causal graph at a (local) score maximum.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> import numpy as np
    >>> from pgmpy.utils import get_example_model
    >>> np.random.seed(42)
    >>> model = get_example_model("alarm")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the BOSS algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import BOSS
    >>> boss = BOSS(scoring_method="bic-d", random_state=42)
    >>> boss.fit(df)
    BOSS(random_state=42, scoring_method='bic-d')
    >>> boss.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base...object at 0x...>
    >>> boss.n_features_in_
    37

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., Sanchez-Romero, R., Camchong, J., &
           Kummerfeld, E. (2023). "Fast Scalable and Accurate Discovery of
           DAGs Using the Best Order Score Search and Grow-Shrink Trees."
           Advances in Neural Information Processing Systems (NeurIPS).
           arXiv:2310.17679.
    """

    def __init__(
        self,
        scoring_method: Optional[Union[str, StructureScore]] = None,
        return_type: str = "pdag",
        use_cache: bool = True,
        random_state: Optional[int] = None,
        max_iter: int = 1000,
    ):
        self.scoring_method = scoring_method
        self.return_type = return_type
        self.use_cache = use_cache
        self.random_state = random_state
        self.max_iter = max_iter

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the BOSS algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.BOSS
            Returns the instance with the fitted attributes.
        """
        self.variables_ = list(X.columns)

        _, score_c = get_scoring_method(self.scoring_method, X, self.use_cache)
        score_fn = score_c.local_score

        # Step 1: Initialize a random permutation of variables.
        rng = np.random.default_rng(self.random_state)
        perm = list(rng.permutation(self.variables_))

        # Step 2: Greedy permutation search via best-move operator.
        for _ in range(self.max_iter):
            perm, improved = self._best_move(perm, score_fn)
            if not improved:
                break

        # Step 3: Construct DAG from the converged permutation.
        current_model = self._project_permutation(perm, score_fn)

        # Step 4: Run BES for asymptotic correctness (always executed).
        current_model = self._run_bes(current_model, score_fn)

        # Step 5: Store results.
        if self.return_type.lower() == "dag":
            self.causal_graph_ = current_model
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = current_model.to_pdag()
        else:
            raise ValueError(
                f"return_type must be one of: dag, pdag. Got: {self.return_type}"
            )

        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, weight=1, dtype="int"
        )

        return self

    def _best_move(self, perm: List, score_fn) -> Tuple[List, bool]:
        """
        Greedy best-move operator over the permutation.

        For each variable in the permutation, considers moving it to every
        earlier position and selects the move that maximizes the total score
        improvement. The score change is computed efficiently by re-running
        ``_project_permutation`` only on the affected sub-permutation.

        Parameters
        ----------
        perm : list
            Current permutation of variables.

        score_fn : callable
            Local scoring function: ``score_fn(variable, parents) -> float``.

        Returns
        -------
        new_perm : list
            The improved permutation (or the original if no improvement found).

        improved : bool
            Whether an improving move was found.
        """
        best_perm = perm
        best_delta = 0.0
        n = len(perm)

        # Compute current scores for each variable under the current permutation.
        current_parents = {}
        current_scores = {}
        for idx, var in enumerate(perm):
            predecessors = perm[:idx]
            parents = self._grow_shrink_parents(var, predecessors, score_fn)
            current_parents[var] = parents
            current_scores[var] = score_fn(var, parents)

        for i in range(1, n):
            var = perm[i]
            for j in range(i):
                # Create candidate permutation: move var from position i to position j.
                candidate = perm[:j] + [var] + perm[j:i] + perm[i + 1 :]

                # Only variables between positions j and i (inclusive of original
                # position range) are affected.
                delta = 0.0
                for k in range(j, i + 1):
                    v = candidate[k]
                    predecessors = candidate[:k]
                    new_parents = self._grow_shrink_parents(v, predecessors, score_fn)
                    new_score = score_fn(v, new_parents)
                    delta += new_score - current_scores[v]

                if delta > best_delta:
                    best_delta = delta
                    best_perm = candidate

        return best_perm, best_delta > 0

    def _project_permutation(self, perm: List, score_fn) -> DAG:
        """
        Grow-Shrink (GS) procedure to construct a DAG from a permutation.

        For each variable in the permutation order, selects its parent set
        from the predecessors using a greedy grow phase (add the parent that
        most improves the score) followed by a shrink phase (remove any parent
        whose removal improves the score).

        Parameters
        ----------
        perm : list
            A permutation (ordering) of variables.

        score_fn : callable
            Local scoring function: ``score_fn(variable, parents) -> float``.

        Returns
        -------
        dag : pgmpy.base.DAG
            The DAG constructed from the permutation.
        """
        dag = DAG()
        dag.add_nodes_from(perm)

        for idx, var in enumerate(perm):
            predecessors = perm[:idx]
            parents = self._grow_shrink_parents(var, predecessors, score_fn)
            for p in parents:
                dag.add_edge(p, var)

        return dag

    @staticmethod
    def _grow_shrink_parents(variable, candidates: List, score_fn) -> List:
        """
        Grow-Shrink parent selection for a single variable.

        Greedily grows the parent set by adding candidates that improve the
        local score, then shrinks by removing parents whose removal improves
        the score.

        Parameters
        ----------
        variable : hashable
            The target variable.

        candidates : list
            Candidate parent variables (predecessors in permutation order).

        score_fn : callable
            Local scoring function.

        Returns
        -------
        parents : list
            The selected parent set.
        """
        parents = []

        # Grow phase: greedily add parents that improve the score.
        improved = True
        while improved:
            improved = False
            best_candidate = None
            best_score = score_fn(variable, parents)

            for c in candidates:
                if c not in parents:
                    candidate_parents = parents + [c]
                    s = score_fn(variable, candidate_parents)
                    if s > best_score:
                        best_score = s
                        best_candidate = c

            if best_candidate is not None:
                parents.append(best_candidate)
                improved = True

        # Shrink phase: remove parents whose removal improves the score.
        improved = True
        while improved:
            improved = False
            current_score = score_fn(variable, parents)

            for p in list(parents):
                reduced_parents = [x for x in parents if x != p]
                s = score_fn(variable, reduced_parents)
                if s > current_score:
                    parents = reduced_parents
                    current_score = s
                    improved = True
                    break

        return parents

    @staticmethod
    def _run_bes(dag: DAG, score_fn) -> DAG:
        """
        Backward Equivalence Search (BES) phase.

        Iteratively removes edges from the DAG whose removal improves the
        total score, ensuring the graph remains a DAG. This phase is always
        executed for asymptotic correctness as recommended by the paper.

        Parameters
        ----------
        dag : pgmpy.base.DAG
            The DAG produced by the permutation search.

        score_fn : callable
            Local scoring function: ``score_fn(variable, parents) -> float``.

        Returns
        -------
        dag : pgmpy.base.DAG
            The refined DAG after BES.
        """
        improved = True
        while improved:
            improved = False
            best_edge = None
            best_delta = 0.0

            for u, v in list(dag.edges()):
                current_parents = list(dag.predecessors(v))
                new_parents = [p for p in current_parents if p != u]
                delta = score_fn(v, new_parents) - score_fn(v, current_parents)

                if delta > best_delta:
                    best_delta = delta
                    best_edge = (u, v)

            if best_edge is not None:
                dag.remove_edge(*best_edge)
                improved = True

        return dag
