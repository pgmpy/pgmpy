import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG, PDAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery, _ScoreMixin
from pgmpy.estimators.StructureScore import StructureScore, get_scoring_method


class BOSS(_ScoreMixin, _BaseCausalDiscovery):
    """
    Score-based causal discovery using Best Order Score Search (BOSS).

    This class implements the BOSS algorithm for causal discovery. Given a
    tabular dataset, the algorithm estimates the causal structure among the
    variables in the data as a Directed Acyclic Graph (DAG) or a Completed
    Partially Directed Acyclic Graph (CPDAG).

    BOSS is a permutation-based algorithm that greedily searches over orderings
    of variables. Unlike graph-space algorithms (e.g., HillClimbSearch, GES),
    BOSS operates in the space of variable permutations and constructs DAGs
    from permutations by selecting parent sets using the Grow-Shrink (GS)
    procedure.

    The algorithm proceeds in three phases:

    1. **Permutation search**:
       Greedy optimization of the variable ordering using the best-move
       operator (Algorithm 5). For each variable, all possible insertion
       positions are evaluated, and moves that improve the score are kept.
       The process repeats until a full pass yields no improvement.

    2. **DAG construction**:
       The Grow-Shrink procedure (Algorithm 3) selects parent sets for each
       variable from its predecessors in the permutation, and a DAG is
       constructed accordingly.

    3. **BES phase**:
       Backward Equivalence Search refines the learned structure by removing
       edges within the Markov equivalence class to improve the score. This
       step is required for asymptotic correctness.

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

    return_type : str, default='dag'
    The type of graph to return. Options are:

    - 'dag': Returns a directed acyclic graph (DAG).
    - 'pdag': Returns a completed partially directed acyclic graph (CPDAG).
    - 'cpdag': Alias for 'pdag'

    use_cache : bool, default=True
    If True, uses caching of local scores for faster computation.
    Note: Caching only works for decomposable scoring methods.

    random_state : int or None, default=None
    Seed for the random number generator used to create the initial
    permutation.

    max_iter : int, default=1000
    Maximum number of outer iterations of permutation search.
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
    >>> boss.fit(df) BOSS(random_state=42, scoring_method='bic-d')
    >>> boss.causal_graph_ # doctest: +ELLIPSIS
    <pgmpy.base...object at 0x...>
    >>> boss.n_features_in_
    37

    Notes
    -----
    This implementation follows Algorithms 3–5 from the BOSS paper but does
    not include Grow-Shrink Trees (GSTs). Instead, it relies on pgmpy's score
    caching along with additional caching for permutation scores and
    Grow-Shrink parent sets.

    The BES phase is implemented as a greedy edge-deletion procedure on a
    CPDAG using Meek's rules. This is a simplified variant of the full GES
    BES operator.
    """

    def __init__(
        self,
        scoring_method: str | StructureScore | None = None,
        return_type: str = "pdag",
        use_cache: bool = True,
        random_state: int | None = None,
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

        self._gs_cache: dict = {}
        self._perm_score_cache: dict = {}
        self.n_features_in_ = X.shape[1]

        _, score_c = get_scoring_method(self.scoring_method, X, self.use_cache)
        score_fn = score_c.local_score

        rng = np.random.default_rng(self.random_state)
        perm: list[str] = list(rng.permutation(self.variables_))

        for _ in range(self.max_iter):
            best_score = self._score_permutation(perm, score_fn)

            for v in list(perm):
                perm = self._best_move(perm, v, score_fn)

            if self._score_permutation(perm, score_fn) <= best_score:
                break

        dag = self._project_permutation(perm, score_fn)

        pdag = dag.to_pdag()

        model = self._run_bes(pdag, score_fn)

        rt = self.return_type.lower()

        if rt == "dag":
            self.causal_graph_ = model
        elif rt in {"pdag", "cpdag"}:
            self.causal_graph_ = model.to_pdag()
        else:
            raise ValueError(f"return_type must be one of: dag, pdag, cpdag. Got: {self.return_type}")

        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype=int)

        return self

    def _score_permutation(self, perm: list[str], score_fn) -> float:
        """
        Compute and cache the total BIC score for a permutation.

        The score is the sum of local scores obtained by running the
        Grow-Shrink procedure (via ``_grow_shrink_parents``) for every
        variable in the permutation order:

            T.score(π) = Σ_{v ∈ π} BIC(X_v, X_{GS(v, pre_π(v))})

        Parameters
        ----------
        perm : list[str]
            A permutation (ordering) of variables.

        score_fn : callable
            Local scoring function: ``score_fn(variable, parents) -> float``.

        Returns
        -------
        score : float
            Total score for the permutation.
        """
        key = tuple(perm)
        if key in self._perm_score_cache:
            return self._perm_score_cache[key]

        score = 0.0
        for idx, var in enumerate(perm):
            predecessors = perm[:idx]
            parents = self._grow_shrink_parents(var, predecessors, score_fn)
            score += score_fn(var, parents)

        self._perm_score_cache[key] = score
        return score

    def _best_move(self, perm: list[str], v: str, score_fn) -> list[str]:
        """
        Apply the best-move operator for a single variable — Algorithm 5.

        Tries moving ``v`` to every position ``i`` in ``{0, …, |π|-1}``.
        If inserting ``v`` at position ``i`` improves the total score, the
        move is *kept* and the search continues from the new permutation
        (i.e. multiple moves for the same variable within one call are
        possible). If a position does not improve the score, ``v`` is left
        at its current position and the next position is tried.

        Concretely this implements::

            best ← T.score(π)
            for i ← 1 to |π| do
                j ← π.index(v)
                π ← π.move(v, i)
                if best < T.score(π) then
                    best ← T.score(π)
                else
                    π ← π.move(v, j)   # revert

        Parameters
        ----------
        perm : list[str]
            Current permutation of variables.

        v : str
            The variable to move.

        score_fn : callable
            Local scoring function: ``score_fn(variable, parents) -> float``.

        Returns
        -------
        perm : list[str]
            Updated permutation (may equal the input if no move helped).
        """
        n = len(perm)
        best = self._score_permutation(perm, score_fn)

        for i in range(n):
            # j is the *current* index of v — recomputed each iteration
            # because a previous improving move may have shifted v.
            j = perm.index(v)

            # Build the candidate permutation: remove v from j, insert at i.
            candidate = perm.copy()
            candidate.pop(j)
            candidate.insert(i, v)

            candidate_score = self._score_permutation(candidate, score_fn)

            if candidate_score > best:
                # Improvement found — keep the move and continue.
                best = candidate_score
                perm = candidate
            # else: perm is unchanged; v stays at its current position j.

        return perm

    def _project_permutation(self, perm: list[str], score_fn) -> DAG:
        """
        Grow-Shrink (GS) projection — Algorithm 3 from the paper.

        For each variable in the permutation order, selects its parent set
        from the predecessors using Grow-Shrink and adds the corresponding
        directed edges to form a DAG.

            foreach v ∈ π do
                Z ← pre_π(v)
                W ← grow(X, v, Z)
                W ← shrink(X, v, W)
                foreach w ∈ W do  E ← E ∪ (v, w)
            G ← (V, E)

        Parameters
        ----------
        perm : list[str]
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

        for i, var in enumerate(perm):
            predecessors = perm[:i]
            parents: list[str] = self._grow_shrink_parents(var, predecessors, score_fn)
            for p in parents:
                dag.add_edge(p, var)

        return dag

    def _grow_shrink_parents(self, variable: str, candidates: list[str], score_fn) -> list[str]:
        """
        Grow-Shrink parent selection for a single variable — Algorithms 1 & 2.

        **Grow** (Algorithm 1): greedily add the candidate that most
        improves ``score_fn(variable, parents ∪ {w})`` until no candidate
        improves the score::

            W ← ∅
            repeat
                w ← argmax_{z ∈ Z} BIC(X_v, X_{W ∪ z})
                if w ≠ ∅ then W ← W ∪ w
            until w = ∅

        **Shrink** (Algorithm 2): remove any parent whose removal improves
        the score::

            repeat
                w ← argmax_{w ∈ W} BIC(X_v, X_{W \\ w})
                if w ≠ ∅ then W ← W \\ w
            until w = ∅

        Results are cached by ``(variable, frozenset(candidates))`` to avoid
        redundant computation when the same prefix is encountered again.

        Parameters
        ----------
        variable : str
            The target variable.

        candidates : list[str]
            Candidate parent variables (predecessors in permutation order).

        score_fn : callable
            Local scoring function.

        Returns
        -------
        parents : list[str]
            The selected parent set.
        """
        key = (variable, tuple(candidates))
        if key in self._gs_cache:
            return self._gs_cache[key]

        parents: list[str] = []
        current_score = score_fn(variable, parents)

        # --- Grow phase (Algorithm 1) ---
        while True:
            best_candidate = None
            best_score = current_score

            for c in candidates:
                if c in parents:
                    continue
                candidate_score = score_fn(variable, parents + [c])
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_candidate = c

            if best_candidate is None:
                break

            parents.append(best_candidate)
            current_score = best_score

        # --- Shrink phase (Algorithm 2) ---
        while True:
            improved = False

            for p in list(parents):
                reduced = [x for x in parents if x != p]
                candidate_score = score_fn(variable, reduced)
                if candidate_score > current_score:
                    parents = reduced
                    current_score = candidate_score
                    improved = True
                    break

            if not improved:
                break

        self._gs_cache[key] = parents
        return parents

    def _run_bes(self, pdag: PDAG, score_fn) -> DAG:
        """
        Backward Equivalence Search (BES) phase.

        Corresponds to the ``BES(G, X)`` call in Algorithm 4. Iteratively
        removes the directed edge whose removal yields the greatest score
        improvement, applying Meek's rules after each deletion to maintain
        a valid PDAG, until no improving deletion exists.

        This is a simplified BES operating on a PDAG (obtained by first
        converting the DAG to its CPDAG via ``find-compelled``/``to_pdag``).
        The full GES BES uses a more general edge-deletion operator; here we
        greedily remove the single best directed edge per iteration, which is
        sufficient to guarantee asymptotic correctness when combined with the
        permutation search (Proposition 2 in the paper).

        Parameters
        ----------
        pdag : pgmpy.base.PDAG
            The DAG produced by the permutation search and projection step.

        score_fn : callable
            Local scoring function: ``score_fn(variable, parents) -> float``.

        Returns
        -------
        dag : pgmpy.base.DAG
            The refined DAG after BES.
        """
        # find-compelled: convert DAG → CPDAG before BES (paper Algorithm 4).

        pdag = pdag.apply_meeks_rules(inplace=False)

        def parents(graph, node):
            return set(graph.directed_parents(node))

        def score_node(node, pa):
            return score_fn(node, list(pa))

        while True:
            best_delta = 0.0
            best_edge = None

            for x, y in list(pdag.directed_edges):
                if not pdag.has_directed_edge(x, y):
                    continue

                pa_y = parents(pdag, y)
                old_score = score_node(y, pa_y)
                new_score = score_node(y, pa_y - {x})

                delta = new_score - old_score

                if delta > best_delta:
                    best_delta = delta
                    best_edge = (x, y)

            if best_edge is None:
                break

            x, y = best_edge
            if (x, y) in pdag.directed_edges:
                pdag.directed_edges.remove((x, y))
            if pdag.has_edge(x, y):
                pdag.remove_edge(x, y)

            pdag = pdag.apply_meeks_rules(inplace=False)

        return pdag.to_dag()
