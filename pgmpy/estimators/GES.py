from collections.abc import Hashable
from itertools import combinations

import numpy as np
import pandas as pd

from pgmpy.base import PDAG
from pgmpy.estimators import ExpertKnowledge, StructureEstimator, StructureScore
from pgmpy.estimators.ScoreCache import ScoreCache
from pgmpy.estimators.StructureScore import get_scoring_method
from pgmpy.global_vars import logger


class GES(StructureEstimator):
    """Implementation of Greedy Equivalence Search (GES) causal discovery algorithm.

    GES is a score-based causal discovery algorithm that searches over the space
    of equivalence classes (CPDAGs) using two phases:
        1. Forward phase: Insert edges using the Insert operator until no improvement.
        2. Backward phase: Delete edges using the Delete operator until no improvement.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.

    use_caching: boolean
        If True, uses caching of score for faster computation.

    References
    ----------
    Chickering, David Maxwell. "Optimal structure identification with greedy search."
      Journal of machine learning research 3.Nov (2002): 507-554.

    """

    def __init__(self, data: pd.DataFrame, use_cache: bool = True, **kwargs):
        self.use_cache = use_cache
        super().__init__(data=data, **kwargs)

    def _neighbors_of_in_pdag(self, pdag: PDAG, node: Hashable) -> set[Hashable]:
        """Returns the set of neighbors (undirected adjacencies) of node in pdag."""
        return pdag.undirected_neighbors(node)

    def _adjacent_in_pdag(self, pdag: PDAG, node: Hashable) -> set[Hashable]:
        """Returns all nodes adjacent to node (directed or undirected)."""
        return pdag.all_neighbors(node)

    def _is_clique(self, pdag: PDAG, nodes: set[Hashable]) -> bool:
        """Check if the given set of nodes forms a clique in the PDAG."""
        for u, v in combinations(nodes, 2):
            if not pdag.is_adjacent(u, v):
                return False
        return True

    def _valid_insert_operators(
        self,
        pdag: PDAG,
        expert_knowledge: ExpertKnowledge,
    ) -> list[tuple[Hashable, Hashable, set[Hashable]]]:
        """Find all valid Insert(x, y, T) operators for the current PDAG.

        An Insert(x, y, T) operator adds edge x -> y and orients previously
        undirected edges from T to y as t -> y.

        Validity conditions (Chickering 2002, Theorem 15):
        1. x and y are not adjacent in pdag
        2. T is a subset of neighbors of y that are not adjacent to x
        3. T union neighbors of y that are adjacent to x is a clique
        """
        operators = []
        nodes = list(pdag.nodes())

        for x in nodes:
            for y in nodes:
                if x == y:
                    continue
                if pdag.is_adjacent(x, y):
                    continue
                if (x, y) in expert_knowledge.forbidden_edges:
                    continue

                neighbors_y = self._neighbors_of_in_pdag(pdag, y)
                neighbors_y_adj_x = {n for n in neighbors_y if pdag.is_adjacent(n, x)}
                neighbors_y_not_adj_x = neighbors_y - neighbors_y_adj_x

                for t_size in range(len(neighbors_y_not_adj_x) + 1):
                    for T in combinations(neighbors_y_not_adj_x, t_size):
                        T = set(T)
                        clique_set = T | neighbors_y_adj_x
                        if self._is_clique(pdag, clique_set):
                            operators.append((x, y, T))

        return operators

    def _valid_delete_operators(
        self,
        pdag: PDAG,
        expert_knowledge: ExpertKnowledge,
    ) -> list[tuple[Hashable, Hashable, set[Hashable]]]:
        """Find all valid Delete(x, y, H) operators for the current PDAG.

        A Delete(x, y, H) operator removes edge x - y (or x -> y) and orients
        previously undirected edges from H to y as h -> y.

        Validity conditions (Chickering 2002, Theorem 17):
        1. x and y are adjacent in pdag
        2. H is a subset of neighbors of y that are adjacent to x
        3. H union neighbors of y that are not adjacent to x is a clique
        """
        operators = []

        for x, y in list(pdag.edges()):
            if (x, y) in expert_knowledge.required_edges:
                continue

            is_undirected = pdag.has_undirected_edge(x, y)

            neighbors_y = self._neighbors_of_in_pdag(pdag, y)
            if is_undirected:
                neighbors_y = neighbors_y - {x}

            neighbors_y_adj_x = {n for n in neighbors_y if pdag.is_adjacent(n, x)}
            neighbors_y_not_adj_x = neighbors_y - neighbors_y_adj_x

            for h_size in range(len(neighbors_y_adj_x) + 1):
                for H in combinations(neighbors_y_adj_x, h_size):
                    H = set(H)
                    clique_set = H | neighbors_y_not_adj_x
                    if self._is_clique(pdag, clique_set):
                        operators.append((x, y, H))

        return operators

    def _apply_insert(
        self,
        pdag: PDAG,
        x: Hashable,
        y: Hashable,
        T: set[Hashable],
    ) -> PDAG:
        """Apply Insert(x, y, T) operator to the PDAG.

        Adds edge x -> y and orients edges from T to y.
        Then applies Meek's rules to complete the CPDAG.
        """
        new_pdag = pdag.copy()

        new_pdag.add_edge(x, y)
        new_pdag.directed_edges.add((x, y))

        for t in T:
            if new_pdag.has_undirected_edge(t, y):
                new_pdag.orient_undirected_edge(t, y, inplace=True)

        new_pdag = new_pdag.apply_meeks_rules(apply_r4=True, inplace=False)
        return new_pdag

    def _apply_delete(
        self,
        pdag: PDAG,
        x: Hashable,
        y: Hashable,
        H: set[Hashable],
    ) -> PDAG:
        """Apply Delete(x, y, H) operator to the PDAG.

        Removes edge x - y (or x -> y) and orients edges from H to y.
        Then applies Meek's rules to complete the CPDAG.
        """
        new_pdag = pdag.copy()

        if pdag.has_undirected_edge(x, y):
            new_pdag.undirected_edges.discard((x, y))
            new_pdag.undirected_edges.discard((y, x))
            new_pdag.remove_edge(x, y)
            new_pdag.remove_edge(y, x)
        else:
            new_pdag.directed_edges.discard((x, y))
            new_pdag.remove_edge(x, y)

        for h in H:
            if new_pdag.has_undirected_edge(h, y):
                new_pdag.orient_undirected_edge(h, y, inplace=True)

        new_pdag = new_pdag.apply_meeks_rules(apply_r4=True, inplace=False)
        return new_pdag

    def _compute_insert_delta(
        self,
        pdag: PDAG,
        x: Hashable,
        y: Hashable,
        T: set[Hashable],
        score_fn,
    ) -> float:
        """Compute the score delta for Insert(x, y, T).

        For decomposable scores, the score change only depends on the
        local score at y. The new parents of y will be the old parents
        plus x and T.
        """
        neighbors_y_adj_x = {
            n for n in self._neighbors_of_in_pdag(pdag, y) if pdag.is_adjacent(n, x)
        }
        old_parents = pdag.directed_parents(y) | neighbors_y_adj_x
        new_parents = old_parents | {x} | T

        old_score = score_fn(y, list(old_parents))
        new_score = score_fn(y, list(new_parents))

        return new_score - old_score

    def _compute_delete_delta(
        self,
        pdag: PDAG,
        x: Hashable,
        y: Hashable,
        H: set[Hashable],
        score_fn,
    ) -> float:
        """Compute the score delta for Delete(x, y, H).

        For decomposable scores, the score change only depends on the
        local score at y.
        """
        neighbors_y = self._neighbors_of_in_pdag(pdag, y)
        if pdag.has_undirected_edge(x, y):
            neighbors_y = neighbors_y - {x}

        neighbors_y_not_adj_x = {n for n in neighbors_y if not pdag.is_adjacent(n, x)}

        old_parents = pdag.directed_parents(y) | neighbors_y_not_adj_x | H
        if pdag.has_directed_edge(x, y):
            old_parents = old_parents | {x}

        new_parents = old_parents - {x}

        old_score = score_fn(y, list(old_parents))
        new_score = score_fn(y, list(new_parents))

        return new_score - old_score

    def estimate(
        self,
        scoring_method: str | StructureScore | None = None,
        expert_knowledge: ExpertKnowledge | None = None,
        min_improvement: float = 1e-6,
        debug: bool = False,
    ) -> PDAG:
        """Estimates the equivalence class (CPDAG) from the data using GES.

        Parameters
        ----------
        scoring_method: str or StructureScore instance
            The score to be optimized during structure estimation. Supported
            structure scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g,
            ll-cg, aic-cg, bic-cg.

        expert_knowledge: pgmpy.estimators.ExpertKnowledge instance (default: None)
            Expert knowledge to be used with the algorithm.

        min_improvement: float
            Minimum score improvement required to apply an operator.

        debug: bool
            If True, prints debug information during search.

        Returns
        -------
        Estimated model: pgmpy.base.PDAG
            A CPDAG representing the learned equivalence class.

        Examples
        --------
        >>> import numpy as np
        >>> from pgmpy.utils import get_example_model
        >>> np.random.seed(42)
        >>> model = get_example_model("alarm")
        >>> model.seed = 42
        >>> df = model.simulate(int(1e3))

        >>> from pgmpy.estimators import GES
        >>> est = GES(df)
        >>> cpdag = est.estimate(scoring_method="bic-d")
        >>> len(cpdag.nodes())
        37

        """
        score_c: ScoreCache
        _, score_c = get_scoring_method(scoring_method, self.data, self.use_cache)
        score_fn = score_c.local_score

        current_pdag = PDAG()
        current_pdag.add_nodes_from(list(self.data.columns))

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(self.data.columns)

        while True:
            operators = self._valid_insert_operators(current_pdag, expert_knowledge)

            if not operators:
                break

            best_delta = -np.inf
            best_op = None

            for x, y, T in operators:
                delta = self._compute_insert_delta(current_pdag, x, y, T, score_fn)
                if delta > best_delta:
                    best_delta = delta
                    best_op = (x, y, T)

            if best_delta < min_improvement:
                break

            x, y, T = best_op
            current_pdag = self._apply_insert(current_pdag, x, y, T)

            if debug:
                logger.info(
                    f"Insert({x}, {y}, {T}). Score improvement: {best_delta:.4f}"
                )

        while True:
            operators = self._valid_delete_operators(current_pdag, expert_knowledge)

            if not operators:
                break

            best_delta = -np.inf
            best_op = None

            for x, y, H in operators:
                delta = self._compute_delete_delta(current_pdag, x, y, H, score_fn)
                if delta > best_delta:
                    best_delta = delta
                    best_op = (x, y, H)

            if best_delta < min_improvement:
                break

            x, y, H = best_op
            current_pdag = self._apply_delete(current_pdag, x, y, H)

            if debug:
                logger.info(
                    f"Delete({x}, {y}, {H}). Score improvement: {best_delta:.4f}"
                )

        return current_pdag
