from typing import (
    List,
    Optional,
    Union,
)

import networkx as nx
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseConstraintCausalDiscovery
from pgmpy.estimators.ScoreCache import ScoreCache
from pgmpy.estimators.StructureScore import BICGauss, StructureScore, get_scoring_method


class TOPIC(_BaseConstraintCausalDiscovery):
    """
    The TOPIC algorithm for causal discovery / structure learning.

    This class implements the TOPIC algorithm [1] for causal discovery. Given a
    tabular dataset, TOPIC estimates the causal structure among the
    variables in the data in a Directed Acyclic Graph (DAG). The algorithm works by
    establishing a topological ordering among the variables using a local scoring
    criterion, and in the process adds respectively prunes directed edges among the variables
    that are consistent with the topological ordering.

    *Note: will inherit from _BaseCausalDiscovery or _BaseScoreCausalDiscovery once those are sklearn-compatible.*

    Parameters
    ----------
    variant: str, default="orig"
        The variant of TOPIC to run.

        - "orig": The original TOPIC algorithm. Might not give the same results in different runs.

    scoring_method: str or StructureScore instance
        The score to be optimized during structure estimation.  Supported
        structure scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g,
        ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should
        be an instance of `StructureScore`.

    return_type : str, default="dag"
        The type of structure to return. Can be one of: `dag`, `pdag`.
        TOPIC by default orients all edges and returns a fully directed structure.

        - If `return_type=dag`, a fully directed structure is returned.
        - If `return_type=pdag`: the (fully) directed structure is converted to a PDAG instance.

    significance_level : float, default=0.05
        The p-value threshold to use for the no-hypercompression test (Gruenwald, 2005) for edge addition and removal.
        If the structure_score specified in `scoring_method` is an MDL (Minimal Description Length) score,
        this  `significance_level` will be used to check whether score differences are large enough
        to be considered significant for edge addition and removal.

    min_improvement : float, default=1e-6
        The minimal score improvement used for edge addition and removal
        If the structure_score specified in `scoring_method` is not an MDL score but BIC, AIC, etc.,
        this  `min_improvement` will be used to check whether score differences are large enough
        to be considered sufficient for edge  addition and  removal.

    show_progress : bool, default=False
        If True, shows a progress bar while learning the causal structure.

    use_cache : bool, default=True
        If True, uses caching of the score given in `scoring_method`.

    Attributes
    ----------
    causal_graph_ : :class:`~pgmpy.base.DAG` or :class: `~pgmpy.base.PDAG`
        The learned causal graph.

        - If `return_type="dag"`, this will be a DAG instance.
        - If `return_type="pdag"`, this will be a PDAG instance.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph, i.e. `causal_graph_`.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> import logging
    >>> logging.getLogger("pgmpy").setLevel(logging.ERROR)

    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model("ecoli70")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the TOPIC algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery.TOPIC import TOPIC
    >>> topic = TOPIC()
    >>> _ = topic.fit(df)
    >>> edges = sorted(topic.causal_graph_.edges())
    >>> len(edges) > 0
    True


    References
    ----------
    .. [1] Xu, S., Mameche, S. and Vreeken, J. Information-Theoretic Causal Discovery in Topological Order.
           International Conference on Artificial Intelligence and Statistics (AISTATS), 2025.

    """

    score_ = BICGauss
    score_fn_ = None

    def __init__(
        self,
        variant: str = "orig",
        scoring_method: Optional[Union[str, StructureScore]] = None,
        return_type: str = "dag",
        significance_level: float = 0.05,
        min_improvement: float = 1e-6,
        show_progress: bool = False,
        use_cache: bool = True,
    ):
        self.variant = variant
        self.return_type = return_type
        self.scoring_method = scoring_method
        self.significance_level = significance_level
        self.min_improvement = min_improvement
        self.show_progress = show_progress
        self.use_cache = use_cache

    def _fit(self, X: pd.DataFrame, independencies=None):
        """
        The fitting procedure for the TOPIC algorithm.

        Parameters
        ----------
        X: pd.DataFrame
            The input dataset

        independencies: Independencies
            Not used *(Note: will inherit from _BaseCausalDiscovery or _BaseScoreCausalDiscovery once
            sklearn-compatible and then this argument can be removed)*
        """

        # 0. Initialization (Data, DAG)
        self._init_score(X)
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)

        dag_current = DAG()
        dag_current.add_nodes_from(list(X.columns))
        candidates_ = list(dag_current.nodes)
        topological_order_ = []
        topic_history_ = []

        # 1. Discover a topological order, prune and add edges
        n_nodes = len(dag_current.nodes)
        pbar = (
            tqdm(total=n_nodes, desc="Topological order", unit="node")
            if self.show_progress and config.SHOW_PROGRESS
            else None
        )

        it = 0
        while it < n_nodes:
            source, source_hist = self._next_node_in_topological_order(
                candidates_, dag_current
            )
            candidates_.remove(source)
            topological_order_.append(source)

            if pbar is not None:
                pbar.set_description(f"Processing: {source}")
                pbar.set_postfix_str(f"remaining={len(candidates_)}")

            edges_added_hist, considered_adding_hist = self._add_outgoing_edges(
                source, candidates_, dag_current
            )
            edges_pruned_hist, considered_pruning_hist = self._remove_ingoing_edges(
                source, dag_current
            )

            # History
            topic_history_.append(
                {
                    "iteration": it,
                    "source": source,
                    "topological_order": [n for n in topological_order_],
                    "remaining_candidates": [c for c in candidates_],
                    "source_selection": source_hist,
                    "edges_added": edges_added_hist,
                    "edges_pruned": edges_pruned_hist,
                    "considered_adding": considered_adding_hist,
                    "considered_pruning": considered_pruning_hist,
                }
            )

            if pbar is not None:
                pbar.update(1)
            it += 1

        if pbar is not None:
            pbar.set_description("Topological order")
            pbar.set_postfix_str("")
            pbar.close()

        if self.return_type == "dag":
            self.causal_graph_ = dag_current
        elif self.return_type in ("pdag", "cpdag"):
            self.causal_graph_ = dag_current.to_pdag()
        else:
            raise ValueError(
                f"return_type must be one of: dag, pdag, or cpdag. Got: {self.return_type}"
            )
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_)
        self.topological_order_ = topological_order_
        self.history_ = topic_history_

        return self

    def _init_score(self, X: pd.DataFrame):
        """
        Initializes the local scoring function and score cache for the given dataset.

        Parameters
        ----------
        X: pd.DataFrame
            The input dataset
        """
        score_c: ScoreCache
        score, score_c = get_scoring_method(self.scoring_method, X, self.use_cache)
        score_fn = score_c.local_score
        self.score_ = score
        self.score_fn_ = score_fn

    def _next_node_in_topological_order(
        self, candidates: List[int | str], dag_current: DAG
    ) -> tuple[int | str, dict]:
        """
        Returns the next node in topological order.

        Parameters
        ----------
        candidates: List
            Remaining nodes that are candidates to be the next node in topological order
        dag_current: DAG
            The causal graph constructed so far; by construction, all edges are outgoing from nodes not in candidates

        Returns
        -------
        next_node: int
            The next node in topological order

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.causal_discovery.TOPIC import TOPIC
        >>> from pgmpy.base import DAG
        >>> data = pd.DataFrame(
        ...     np.random.randint(0, 4, size=(5000, 3)), columns=list("ABD")
        ... )
        >>> data["C"] = data["A"] - data["B"]
        >>> data["D"] += data["A"]
        >>> model = TOPIC()
        >>> model._init_score(data)
        >>> dag = DAG()
        >>> dag.add_nodes_from(list(data.columns))
        >>> next_node = model._next_node_in_topological_order(list(data.columns), dag)
        >>> next_node[0] in list(data.columns)
        True
        """

        if self.score_fn_ is None:
            raise ValueError(
                "Score function not initialized. Call _init_score(data) or fit(data) first."
            )

        improvement = self._improvement_matrix(candidates, dag_current)
        delta = improvement - improvement.T
        np.fill_diagonal(delta, -np.inf)

        incoming_pressure = np.max(delta, axis=0)
        source_idx = int(np.argmin(incoming_pressure))
        source = candidates[source_idx]
        best_delta_per_node = np.max(delta, axis=1)

        order_idx = np.argsort(incoming_pressure)
        ranking = [
            {
                "node": candidates[i],
                "incoming_pressure": float(incoming_pressure[i]),
                "best_delta": float(best_delta_per_node[i]),
            }
            for i in order_idx
        ]

        meta = {
            "candidates": [c for c in candidates],
            "improvement_matrix": improvement.tolist(),
            "delta_matrix": delta.tolist(),
            "best_delta": [float(x) for x in best_delta_per_node],
            "ranking": ranking,
            "source_idx": int(source_idx),
        }
        return source, meta

    # %% Helpers for search
    def _improvement_matrix(
        self,
        candidates: List,
        dag_current: DAG,
    ) -> np.ndarray:
        """Pair-wise improvement matrix: score improvements for each pair-wise edge under the current model

        Parameters
        ----------
        candidates:
            pair-wise edge candidates
        dag_current:
            current DAG

        Returns
        -------
        improvement_matrix: np.ndarray
             score improvement for each pair-wise edge
        """
        improvement_matrix = np.zeros((len(candidates), len(candidates)))
        idx = {node: i for i, node in enumerate(candidates)}
        for cause in candidates:
            for effect in candidates:
                if cause == effect:
                    continue
                score_improv = self._addition_gain(cause, effect, dag_current)
                improvement_matrix[idx[cause], idx[effect]] = score_improv
        return improvement_matrix

    def _add_outgoing_edges(
        self, source: int | str, candidates: List[int | str], dag_current: DAG
    ) -> tuple[list[dict], list[dict]]:
        """Adds outgoing edges from source node

        Parameters
        ----------
        source:
            source node
        candidates:
            candidate nodes; by construction not upstream from source in the topological order

        Returns
        -------
        added_edges: List
             added edges
        considered_edges: List
             all considered edges
        """
        added_edges = []
        considered_edges = []

        for node in candidates:
            if node == source:
                continue

            gain = self._addition_gain(source, node, dag_current)
            significant = self._score_significant(gain)

            considered_edges.append(
                {
                    "from": str(source),
                    "to": str(node),
                    "gain": gain,
                    "significant": significant,
                }
            )

            if significant:
                dag_current.add_edge(source, node)
                added_edges.append({"from": str(source), "to": str(node), "gain": gain})

        return added_edges, considered_edges

    def _remove_ingoing_edges(
        self,
        source: int | str,
        dag_current: DAG,
    ) -> [List, List]:
        """Removes ingoing edges from source node

        Parameters
        ----------
        source:
            source node
        dag_current:
            current model

        Returns
        -------
        added_edges: List
             removed edges
        considered_edges: List
             all considered edges
        """
        pruned_edges = []
        considered_edges = []
        current_parents = list(dag_current.get_parents(source)).copy()
        while len(current_parents) > 0:
            removed_found, removed_parent, best_diff, candidate_diffs = (
                self._find_removable_edge(current_parents, source)
            )

            for parent, diff in candidate_diffs:
                considered_edges.append(
                    {
                        "from": str(parent),
                        "to": str(source),
                        "diff": diff,
                    }
                )

            if removed_parent is None:
                break
            dag_current.remove_edge(removed_parent, source)
            current_parents.remove(removed_parent)

            pruned_edges.append(
                {
                    "from": str(removed_parent),
                    "to": str(source),
                    "diff": best_diff,
                }
            )
        return pruned_edges, considered_edges

    def _find_removable_edge(
        self, parents: List[int | str], child: int | str, noise_epsilon: float = 1e-10
    ):
        """Helper function for finding removable edges from a parent set to

        Parameters
        ----------
        parents:
            parent nodes
        child:
            child node
        noise_epsilon:
            noise threshold
        """
        old_score = self._score(child, parents)

        best_parent = None
        best_harm = float("-inf")
        candidate_stats: List[tuple[int | str, float]] = []

        for parent in parents:
            new_parents = [p for p in parents if p != parent]
            if len(new_parents) == 0:
                continue

            new_score = self._score(child, new_parents)
            if new_score is None:
                continue

            harm = float(new_score - old_score)
            candidate_stats.append((parent, harm))

            if harm >= -noise_epsilon and harm > best_harm:
                best_harm = harm
                best_parent = parent

        removed_found = best_parent is not None

        if not removed_found:
            return (
                (False, None, float("inf"), candidate_stats)
                if len(candidate_stats) == 0
                else (False, None, 0.0, candidate_stats)
            )

        return True, best_parent, best_harm, candidate_stats

    # %% Helpers for scoring
    def _score(self, effect, parents) -> float:
        """Wrapper for local scoring

        Parameters
        ----------
        effect:
            A node in the DAG
        parents:
            A set of parent nodes in the DAG
        Returns
        -------
        score: int
            local score of ``parents`` -> ``effect``
        """
        if self.score_fn_ is None:
            raise ValueError("Score function not initialized")

        score = self.score_fn_(effect, parents)
        return score

    def _score_significant(self, score_improvement):
        """Checks whether a score difference is large enough to be considered an improvement

        *Note:  once MDL scores MDLScoreXY are implemented, check and use 2 ^(-score) < self.significance_level.*

        Parameters
        ----------
        score_improvement:
            updated_score - previous_score, for two models (updated_DAG, previous_DAG)
        Returns
        -------
        significant: bool
             whether score difference is large enough to prefer updated_DAG over previous_DAG
        """
        significant = score_improvement > self.min_improvement
        return significant

    def _addition_gain(self, cause, effect, dag_current):
        """Score gain of including an additional edge in the current model

        Parameters
        ----------
        cause:
            cause of edge
        effect:
            effect of edge
        dag_current:
            current DAG

        Returns
        -------
        score_improv: float
             score improvement
        """
        current_parents = list(dag_current.get_parents(effect)).copy()
        old_score = self._score(effect, current_parents)
        current_parents.append(cause)
        new_score = self._score(effect, current_parents)
        score_improv = new_score - old_score
        return score_improv
