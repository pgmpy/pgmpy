from typing import (
    List,
    Optional,
    Union, Callable,
)

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery import _BaseConstraintCausalDiscovery
from pgmpy.estimators import ExpertKnowledge, StructureScore
from pgmpy.estimators.ScoreCache import ScoreCache
from pgmpy.estimators.StructureScore import get_scoring_method


class TOPIC(_BaseConstraintCausalDiscovery):
    """
    The TOPIC algorithm for causal discovery / structure learning.

    This class implements the TOPIC algorithm [1] for causal discovery. Given a
    tabular dataset, TOPIC estimates the causal structure among the
    variables in the data in a Directed Acyclic Graph (DAG). The algorithm works by
    establishing a topological ordering among the variables using a local scoringidentifying
    criterion, adding resp. pruning directed edges consistent with the topological order
    in the process.


    Parameters
    ----------
    variant: str, default="orig"
        The variant of TOPIC to run.

        - "orig": The original TOPIC algorithm. Might not give the same results in different runs.
        - "parallel": Parallel version of TOPIC. Can run on multiple cores.

    scoring_method: str or StructureScore instance
        The score to be optimized during structure estimation.  Supported
        structure scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g,
        ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should
        be an instance of `StructureScore`.

    return_type : str, default="dag"
        The type of structure to return. Can be one of: `pdag`, `cpdag`, `dag`.

        - If `return_type=pdag` or `return_type=cpdag`: a partially directed structure is returned.
        - If `return_type=dag`, a fully directed structure is returned. This DAG is one of the possible orientations of
          the PDAG learned by the PC algorithm.

    significance_level : float, default=0.01
        The p-value threshold to use for the statistical significance tests of scores. If the p-value of a test is
        greater than
        `significance_level`, then the variables are considered independent.

    expert_knowledge : :class:`pgmpy.estimators.ExpertKnowledge`, optional
        Expert knowledge to be used in the causal graph construction. This needs to be an instance of
        :class:`pgmpy.estimators.ExpertKnowledge`. Users can specify knowledge in the form of required/forbidden edges,
        temporal information, or restrict the search space.

    enforce_expert_knowledge : bool, default=False
        If True, the expert knowledge will be strictly enforced. This implies the following:

        - For every edge (u, v) specified in `forbidden_edges`, there will be no edge between u and v.
        - For every edge (u, v) specified in `required_edges`, one of the following would be present in the final model:
          u -> v, u <- v, or u - v (if CPDAG is returned).

        If False, the algorithm attempts to make the edge orientations as specified by expert knowledge after learning
        the skeleton. This implies the following:

        - For every edge (u, v) specified in `forbidden_edges`, the final graph would have either v <- u or no edge
          except if u -> v is part of a collider structure in the learned skeleton.
        - For every edge (u, v) specified in `required_edges`, the final graph would either have u -> v or no edge
          except if v <- u is part of a collider structure in the learned skeleton.

    n_jobs : int, default=-1
        The number of jobs to run in parallel. This is only used when `variant="parallel"`.

    show_progress : bool, default=True
        If True, shows a progress bar while learning the causal structure.

    Attributes
    ----------
    causal_graph_ : :class:`~pgmpy.base.DAG` or :class: `~pgmpy.base.PDAG`
        The learned causal graph.

        - If `return_type="pdag"`, this will be a PDAG instance.
        - If `return_type="dag"`, this will be a DAG instance.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph, i.e. `causal_graph_`.

    skeleton_ : :class:`~pgmpy.base.UndirectedGraph`
        An estimate for the undirected graph skeleton of the DAG underlying the data.

    separating_sets_ : dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes them
            conditionally independent. (needed for edge orientation procedures)

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Simulate some data to use for causal discovery:

    >>> from pgmpy.utils import get_example_model
    >>> model = get_example_model("alarm")
    >>> df = model.simulate(n_samples=1000, seed=42)

    Use the TOPIC algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import TOPIC
    >>> topic = TOPIC()
    >>> topic.fit(df)
    >>> topic.causal_graph_.edges()


    References
    ----------
    .. [1] Xu, S., Mameche, S. and Vreeken, J. Information-Theoretic Causal Discovery in Topological Order.
           International Conference on Artificial Intelligence and Statistics (AISTATS), 2025.
    """

    def __init__(
        self,
        variant: str = "parallel",
        return_type: str = "dag",
        scoring_method: Optional[Union[str, StructureScore]] = None,
        significance_level: float = 0.01,
        expert_knowledge: Optional[ExpertKnowledge] = None,
        enforce_expert_knowledge: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True,
        use_cache: bool = True,
    ):
        self.variant = variant
        self.return_type = return_type
        self.scoring_method = scoring_method
        self.significance_level = significance_level
        self.expert_knowledge = expert_knowledge
        self.enforce_expert_knowledge = enforce_expert_knowledge
        self.n_jobs = n_jobs
        self.show_progress = show_progress
        self.use_cache = use_cache

    def _fit(self, X: pd.DataFrame, independencies=None):
        """
        The fitting procedure for the TOPIC algorithm.
        """

        # Initialization
        if self.expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()
        else:
            expert_knowledge = self.expert_knowledge

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(X.columns)

        # Step 0: Initial checks and setup for arguments
        score_c: ScoreCache
        _, score_c = get_scoring_method(self.scoring_method, X, self.use_cache)
        score_fn = score_c.local_score

        # Step 1: Initialize an empty model.
        dag_current = DAG()
        dag_current.add_nodes_from(list(X.columns))
        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(X.columns)

        expert_knowledge._orient_temporal_forbidden_edges(dag_current, only_edges=False)

        candidates_ = list(dag_current.nodes)
        topological_order_ = []
        topic_history_ = []

        # Step 2: Establish topological order
        n_nodes = len(dag_current.nodes)
        it = 0
        while it < n_nodes:
            source, source_meta = self._next_node_in_topological_order(candidates_, dag_current, score_fn)
            candidates_.remove(source)
            topological_order_.append(source)

            added_edges, outgoing_scores = self._add_outgoing_edges(source, dag_current, score_fn)
            pruned_edges, incoming_scores = self._remove_ingoing_edges(source, dag_current, score_fn)

            # History
            topic_history_.append(
                {
                    "iteration": it,
                    "source": source,
                    "topological_order": [n for n in topological_order_],
                    "remaining_candidates": [c for c in candidates_],
                    "source_selection": source_meta,
                    "added_edges": added_edges,
                    "pruned_edges": pruned_edges,
                    "outgoing_scores": outgoing_scores,
                    "incoming_scores": incoming_scores,
                }
            )
            it += 1

        if self.return_type in ("pdag", "cpdag"):
            self.causal_graph_ = dag_current.to_pdag()
        elif self.return_type == "dag":
            self.causal_graph_ = dag_current
        else:
            raise ValueError(
                f"return_type must be one of: dag, pdag, or cpdag. Got: {self.return_type}"
            )

        self.adjacency_matrix_ = nx.to_pandas_adjacency(
            self.causal_graph_, weight=1, dtype="int"
        )

        return self


    def _next_node_in_topological_order(
        self,
        candidates: List,
        dag_current: DAG,
        score_fn: Callable,
        **kwargs,
    ) -> [int, List]:
        """
        Returns the next node in topological order.

        Parameters
        ----------
        candidates: List
            remaining nodes that are candidates to be the next node
        Returns
        -------
        next_node: int
            the next node in topological order

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.causal_discovery.TOPIC import TOPIC
        >>> data = pd.DataFrame(
        ...     np.random.randint(0, 4, size=(5000, 3)), columns=list("ABD")
        ... )
        >>> data["C"] = data["A"] - data["B"]
        >>> data["D"] += data["A"]
        >>> model = TOPIC()
        >>> score_c: ScoreCache
        >>> _, score_c = get_scoring_method("bic-g", data, True)
        >>> score_fn = score_c.local_score

        >>> next = model._next_node_in_topological_order(list(range(data.columns)), DAG(), score_fn)
        >>> print(next)
        """
        improvement = self._improvement_matrix(candidates, dag_current, score_fn, **kwargs)
        delta = improvement - improvement.T
        np.fill_diagonal(delta, -np.inf)

        incoming_pressure = np.max(delta, axis=0)
        source_idx = int(np.argmin(incoming_pressure))
        source = candidates[source_idx]
        best_delta_per_node = np.max(delta, axis=1)

        order_idx = np.argsort(incoming_pressure)
        ranking = [
            { "node":  candidates[i],
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


    def _improvement_matrix(
        self,
        candidates: List,
        dag_current: DAG,
        score_fn: Callable,
        **kwargs,
    ) -> np.ndarray:
        improvement_matrix = np.zeros((len(candidates), len(candidates)))
        for cause in candidates:
            for effect in candidates:
                if cause == effect: continue
                current_parents = list(dag_current.get_parents(effect)).copy()
                old_score = score_fn(effect, current_parents)
                current_parents.append(cause)
                new_score = score_fn(effect, current_parents)
                improvement_matrix[candidates.index(cause), candidates.index(effect)] = \
                    old_score - new_score #self._gain(new_score, old_score)
        return improvement_matrix

    def _add_outgoing_edges(
        self,
        source: int,
        dag_current: DAG,
        score_fn: Callable,
        **kwargs,
    ) -> [List, List]:
        added_edges = []
        meta_added_edges = []
        return added_edges, meta_added_edges

    def _remove_ingoing_edges(
        self,
        source: int,
        dag_current: DAG,
        score_fn: Callable,
        **kwargs,
    ) -> [List, List]:

        pruned_edges = []
        meta_pruned_edges = []
        return pruned_edges, meta_pruned_edges
