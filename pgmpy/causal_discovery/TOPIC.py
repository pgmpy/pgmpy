from typing import (
    List,
    Optional,
    Union,
)

import networkx as nx
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
        # score_fn = score_c.local_score

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
            source, source_meta = self._next_node_in_topological_order(X, candidates_)
            candidates_.remove(source)
            topological_order_.append(source)

            added_edges, outgoing_scores = self._add_outgoing_edges(X, source)
            pruned_edges, incoming_scores = self._remove_ingoing_edges(X, source)

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

    def _score_edge(
        self,
    ):
        raise NotImplementedError

    def _next_node_in_topological_order(
        self,
        X: pd.DataFrame,
        candidates: List,
        **kwargs,
    ) -> int:
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
        >>> c = TOPIC()
        >>> next = c._next_node_in_topological_order(list(range(data.columns)))
        >>> print(next)
        """

        source = candidates[0]
        meta_source = []
        return source, meta_source

    def _add_outgoing_edges(
        self,
        X: pd.DataFrame,
        source: int,
        **kwargs,
    ) -> [List, List]:
        added_edges = []
        meta_added_edges = []
        return added_edges, meta_added_edges

    def _remove_ingoing_edges(
        self,
        X: pd.DataFrame,
        source: int,
        **kwargs,
    ) -> [List, List]:

        pruned_edges = []
        meta_pruned_edges = []
        return pruned_edges, meta_pruned_edges
