from collections.abc import Hashable
from itertools import combinations

import networkx as nx
import numpy as np
import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery, _ScoreMixin
from pgmpy.estimators import ExpertKnowledge
from pgmpy.estimators.StructureScore import StructureScore, get_scoring_method


class GES(_ScoreMixin, _BaseCausalDiscovery):
    """
    Score-based causal discovery using Greedy Equivalence Search (GES).

    This class implements the GES algorithm [1]_ for causal discovery. Given a
    tabular dataset, the algorithm estimates the causal structure among the
    variables in the data as a Directed Acyclic Graph (DAG) or Partially
    Directed Acyclic Graph (PDAG).

    GES works in three phases:
        1. Forward phase: Edges are added to improve the model score.
        2. Backward phase: Edges are removed to improve the model score.
        3. Edge flipping phase: Edge orientations are flipped to improve the score.

    Parameters
    ----------
    scoring_method : str or StructureScore instance, default=None
        The score to be optimized during structure estimation. Supported
        structure scores:

        - Discrete data: 'k2', 'bdeu', 'bds', 'bic-d', 'aic-d'
        - Continuous data: 'll-g', 'aic-g', 'bic-g'
        - Mixed data: 'll-cg', 'aic-cg', 'bic-cg'

        If None, the appropriate scoring method is automatically selected based
        on the data type. Also accepts a custom score instance that inherits
        from `StructureScore`.

    expert_knowledge : ExpertKnowledge instance, default=None
        Expert knowledge to be used with the algorithm. Expert knowledge
        allows specification of:

        - Required edges that must be present in the final graph
        - Forbidden edges that cannot be present in the final graph
        - Temporal ordering of nodes

    return_type : str, default='pdag'
        The type of graph to return. Options are:

        - 'dag': Returns a directed acyclic graph (DAG).
        - 'pdag': Returns a partially directed acyclic graph (PDAG).

    min_improvement : float, default=1e-6
        The minimum score improvement required to perform an operation
        (edge addition, removal, or flipping). Operations with smaller
        improvements are not performed.

    use_cache : bool, default=True
        If True, uses caching of local scores for faster computation.
        Note: Caching only works for scoring methods which are decomposable.
        Can give incorrect results for custom non-decomposable scoring methods.

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

    Use the GES algorithm to learn the causal structure from data:

    >>> from pgmpy.causal_discovery import GES
    >>> ges = GES(scoring_method="bic-d")
    >>> ges.fit(df)
    GES(scoring_method='bic-d')
    >>> ges.causal_graph_  # doctest: +ELLIPSIS
    <pgmpy.base.PDAG.PDAG object at 0x...>
    >>> ges.n_features_in_
    37

    Use expert knowledge to constrain the search:

    >>> from pgmpy.estimators import ExpertKnowledge
    >>> expert = ExpertKnowledge(forbidden_edges=[("HISTORY", "CVP")])
    >>> ges = GES(scoring_method="bic-d", expert_knowledge=expert)
    >>> ges.fit(df)  # doctest: +ELLIPSIS
    GES(expert_knowledge=<pgmpy.estimators.ExpertKnowledge.ExpertKnowledge object at 0x...>,
        scoring_method='bic-d')

    References
    ----------
    .. [1] Chickering, David Maxwell. "Optimal structure identification with
           greedy search." Journal of machine learning research 3.Nov (2002):
           507-554.
    """

    def __init__(
        self,
        scoring_method: str | StructureScore | None = None,
        expert_knowledge: ExpertKnowledge | None = None,
        return_type: str = "pdag",
        min_improvement: float = 1e-6,
        use_cache: bool = True,
    ):
        self.scoring_method = scoring_method
        self.expert_knowledge = expert_knowledge
        self.return_type = return_type
        self.min_improvement = min_improvement
        self.use_cache = use_cache

    def _legal_edge_additions(
        self, current_model: DAG, expert_knowledge: ExpertKnowledge
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Returns a list of all edges that can be added to the graph such that
        it remains a DAG.
        """
        edges = []
        for u, v in combinations(current_model.nodes(), 2):
            if not (current_model.has_edge(u, v) or current_model.has_edge(v, u)):
                if not nx.has_path(current_model, v, u) and ((u, v) not in expert_knowledge.forbidden_edges):
                    edges.append((u, v))
                if not nx.has_path(current_model, u, v) and ((v, u) not in expert_knowledge.forbidden_edges):
                    edges.append((v, u))
        return edges

    def _legal_edge_removals(
        self, current_model: DAG, expert_knowledge: ExpertKnowledge
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Returns a list of all edges that can be removed from the graph.
        """
        edges = []
        for u, v in current_model.edges():
            if (u, v) not in expert_knowledge.required_edges:
                edges.append((u, v))
        return edges

    def _legal_edge_flips(
        self, current_model: DAG, expert_knowledge: ExpertKnowledge
    ) -> list[tuple[Hashable, Hashable]]:
        """
        Returns a list of all edges that can be flipped such that the model
        remains a DAG.
        """
        potential_flips = []
        edges = list(current_model.edges())
        for u, v in edges:
            if ((u, v) not in expert_knowledge.required_edges) and ((v, u) not in expert_knowledge.forbidden_edges):
                current_model.remove_edge(u, v)
                if not nx.has_path(current_model, u, v):
                    potential_flips.append((v, u))
                current_model.add_edge(u, v)
        return potential_flips

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the GES algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : pgmpy.causal_discovery.GES
            Returns the instance with the fitted attributes.
        """
        self.variables_ = list(X.columns)

        _, score_c = get_scoring_method(self.scoring_method, X, self.use_cache)
        score_fn = score_c.local_score

        current_model = DAG()
        current_model.add_nodes_from(self.variables_)

        if self.expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()
        else:
            expert_knowledge = self.expert_knowledge

        if expert_knowledge.search_space:
            expert_knowledge.limit_search_space(X.columns)

        expert_knowledge._orient_temporal_forbidden_edges(current_model, only_edges=False)

        while True:
            potential_edges = self._legal_edge_additions(current_model, expert_knowledge)
            score_deltas = np.zeros(len(potential_edges))

            for index, (u, v) in enumerate(potential_edges):
                current_parents = current_model.get_parents(v)
                score_deltas[index] = score_fn(v, current_parents + [u]) - score_fn(v, current_parents)

            if len(potential_edges) == 0 or np.all(score_deltas < self.min_improvement):
                break

            edge_to_add = potential_edges[np.argmax(score_deltas)]
            current_model.add_edge(edge_to_add[0], edge_to_add[1])

        while True:
            potential_removals = self._legal_edge_removals(current_model, expert_knowledge)
            score_deltas = np.zeros(len(potential_removals))

            for index, (u, v) in enumerate(potential_removals):
                current_parents = current_model.get_parents(v)
                score_deltas[index] = score_fn(v, [node for node in current_parents if node != u]) - score_fn(
                    v, current_parents
                )

            if len(potential_removals) == 0 or np.all(score_deltas < self.min_improvement):
                break

            edge_to_remove = potential_removals[np.argmax(score_deltas)]
            current_model.remove_edge(edge_to_remove[0], edge_to_remove[1])

        while True:
            potential_flips = self._legal_edge_flips(current_model, expert_knowledge)
            score_deltas = np.zeros(len(potential_flips))

            for index, (u, v) in enumerate(potential_flips):
                v_parents = current_model.get_parents(v)
                u_parents = current_model.get_parents(u)
                score_deltas[index] = (score_fn(v, v_parents + [u]) - score_fn(v, v_parents)) + (
                    score_fn(u, [node for node in u_parents if node != v]) - score_fn(u, u_parents)
                )

            if len(potential_flips) == 0 or np.all(score_deltas < self.min_improvement):
                break

            edge_to_flip = potential_flips[np.argmax(score_deltas)]
            current_model.remove_edge(edge_to_flip[1], edge_to_flip[0])
            current_model.add_edge(edge_to_flip[0], edge_to_flip[1])

        if self.return_type.lower() == "dag":
            self.causal_graph_ = current_model
        elif self.return_type.lower() == "pdag":
            self.causal_graph_ = current_model.to_pdag()
        else:
            raise ValueError(f"return_type must be one of: dag, pdag. Got: {self.return_type}")

        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype="int")

        return self
