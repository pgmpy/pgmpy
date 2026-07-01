from __future__ import annotations

from itertools import combinations, pairwise

import networkx as nx
import pandas as pd
from sklearn.base import clone

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import BaseCausalDiscovery
from pgmpy.ci_tests import get_ci_test
from pgmpy.global_vars import logger


class ExpertInLoop(BaseCausalDiscovery):
    """
    Expert-in-the-loop causal discovery algorithm.

    This class implements an iterative causal discovery algorithm that combines statistical independence testing with
    expert knowledge for edge orientation. The algorithm works by iteratively adding and removing edges between
    variables based on conditional independence tests, similar to the Greedy Equivalence Search (GES) algorithm. When
    adding edges, the algorithm orients each pair using a pairwise causal discovery estimator (e.g. `LLMPairwise`).

    The algorithm can use various sources for edge orientation:
    - A pairwise causal discovery estimator.
    - Specified `expert_knowledge` argument.

    Parameters
    ----------
    pval_threshold : float, default=0.05
        The p-value threshold used in conditional independence tests. If the p-value is greater than this threshold, the
        variables are considered conditionally independent.

    effect_size_threshold : float, default=0.05
        The effect size threshold for edge suggestions.
        - If the conditional effect size between two variables is greater
          than this threshold, the algorithm suggests adding an edge.
        - If the effect size for an existing edge is less than this threshold,
          the algorithm suggests removing the edge.

    ci_test : str or callable, default=None
        The Conditional Independence test to use. When None, the algorithm
        tries to automatically detect a suitable CI test based on the variable
        types. See :mod:`pgmpy.estimators.CITests` for available tests.

    pairwise_estimator : pgmpy.causal_discovery estimator, default=None
        A pairwise causal discovery estimator (e.g. `LLMPairwise`) fit on each
        candidate pair, with the edge read from its `causal_graph_`. Required;
        used for any pair whose direction is not already constrained by
        `expert_knowledge`.

    expert_knowledge : ExpertKnowledge, default=None
        Expert knowledge about the causal structure. Can include:
        - forbidden_edges: Edges that should not be present in the final model
        - required_edges: Edges that must be present in the final model
        - temporal_order: The temporal ordering of variables

        The expert knowledge is resolved with ``ExpertKnowledge.fit(data)``,
        whose fitted ``forbidden_edges_`` fold the temporal order (later-tier to
        earlier-tier edges are forbidden) and any search space into a single set
        of forbidden directed edges that constrain orientation. To force a
        specific orientation ``u -> v``, forbid its reverse with
        ``ExpertKnowledge(forbidden_edges={(v, u)})``.

    show_progress : bool, default=True
        If True, prints information about the running status.

    Attributes
    ----------
    causal_graph_ : DAG
        The learned causal graph as a DAG.

    adjacency_matrix_ : pd.DataFrame
        Adjacency matrix representation of the learned causal graph.

    n_features_in_ : int
        The number of features in the data used to learn the causal graph.

    feature_names_in_ : np.ndarray
        The feature names in the data used to learn the causal graph.

    Examples
    --------
    Basic usage with a custom pairwise estimator:

    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.base import DAG
    >>> from pgmpy.causal_discovery import ExpertInLoop
    >>> from pgmpy.causal_discovery._base import BaseCausalDiscovery
    >>> model = get_example_model("cancer")
    >>> df = model.simulate(int(1e3))
    >>> class AlphabeticalPairwise(BaseCausalDiscovery):
    ...     def _fit(self, X):
    ...         u, v = sorted(self.feature_names_in_)
    ...         self.causal_graph_ = DAG([(u, v)])
    ...         return self
    ...
    >>> eil = ExpertInLoop(
    ...     pairwise_estimator=AlphabeticalPairwise(), effect_size_threshold=0.0001
    ... )
    >>> eil.fit(df)  # doctest: +ELLIPSIS
    ExpertInLoop(effect_size_threshold=0.0001,
                 pairwise_estimator=AlphabeticalPairwise())
    >>> _ = eil.causal_graph_.edges()

    Using expert knowledge with temporal ordering:

    `expert_knowledge` constrains the orientation for the pairs it covers;
    `pairwise_estimator` orients everything else and is always required. To
    force a specific orientation ``u -> v``, forbid its reverse with
    ``ExpertKnowledge(forbidden_edges={(v, u)})``.

    >>> from pgmpy.causal_discovery import ExpertKnowledge
    >>> expert = ExpertKnowledge(
    ...     temporal_order=[["Pollution", "Smoker"], ["Cancer"], ["Xray", "Dyspnoea"]]
    ... )
    >>> eil = ExpertInLoop(
    ...     pairwise_estimator=AlphabeticalPairwise(),
    ...     expert_knowledge=expert,
    ...     effect_size_threshold=0.0001,
    ... )
    >>> _ = eil.fit(df)

    Using LLM-based orientation through a pairwise estimator (requires API key):

    >>> from pgmpy.causal_discovery import LLMPairwise
    >>> variable_descriptions = {
    ...     "Smoker": "Whether a person smokes",
    ...     "Cancer": "Whether a person has cancer",
    ... }
    >>> pairwise_estimator = LLMPairwise(
    ...     descriptions=variable_descriptions,
    ...     llm_model="gemini/gemini-2.5-flash",
    ... )
    >>> eil = ExpertInLoop(pairwise_estimator=pairwise_estimator)
    >>> eil.fit(df)  # doctest: +SKIP

    References
    ----------
    The algorithm is inspired by active learning approaches to causal discovery
    and the GES algorithm.
    """

    def __init__(
        self,
        pval_threshold: float = 0.05,
        effect_size_threshold: float = 0.05,
        ci_test: str | None = None,
        pairwise_estimator=None,
        expert_knowledge=None,
        show_progress: bool = True,
    ):
        self.pval_threshold = pval_threshold
        self.effect_size_threshold = effect_size_threshold
        self.ci_test = ci_test
        self.pairwise_estimator = pairwise_estimator
        self.expert_knowledge = expert_knowledge
        self.show_progress = show_progress

    def _test_all(self, ci_test, dag: DAG, data: pd.DataFrame) -> pd.DataFrame:
        """
        Runs CI tests on all possible combinations of variables in `dag`.

        Parameters
        ----------
        ci_test : callable
            The CI test function to use.

        dag : pgmpy.base.DAG
            The DAG on which to run the tests.

        data : pd.DataFrame
            The data to use for CI testing.

        Returns
        -------
        pd.DataFrame
            The results with p-values and effect sizes of all the tests.
        """
        cis = []
        for u, v in combinations(list(dag.nodes()), 2):
            u_parents = set(dag.get_parents(u))
            v_parents = set(dag.get_parents(v))

            if v in u_parents:
                u_parents -= {v}
                edge_present = True
            elif u in v_parents:
                v_parents -= {u}
                edge_present = True
            else:
                edge_present = False

            cond_set = list(set(u_parents).union(v_parents))
            effect, p_value = ci_test.run_test(X=u, Y=v, Z=cond_set)

            cis.append([u, v, cond_set, edge_present, effect, p_value])

        return pd.DataFrame(cis, columns=["u", "v", "z", "edge_present", "effect", "p_val"])

    def _break_cycle(self, dag, u, v, ci_test, data, effect_size_threshold, pval_threshold):
        """
        Subroutine to break any cycles that get created.

        Parameters
        ----------
        dag : pgmpy.base.DAG
            The current DAG that still doesn't have cycles.

        u, v : hashable
            The variables that create a cycle in `dag` when (u, v) edge is added.

        ci_test : callable
            The Conditional Independence test to use.

        data : pd.DataFrame
            The data for CI testing.

        effect_size_threshold : float
            Threshold for effect size.

        pval_threshold : float
            Threshold for p-value.

        Returns
        -------
        list
            List of edges to remove to break the cycle.
        """
        logger.info("Returned edge orientation creates a cycle. Trying to identify the incorrect edge.")
        edges_to_remove = []
        temp_dag = dag.copy()
        temp_dag.add_edges_from([(u, v)])
        for cycle in nx.simple_cycles(temp_dag):
            for x, y in pairwise(cycle):
                if not ((x == u) and (y == v)):
                    Z = set(cycle) - {x, y}
                    effect, pvalue = ci_test.run_test(x, y, Z=Z)
                    if (effect < effect_size_threshold) and (pvalue > pval_threshold):
                        edges_to_remove.append((x, y))
                        logger.info(f"Removing edge: {x} -> {y} to fix cycle")

        return edges_to_remove

    def _fit(self, X: pd.DataFrame):
        """
        The fitting procedure for the ExpertInLoop algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : ExpertInLoop
            Returns the instance with the fitted attributes.
        """
        # Step 0: Validate arguments and assign defaults before any computation.
        if self.pairwise_estimator is None:
            raise ValueError(
                "`pairwise_estimator` must be provided to orient edges, "
                "e.g. `ExpertInLoop(pairwise_estimator=LLMPairwise(...))`."
            )

        self.variables_ = list(X.columns)

        # Resolve expert knowledge into forbidden/required edges via a cloned, fit ExpertKnowledge (mirrors `PC`).
        forbidden_edges = set()
        required_edges = []
        if self.expert_knowledge is not None:
            expert_knowledge = clone(self.expert_knowledge)
            expert_knowledge.fit(X)
            forbidden_edges = expert_knowledge.forbidden_edges_
            required_edges = list(expert_knowledge.required_edges_)

        # Blacklist of excluded unordered pairs: seeded with pairs forbidden both ways; also grows during the run.
        blacklist = {frozenset((u, v)) for (u, v) in forbidden_edges if (v, u) in forbidden_edges}

        # Initialize the working DAG (seeded with required edges) and the CI test.
        dag = DAG()
        dag.add_nodes_from(self.variables_)
        dag.add_edges_from(required_edges)
        ci_test = get_ci_test(test=self.ci_test, data=X)

        while True:
            # Step 1: Compute effects and p-values between every combination of variables
            all_effects = self._test_all(dag=dag, ci_test=ci_test, data=X)

            # Edge case: if only 1 feature, no combinations exist
            if all_effects.empty:
                break

            # Step 2: Remove any edges between variables that are not sufficiently associated
            edge_effects = all_effects[all_effects.edge_present]
            edge_effects = edge_effects[
                (edge_effects.effect < self.effect_size_threshold) & (edge_effects.p_val > self.pval_threshold)
            ]
            remove_edges = list(edge_effects.loc[:, ("u", "v")].to_records(index=False))
            for edge in remove_edges:
                dag.remove_edge(edge[0], edge[1])

            # Step 3: Add edge between variables which have significant association
            # Step 3.1: Find edges that are not present in the DAG but have significant association
            nonedge_effects = all_effects[all_effects.edge_present == False]
            nonedge_effects = nonedge_effects[
                (nonedge_effects.effect >= self.effect_size_threshold) & (nonedge_effects.p_val <= self.pval_threshold)
            ]

            # Step 3.2: Remove any blacklisted (fully excluded) pair, matching the unordered pair exactly.
            if blacklist:
                keep = [frozenset((u, v)) not in blacklist for u, v in zip(nonedge_effects.u, nonedge_effects.v)]
                nonedge_effects = nonedge_effects[keep]

            # Step 3.3: Exit loop if all correlations in data are explained by the model
            if (edge_effects.shape[0] == 0) and (nonedge_effects.shape[0] == 0):
                break

            # If there are only removals and no candidate additions, continue
            # to the next iteration after having applied removals.
            if nonedge_effects.shape[0] == 0:
                continue

            # Step 3.4: Find the pair of variables with the highest effect size
            selected_edge = nonedge_effects.iloc[nonedge_effects.effect.argmax()]
            edge_direction = None

            # Step 3.5: Orient via `forbidden_edges`; fall back to the pairwise estimator when neither dir is forbidden.
            u, v = selected_edge.u, selected_edge.v

            if (u, v) in forbidden_edges and (v, u) in forbidden_edges:
                # Both directions forbidden; normally already excluded via `blacklist`.
                edge_direction = None
            elif (u, v) in forbidden_edges:
                edge_direction = (v, u)
            elif (v, u) in forbidden_edges:
                edge_direction = (u, v)
            else:
                self.pairwise_estimator.fit(X[[u, v]])
                edges = list(self.pairwise_estimator.causal_graph_.edges())
                edge_direction = edges[0] if edges else None

                if config.SHOW_PROGRESS and self.show_progress and edge_direction is not None:
                    logger.info(
                        "\rQueried for edge orientation between "
                        f"{u} and {v}. Got: {edge_direction[0]} -> {edge_direction[1]}"
                    )

            # Step 3.6: Handle the edge direction
            # 1. If the orientation source returns None, do not add the edge
            # 2. If new edge creates a cycle, try to resolve it
            # 3. Otherwise, add the edge
            if edge_direction is None:
                logger.info(f"Orientation returned None for edge {u} - {v}. Skipping this edge.")
                blacklist.add(frozenset((u, v)))
            elif nx.has_path(dag, edge_direction[1], edge_direction[0]):
                edges_to_remove = self._break_cycle(
                    dag,
                    edge_direction[0],
                    edge_direction[1],
                    ci_test=ci_test,
                    data=X,
                    effect_size_threshold=self.effect_size_threshold,
                    pval_threshold=self.pval_threshold,
                )
                blacklist.update(frozenset(e) for e in edges_to_remove)
                dag.remove_edges_from(edges_to_remove)
                dag.add_edges_from([(edge_direction[0], edge_direction[1])])
            else:
                dag.add_edges_from([edge_direction])

        # Set the fitted attributes
        self.causal_graph_ = dag
        self.adjacency_matrix_ = pd.DataFrame(
            nx.adjacency_matrix(dag, nodelist=self.variables_, weight=None).todense(),
            index=self.variables_,
            columns=self.variables_,
        )

        return self
