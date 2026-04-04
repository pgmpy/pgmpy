from __future__ import annotations

from collections.abc import Callable
from itertools import combinations

import networkx as nx
import pandas as pd

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.ci_tests import get_ci_test
from pgmpy.global_vars import logger
from pgmpy.utils import llm_pairwise_orient


class ExpertInLoop(_BaseCausalDiscovery):
    """
    Expert-in-the-loop causal discovery algorithm.

    This class implements an iterative causal discovery algorithm that combines statistical independence testing with
    expert knowledge for edge orientation. The algorithm works by iteratively adding and removing edges between
    variables based on conditional independence tests, similar to the Greedy Equivalence Search (GES) algorithm. When
    adding edges, the algorithm queries an expert (human or automated through LLMs) for the edge orientation.

    The algorithm can use various sources for edge orientation:
    - Manual user input
    - Large Language Models (LLMs)
    - Custom orientation functions
    - Pre-specified orientations
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

    orientation_fn : callable, default=llm_pairwise_orient
        A function to determine edge orientation. The function should take at
        least two arguments (the names of the two variables) and return either:
        - A tuple (source, target) representing the directed edge from source
          to target
        - None, representing no edge between the variables

        Built-in functions that can be used:
        - `pgmpy.utils.manual_pairwise_orient`: Prompts the user to specify direction.
        - `pgmpy.utils.llm_pairwise_orient`: Uses an LLM to determine direction.

    orientations : set, default=set()
        A set of edges that will be used as the preferred orientation over
        the output of `orientation_fn`. Edges should be specified as tuples
        (source, target).

    expert_knowledge : ExpertKnowledge, default=None
        Expert knowledge about the causal structure. Can include:
        - forbidden_edges: Edges that should not be present in the final model
        - required_edges: Edges that must be present in the final model
        - temporal_order: The temporal ordering of variables

        Note: Explicit orientations in the `orientations` parameter take
        precedence over temporal ordering.

    use_cache : bool, default=True
        If True, the algorithm caches results from `orientation_fn` and reuses
        them in future calls instead of querying the orientation function again.

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

    orientation_cache_ : set
        Cache of edge orientations learned during fitting.

    Examples
    --------
    Basic usage with custom orientation function:

    >>> from pgmpy.utils import get_example_model
    >>> from pgmpy.causal_discovery import ExpertInLoop
    >>> model = get_example_model("cancer")
    >>> df = model.simulate(int(1e3))
    >>> def custom_orient(var1, var2, **kwargs):
    ...     return (var1, var2) if var1 < var2 else (var2, var1)
    ...
    >>> eil = ExpertInLoop(orientation_fn=custom_orient, effect_size_threshold=0.0001)
    >>> eil.fit(df)
    >>> eil.causal_graph_.edges()

    Using pre-specified orientations:

    >>> orientations = {("Pollution", "Cancer"), ("Smoker", "Cancer")}
    >>> eil = ExpertInLoop(orientations=orientations, effect_size_threshold=0.0001)
    >>> eil.fit(df)

    Using expert knowledge with temporal ordering:

    >>> from pgmpy.estimators import ExpertKnowledge
    >>> expert = ExpertKnowledge(
    ...     temporal_order=[["Pollution", "Smoker"], ["Cancer"], ["Xray", "Dyspnoea"]]
    ... )
    >>> eil = ExpertInLoop(expert_knowledge=expert, effect_size_threshold=0.0001)
    >>> eil.fit(df)

    Using LLM-based orientation (requires API key):

    >>> from functools import partial
    >>> from pgmpy.utils import llm_pairwise_orient
    >>> variable_descriptions = {
    ...     "Smoker": "Whether a person smokes",
    ...     "Cancer": "Whether a person has cancer",
    ... }
    >>> orientation_fn = partial(
    ...     llm_pairwise_orient,
    ...     variable_descriptions=variable_descriptions,
    ...     llm_model="gemini/gemini-1.5-flash",
    ... )
    >>> eil = ExpertInLoop(orientation_fn=orientation_fn)
    >>> eil.fit(df)

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
        orientation_fn: Callable = llm_pairwise_orient,
        orientations: set[tuple[str, str]] | None = None,
        expert_knowledge=None,
        use_cache: bool = True,
        show_progress: bool = True,
    ):
        self.pval_threshold = pval_threshold
        self.effect_size_threshold = effect_size_threshold
        self.ci_test = ci_test
        self.orientation_fn = orientation_fn
        self.orientations = orientations
        self.expert_knowledge = expert_knowledge
        self.use_cache = use_cache
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
            for x, y in zip(cycle, cycle[1:]):
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
        self.variables_ = list(X.columns)

        # Initialize orientation cache (preserve if pre-populated)
        if not hasattr(self, "orientation_cache_"):
            self.orientation_cache_ = set()

        # Step 0: Create a new DAG on all the variables with no edge.
        dag = DAG()
        dag.add_nodes_from(self.variables_)

        # Get the CI test
        ci_test = get_ci_test(test=self.ci_test, data=X)

        # Initialize blacklisted_edges with forbidden_edges from expert knowledge
        blacklisted_edges = []
        if self.expert_knowledge is not None:
            blacklisted_edges = list(self.expert_knowledge.forbidden_edges)
            # Add required edges to the DAG
            if self.expert_knowledge.required_edges:
                dag.add_edges_from(self.expert_knowledge.required_edges)

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

            # Step 3.2: Remove any pair of variables that are blacklisted
            if len(blacklisted_edges) > 0:
                blacklisted_edges_us = [edge[0] for edge in blacklisted_edges]
                blacklisted_edges_vs = [edge[1] for edge in blacklisted_edges]
                nonedge_effects = nonedge_effects.loc[
                    ~(
                        (nonedge_effects.u.isin(blacklisted_edges_us) & nonedge_effects.v.isin(blacklisted_edges_vs))
                        | (nonedge_effects.u.isin(blacklisted_edges_vs) & nonedge_effects.v.isin(blacklisted_edges_us))
                    ),
                    :,
                ]

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

            # Step 3.5: Find the edge orientation for the selected pair of variables
            #
            # Priority order:
            # 1. If `orientations` are provided, use them
            # 2. Check temporal ordering from expert_knowledge
            # 3. Try to use cached orientations if `use_cache=True`
            # 4. If no cached orientation, call the orientation_fn

            # Get orientations set (handle None case)
            orientations_set = self.orientations if self.orientations is not None else set()

            if (selected_edge.u, selected_edge.v) in orientations_set:
                edge_direction = (selected_edge.u, selected_edge.v)
            elif (selected_edge.v, selected_edge.u) in orientations_set:
                edge_direction = (selected_edge.v, selected_edge.u)
            elif self.expert_knowledge is not None and self.expert_knowledge.temporal_ordering:
                # Check if temporal order can determine the direction
                u_order = self.expert_knowledge.temporal_ordering.get(selected_edge.u)
                v_order = self.expert_knowledge.temporal_ordering.get(selected_edge.v)
                if u_order is not None and v_order is not None:
                    if u_order < v_order:
                        edge_direction = (selected_edge.u, selected_edge.v)
                    elif v_order < u_order:
                        edge_direction = (selected_edge.v, selected_edge.u)
            elif self.use_cache and (selected_edge.u, selected_edge.v) in self.orientation_cache_:
                edge_direction = (selected_edge.u, selected_edge.v)
            elif self.use_cache and (selected_edge.v, selected_edge.u) in self.orientation_cache_:
                edge_direction = (selected_edge.v, selected_edge.u)
            else:
                edge_direction = self.orientation_fn(selected_edge.u, selected_edge.v)
                if self.use_cache is True and edge_direction is not None:
                    self.orientation_cache_.add(edge_direction)

                if config.SHOW_PROGRESS and self.show_progress and edge_direction is not None:
                    logger.info(
                        "\rQueried for edge orientation between "
                        f"{selected_edge.u} and {selected_edge.v}. Got: "
                        f"{edge_direction[0]} -> {edge_direction[1]}"
                    )

            # Step 3.6: Handle the edge direction
            # 1. If orientation function returns None, do not add the edge
            # 2. If new edge creates a cycle, try to resolve it
            # 3. Otherwise, add the edge
            if edge_direction is None:
                logger.info(
                    f"Orientation function returned None for edge {selected_edge.u} - {selected_edge.v}. "
                    "Skipping this edge."
                )
                blacklisted_edges.append((selected_edge.u, selected_edge.v))
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
                blacklisted_edges.extend(edges_to_remove)
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
