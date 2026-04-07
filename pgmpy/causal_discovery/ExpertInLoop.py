from __future__ import annotations

from collections.abc import Callable
from functools import partial
from itertools import combinations

import networkx as nx
import numpy as np
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

    orientation_fn : callable, default=None
        A function to determine edge orientation. The function should take at
        least two arguments (the names of the two variables) and return either:
        - A tuple (source, target) representing the directed edge from source
          to target
        - None, representing no edge between the variables

        Built-in functions that can be used:
        - `pgmpy.utils.manual_pairwise_orient`: Prompts the user to specify direction.
        - `pgmpy.utils.llm_pairwise_orient`: Uses an LLM to determine direction.

    orientations : set, default=None
        A set of edges that will be used as the preferred orientation over
        the output of `orientation_fn`. Edges should be specified as tuples
        (source, target).

    expert_knowledge : ExpertKnowledge, default=None
        Expert knowledge about the causal structure. Can include:
        - forbidden_edges: Edges that should not be present in the final model
        - required_edges: Edges that must be present in the final model
        - temporal_order: The temporal ordering of variables
        - orientation_fn: Function to determine edge orientations

        Note: Explicit orientations in the `orientations` parameter take
        precedence over temporal ordering.

    use_cache : bool, default=True
        If True, the algorithm caches results from `orientation_fn` and reuses
        them in future calls instead of querying the orientation function again.

    show_progress : bool, default=True
        If True, prints information about the running status.

    max_iter : int, default=1000
        Maximum number of iterations for the main learning loop. Useful for
        controlling runtime on large datasets.

    descriptions : dict[str, str], default=None
        A dictionary mapping variable names to their natural language descriptions.
        REQUIRED ONLY when using LLM-based orientation (llm_pairwise_orient).
        Can also be provided via functools.partial() wrapping the orientation function.

        Example::

            descriptions = {
                "Age": "Person's age in years",
                "Income": "Annual household income in USD",
                "Education": "Years of formal education completed",
            }

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

    ci_cache_ : dict
        Cache of conditional independence test results.

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

    Using LLM-based orientation with descriptions in __init__ (Method 1):

    >>> from pgmpy.utils import llm_pairwise_orient
    >>> descriptions = {
    ...     "Smoker": "Whether a person smokes (True/False)",
    ...     "Cancer": "Whether a person has lung cancer (True/False)",
    ... }
    >>> eil = ExpertInLoop(
    ...     orientation_fn=llm_pairwise_orient,
    ...     descriptions=descriptions,
    ...     effect_size_threshold=0.0001,
    ... )
    >>> eil.fit(df)  # doctest: +SKIP

    Using LLM-based orientation with partial() (Method 2 - Recommended):

    >>> from functools import partial
    >>> from pgmpy.utils import llm_pairwise_orient
    >>> descriptions = {
    ...     "Smoker": "Whether a person smokes (True/False)",
    ...     "Cancer": "Whether a person has lung cancer (True/False)",
    ... }
    >>> orientation_fn = partial(
    ...     llm_pairwise_orient,
    ...     descriptions=descriptions,
    ...     llm_model="gemini/gemini-1.5-flash",
    ... )
    >>> eil = ExpertInLoop(
    ...     orientation_fn=orientation_fn,
    ...     effect_size_threshold=0.0001,
    ... )
    >>> eil.fit(df)  # doctest: +SKIP

    Using LLM-based orientation with attribute assignment (Method 3):

    >>> from pgmpy.utils import llm_pairwise_orient
    >>> descriptions = {
    ...     "Smoker": "Whether a person smokes (True/False)",
    ...     "Cancer": "Whether a person has lung cancer (True/False)",
    ... }
    >>> eil = ExpertInLoop(
    ...     orientation_fn=llm_pairwise_orient,
    ...     effect_size_threshold=0.0001,
    ... )
    >>> eil.descriptions = descriptions  # Set before calling fit()
    >>> eil.fit(df)  # doctest: +SKIP

    Combining LLM orientation with expert knowledge:

    >>> from functools import partial
    >>> from pgmpy.utils import llm_pairwise_orient
    >>> from pgmpy.estimators import ExpertKnowledge
    >>> descriptions = {
    ...     "Smoker": "Whether a person smokes",
    ...     "Cancer": "Whether a person has cancer",
    ... }
    >>> orientation_fn = partial(
    ...     llm_pairwise_orient,
    ...     descriptions=descriptions,
    ...     llm_model="gemini/gemini-1.5-flash",
    ... )
    >>> expert = ExpertKnowledge(
    ...     forbidden_edges=[("Cancer", "Smoker")],  # Cancer doesn't cause smoking
    ...     temporal_order=[["Age"], ["Smoker"], ["Cancer"]],  # Age -> Smoker -> Cancer
    ... )
    >>> eil = ExpertInLoop(
    ...     orientation_fn=orientation_fn,
    ...     expert_knowledge=expert,
    ...     effect_size_threshold=0.0001,
    ... )
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
        orientation_fn: Callable | None = None,
        orientations: set[tuple[str, str]] | None = None,
        expert_knowledge=None,
        use_cache: bool = True,
        show_progress: bool = True,
        max_iter: int = 1000,
        descriptions: dict[str, str] | None = None,
    ):
        """
        Initialize the ExpertInLoop causal discovery estimator.

        Parameters
        ----------
        pval_threshold : float, default=0.05
            The p-value threshold for conditional independence tests.
        effect_size_threshold : float, default=0.05
            The effect size threshold for adding/removing edges.
        ci_test : str or callable, default=None
            The conditional independence test to use.
        orientation_fn : callable, default=None
            Function to orient edges. If None, temporal_order from expert_knowledge is used.
        orientations : set of tuple, default=None
            Pre-specified edge orientations.
        expert_knowledge : ExpertKnowledge, default=None
            Expert knowledge about the graph structure.
        use_cache : bool, default=True
            Whether to cache orientation and CI test results.
        show_progress : bool, default=True
            Whether to print progress information.
        max_iter : int, default=1000
            Maximum number of algorithm iterations.
        descriptions : dict[str, str], default=None
            Variable descriptions required for LLM-based orientation.
            Maps variable names to natural language descriptions.

        Raises
        ------
        ValueError
            If LLM orientation is used without providing descriptions.
        """
        self.pval_threshold = pval_threshold
        self.effect_size_threshold = effect_size_threshold
        self.ci_test = ci_test
        self.orientation_fn = orientation_fn
        self.orientations = orientations
        self.expert_knowledge = expert_knowledge
        self.use_cache = use_cache
        self.show_progress = show_progress
        self.max_iter = max_iter
        self.descriptions = descriptions

    def _test_all(self, ci_test, dag, data, blacklisted=None):
        """
        Runs CI tests on all possible combinations of variables.

        If blacklisted is provided, skips recording non-edge candidates present
        in blacklist (either direction), reducing downstream filtering work.

        Parameters
        ----------
        ci_test : callable
            The CI test function to use.
        dag : DAG
            The current DAG structure.
        data : pd.DataFrame
            The data for CI testing.
        blacklisted : set, optional
            Set of edges to skip as non-edge candidates.

        Returns
        -------
        pd.DataFrame
            Results with columns: u, v, z, edge_present, effect, p_val
        """
        cis = []
        # ci_cache_ is initialized in _fit() before this method is called
        ci_cache = self.ci_cache_

        for u, v in combinations(list(dag.nodes()), 2):
            u_parents = set(dag.get_parents(u))
            v_parents = set(dag.get_parents(v))

            if v in u_parents:
                conditioning_set = u_parents - {v}
                edge_present = True
            elif u in v_parents:
                conditioning_set = v_parents - {u}
                edge_present = True
            else:
                conditioning_set = u_parents | v_parents
                edge_present = False
                if blacklisted is not None:
                    if (u, v) in blacklisted or (v, u) in blacklisted:
                        continue

            # FIXED: Keep (u, v) in order to preserve directionality
            cache_key = (u, v, frozenset(conditioning_set))

            if cache_key in ci_cache:
                effect, p_value = ci_cache[cache_key]
            else:
                effect, p_value = ci_test.run_test(X=u, Y=v, Z=list(conditioning_set))
                ci_cache[cache_key] = (effect, p_value)

            cis.append([u, v, list(conditioning_set), edge_present, effect, p_value])

        return pd.DataFrame(cis, columns=["u", "v", "z", "edge_present", "effect", "p_val"])

    def _break_cycle(self, dag, u, v, ci_test, data, effect_size_threshold, pval_threshold):
        """
        Subroutine to break any cycles that get created.

        Parameters
        ----------
        dag : DAG
            The current DAG that still doesn't have cycles.
        u, v : hashable
            The variables that create a cycle when (u, v) edge is added.
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
        edges_to_remove = []
        temp_dag = nx.DiGraph(dag)
        temp_dag.add_edges_from([(u, v)])
        for cycle in nx.simple_cycles(temp_dag):
            for x, y in zip(cycle, cycle[1:] + [cycle[0]]):
                if not ((x == u) and (y == v)):
                    Z = set(cycle) - {x, y}
                    effect, pvalue = ci_test.run_test(x, y, Z=Z)
                    if (effect < effect_size_threshold) and (pvalue > pval_threshold):
                        edges_to_remove.append((x, y))
                        if self.show_progress or config.SHOW_PROGRESS:
                            logger.info(f"Removing edge: {x} -> {y} to fix cycle")
        return edges_to_remove

    def _get_edge_orientation(self, u: str, v: str) -> tuple[str, str] | None:
        """
        Determines orientation robust to fit state.

        Priority order:
        1. Explicit orientations (from ExpertKnowledge or constructor)
        2. Orientation function result
        3. Temporal ordering as fallback
        """
        expert_knowledge = getattr(self, "expert_knowledge_", self.expert_knowledge)
        to = getattr(expert_knowledge, "temporal_ordering", {}) if expert_knowledge else {}

        # 1a. Check ExpertKnowledge orientations
        if expert_knowledge and hasattr(expert_knowledge, "orientations") and expert_knowledge.orientations:
            if (u, v) in expert_knowledge.orientations:
                # Return explicit orientation as-is (DO NOT override with temporal)
                return (u, v)
            if (v, u) in expert_knowledge.orientations:
                return (v, u)

        # 1b. Check constructor orientations
        if self.orientations:
            if (u, v) in self.orientations:
                return (u, v)
            if (v, u) in self.orientations:
                return (v, u)

        if not hasattr(self, "orientation_cache_"):
            self.orientation_cache_ = set()

        orientation_cache = self.orientation_cache_

        cache_key = (u, v, tuple(sorted(to.items())) if to else None)
        cache_key_rev = (v, u, tuple(sorted(to.items())) if to else None)

        if self.use_cache:
            if cache_key in orientation_cache:
                return orientation_cache[cache_key]
            if cache_key_rev in orientation_cache:
                return orientation_cache[cache_key_rev]

        orient_fn = self.orientation_fn or getattr(expert_knowledge, "orientation_fn", None)

        if orient_fn is not None:
            # Check for llm_pairwise_orient (handles partial objects)
            test_fn = orient_fn.func if isinstance(orient_fn, partial) else orient_fn
            is_llm = (test_fn == llm_pairwise_orient) or (getattr(test_fn, "__name__", "") == "llm_pairwise_orient")

            if is_llm:
                # Get descriptions from various sources in priority order:
                # 1. Descriptions passed as partial() argument
                # 2. Descriptions from __init__ parameter
                # 3. Descriptions attribute set on instance

                descriptions = None

                # Check if partial has descriptions already
                if isinstance(orient_fn, partial) and "descriptions" in orient_fn.keywords:
                    descriptions = orient_fn.keywords["descriptions"]

                # Fallback to __init__ parameter or instance attribute
                if not descriptions:
                    descriptions = self.descriptions or getattr(self, "_descriptions", {})

                if not descriptions:
                    raise ValueError(
                        "LLM orientation requires variable descriptions. "
                        "Provide via one of these methods:\n"
                        "  1. ExpertInLoop(descriptions={'var1': 'description1', ...})\n"
                        "  2. partial(llm_pairwise_orient, descriptions={...})\n"
                        "  3. estimator.descriptions = {...} before calling fit()"
                    )

                # Call orientation function with descriptions
                # Only pass descriptions if not already in partial keywords
                if isinstance(orient_fn, partial) and "descriptions" in orient_fn.keywords:
                    res = orient_fn(u, v)  # partial already has descriptions
                else:
                    res = orient_fn(u, v, descriptions=descriptions)
            else:
                res = orient_fn(u, v)

            # Enforce temporal ordering if available
            to = getattr(expert_knowledge, "temporal_ordering", {})
            if res and u in to and v in to and to[res[0]] > to[res[1]]:
                res = (res[1], res[0])

            if res and self.use_cache:
                orientation_cache.add(res)

            if (self.show_progress or config.SHOW_PROGRESS) and res:
                logger.info(f"Queried for edge orientation: {u} - {v} -> {res}")
            return res

        if u in to and v in to:
            if to[u] < to[v]:
                return (u, v)
            elif to[v] < to[u]:
                return (v, u)
            else:
                # Same temporal tier - no direction can be determined
                return None

        raise ValueError(
            "No orientation function is available. "
            "Provide at least one of: orientation_fn, orientations, expert_knowledge, or temporal_order."
        )

    def _fit(self, X: pd.DataFrame):
        """
        Fit the ExpertInLoop causal discovery algorithm.

        Parameters
        ----------
        X : pd.DataFrame
            The data to learn the causal structure from.

        Returns
        -------
        self : ExpertInLoop
            Returns the instance with the fitted attributes.
        """
        from pgmpy.estimators import ExpertKnowledge

        self.variables_ = list(X.columns)
        self.n_iter_ = 0

        # Initialize caches FIRST - these are required by _test_all() and
        # _get_edge_orientation() which are called in the main loop below
        self.ci_cache_ = {}
        self.orientation_cache_ = set()

        # Handle expert knowledge setup in fit to remain scikit-learn compliant
        if self.expert_knowledge is None:
            self.expert_knowledge_ = ExpertKnowledge(
                orientation_fn=self.orientation_fn,
                orientations=self.orientations,
            )
        else:
            self.expert_knowledge_ = self.expert_knowledge
            # If explicit parameters passed to ExpertInLoop, they override expert_knowledge defaults
            if self.orientation_fn is not None:
                self.expert_knowledge_.orientation_fn = self.orientation_fn
            if self.orientations is not None:
                if isinstance(self.expert_knowledge_.orientations, list):
                    original_orientations = set(self.expert_knowledge_.orientations)
                    original_orientations.update(self.orientations)
                    self.expert_knowledge_.orientations = list(original_orientations)
                else:
                    self.expert_knowledge_.orientations.update(self.orientations)

        dag = DAG()
        dag.add_nodes_from(self.variables_)

        # Robust categorical detection
        cat_cols = X.select_dtypes(include=["category", "object"]).columns
        test_param = self.ci_test or ("chi_square" if len(cat_cols) > 0 else None)
        ci_test = get_ci_test(test=test_param, data=X)

        blacklisted_edges = list(self.expert_knowledge_.forbidden_edges) if self.expert_knowledge_ else []
        if self.expert_knowledge_ and self.expert_knowledge_.required_edges:
            dag.add_edges_from(self.expert_knowledge_.required_edges)

        while self.n_iter_ < self.max_iter:
            self.n_iter_ += 1
            # Build blacklist set including reverse direction once per iteration
            bl_set_iter = set(blacklisted_edges) | {(v, u) for u, v in blacklisted_edges}
            all_effects = self._test_all(dag=dag, ci_test=ci_test, data=X, blacklisted=bl_set_iter)
            if all_effects.empty:
                break

            edge_effects = all_effects[all_effects.edge_present]
            edge_effects = edge_effects[
                (edge_effects.effect < self.effect_size_threshold) & (edge_effects.p_val > self.pval_threshold)
            ]
            remove_edges = [tuple(x) for x in edge_effects[["u", "v"]].values]

            if self.expert_knowledge_ and self.expert_knowledge_.required_edges:
                req_set = set(self.expert_knowledge_.required_edges)
                req_set.update([(v, u) for u, v in self.expert_knowledge_.required_edges])
                remove_edges = [edge for edge in remove_edges if edge not in req_set]

            for edge in remove_edges:
                dag.remove_edge(edge[0], edge[1])

            nonedge_effects = all_effects[~all_effects.edge_present]
            nonedge_effects = nonedge_effects[
                (nonedge_effects.effect >= self.effect_size_threshold) & (nonedge_effects.p_val <= self.pval_threshold)
            ]

            if len(blacklisted_edges) > 0 and not nonedge_effects.empty:
                bl_set_iter = set(blacklisted_edges) | {(v, u) for u, v in blacklisted_edges}

                nonedge_effects = nonedge_effects[
                    ~nonedge_effects[["u", "v"]].apply(lambda row: tuple(row) in bl_set_iter, axis=1)
                ]

            if nonedge_effects.empty:
                if edge_effects.empty:
                    break
                else:
                    continue

            selected_edge = nonedge_effects.iloc[nonedge_effects.effect.argmax()]
            edge_direction = self._get_edge_orientation(selected_edge.u, selected_edge.v)

            if edge_direction is None:
                if self.show_progress or config.SHOW_PROGRESS:
                    logger.info(
                        f"Orientation function returned None for edge {selected_edge.u}-{selected_edge.v}. Skipping."
                    )
                blacklisted_edges.append((selected_edge.u, selected_edge.v))
            elif nx.has_path(dag, edge_direction[1], edge_direction[0]):
                if self.show_progress or config.SHOW_PROGRESS:
                    logger.info("Returned edge orientation creates a cycle. Trying to identify the incorrect edge.")
                edges_to_remove = self._break_cycle(
                    dag,
                    edge_direction[0],
                    edge_direction[1],
                    ci_test,
                    X,
                    self.effect_size_threshold,
                    self.pval_threshold,
                )
                if not edges_to_remove:
                    blacklisted_edges.append(edge_direction)
                elif [tuple(e) == tuple(edge_direction) for e in edges_to_remove].count(True) > 0:
                    if self.show_progress or config.SHOW_PROGRESS:
                        logger.info(
                            f"Cycle-breaking subroutine suggested removing the new edge {edge_direction}. Rejecting it."
                        )
                    blacklisted_edges.append(edge_direction)
                else:
                    blacklisted_edges.extend(edges_to_remove)
                    dag.remove_edges_from(edges_to_remove)
                    dag.add_edges_from([edge_direction])
            else:
                dag.add_edges_from([edge_direction])

        if self.n_iter_ >= self.max_iter and self.show_progress:
            logger.warning(
                f"ExpertInLoop stopped after reaching max_iter={self.max_iter}. "
                f"Graph may be incomplete. Increase max_iter if needed."
            )

        # In _fit method, around line 501-506
        self.causal_graph_ = dag
        self.adjacency_matrix_ = pd.DataFrame(
            nx.adjacency_matrix(dag, nodelist=self.variables_, weight=None).toarray().astype(int),
            index=self.variables_,
            columns=self.variables_,
        )
        self.n_features_in_ = len(self.variables_)
        self.feature_names_in_ = np.array(self.variables_)
        return self
