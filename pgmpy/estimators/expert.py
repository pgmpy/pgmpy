from itertools import combinations

import pandas as pd

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.CITests import pillai_trace
from pgmpy.global_vars import logger
from pgmpy.utils import llm_pairwise_orient


class ExpertInLoop(StructureEstimator):
    """
    A structure estimator that combines expert knowledge with iterative learning.

    The ExpertInLoop class learns a DAG structure by iteratively adding and removing
    edges based on conditional independence tests, while respecting constraints from
    expert knowledge and consulting experts/LLMs for edge orientations.

    Expert knowledge constraints are handled in the following order:
    1. Forbidden edges (blacklist) are never added to the graph
    2. Required edges are added at initialization and protected from removal
    3. Temporal order is used to resolve edge directions before consulting experts/LLMs

    Parameters
    ----------
    data: pandas.DataFrame
        DataFrame containing the data. Each column represents one variable/node.

    expert_knowledge: ExpertKnowledge, optional (default: None)
        Expert knowledge containing:
        - forbidden_edges: edges that should never appear in the graph
        - required_edges: edges that must be present (but may be pruned if data strongly suggests)
        - temporal_order: temporal/causal ordering of variables to help determine edge directions

    **kwargs: dict
        Additional arguments to be passed to StructureEstimator.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.estimators import ExpertInLoop, ExpertKnowledge
    >>> from pgmpy.utils import get_example_model
    >>>
    >>> # Get example data
    >>> model = get_example_model("cancer")
    >>> data = model.simulate(int(1e3))
    >>>
    >>> # Define expert knowledge
    >>> expert = ExpertKnowledge(
    ...     forbidden_edges=[("Xray", "Smoker")],
    ...     required_edges=[("Smoker", "Cancer")],
    ...     temporal_order=[["Pollution", "Smoker"], ["Cancer"], ["Dyspnoea", "Xray"]]
    ... )
    >>>
    >>> # Learn structure with expert knowledge
    >>> estimator = ExpertInLoop(data, expert_knowledge=expert)
    >>> dag = estimator.estimate(
    ...     effect_size_threshold=0.01,
    ...     show_progress=True
    ... )
    >>> dag.edges()
    OutEdgeView([('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea'), ('Pollution', 'Cancer')])

    Notes
    -----
    The algorithm follows these steps:
    1. Initialize DAG with required edges from expert knowledge
    2. Iteratively:
        a. Remove edges with insufficient statistical support (except required edges)
        b. Add edges with strong statistical support (respecting forbidden edges)
        c. Determine edge directions using:
            - Required edge directions from expert knowledge
            - Temporal ordering from expert knowledge
            - Expert/LLM consultation via orientation_fn
    """

    def __init__(self, data=None, expert_knowledge=None, **kwargs):
        """
        Initialize ExpertInLoop with data and optional expert knowledge.

        Parameters
        ----------
        data: pandas.DataFrame
            DataFrame containing the data. Each column represents one variable/node.

        expert_knowledge: ExpertKnowledge, optional (default: None)
            Expert knowledge containing forbidden_edges, required_edges, and temporal_order
            information. These constraints will be respected during structure learning.

        **kwargs: dict
            Additional arguments to be passed to StructureEstimator.
        """
        super(ExpertInLoop, self).__init__(data=data, **kwargs)
        self.orientation_cache = set([])
        self.expert_knowledge = expert_knowledge

    def test_all(self, dag):
        """
        Runs CI tests on all possible combinations of variables in `dag`.

        Parameters
        ----------
        dag: pgmpy.base.DAG
            The DAG on which to run the tests.

        Returns
        -------
        pd.DataFrame: The results with p-values and effect sizes of all the tests.
        """
        cis = []
        for u, v in combinations(list(dag.nodes()), 2):
            # Skip if edge is forbidden by expert knowledge
            if self.expert_knowledge is not None and (
                (u, v) in self.expert_knowledge.forbidden_edges
                or (v, u) in self.expert_knowledge.forbidden_edges
            ):
                continue

            u_parents = set(dag.get_parents(u))
            v_parents = set(dag.get_parents(v))

            if v in u_parents:
                u_parents -= set([v])
                edge_present = True
            elif u in v_parents:
                v_parents -= set([u])
                edge_present = True
            else:
                edge_present = False

            cond_set = list(set(u_parents).union(v_parents))
            effect, p_value = pillai_trace(
                X=u, Y=v, Z=cond_set, data=self.data, boolean=False
            )
            cis.append([u, v, cond_set, edge_present, effect, p_value])

        return pd.DataFrame(
            cis, columns=["u", "v", "z", "edge_present", "effect", "p_val"]
        )

    def _resolve_edge_direction(self, var1, var2, orientation_fn, **kwargs):
        """
        Resolve edge direction using temporal order if available, otherwise use orientation_fn.

        Parameters
        ----------
        var1, var2: str
            The variable names between which to determine edge direction.
        orientation_fn: callable
            Function to determine edge orientation if temporal order doesn't resolve it.
        **kwargs:
            Additional arguments to pass to orientation_fn.

        Returns
        -------
        tuple or None: (source, target) representing edge direction, or None if no edge.
        """
        if self.expert_knowledge is not None:
            # Check if direction is determined by required edges
            if (var1, var2) in self.expert_knowledge.required_edges:
                return (var1, var2)
            if (var2, var1) in self.expert_knowledge.required_edges:
                return (var2, var1)

            # Check temporal order
            if (
                var1 in self.expert_knowledge.temporal_ordering
                and var2 in self.expert_knowledge.temporal_ordering
            ):
                order1 = self.expert_knowledge.temporal_ordering[var1]
                order2 = self.expert_knowledge.temporal_ordering[var2]
                if order1 < order2:
                    return (var1, var2)
                elif order2 < order1:
                    return (var2, var1)

        # If no expert knowledge resolves direction, use orientation_fn
        return orientation_fn(var1, var2, **kwargs)

    def estimate(
        self,
        pval_threshold=0.05,
        effect_size_threshold=0.05,
        orientation_fn=llm_pairwise_orient,
        orientations=set([]),
        use_cache=True,
        show_progress=True,
        **kwargs,
    ):
        """
        Estimates a DAG from the data by utilizing expert knowledge and optional expert input.

        The method iteratively adds and removes edges between variables based on:
        1. Expert knowledge constraints (forbidden_edges, required_edges, temporal_order)
        2. Conditional independence tests
        3. Expert/LLM input for edge orientation when not determined by expert knowledge

        Parameters
        ----------
        pval_threshold: float
            The p-value threshold to use for the test to determine whether
            there is a significant association between the variables or not.

        effect_size_threshold: float
            The effect size threshold to use to suggest a new edge.

        orientation_fn: callable (default: pgmpy.utils.llm_pairwise_orient)
            A function to determine edge orientation when not determined by expert knowledge.

        orientations: set
            Users can specify a set of edges which would be used as the
            preferred orientation for edges over the output of orientation_fn.

        use_cache: bool
            If True, the method will cache the results returned by
            `orientation_fn` and reuse it in future calls.

        show_progress: bool (default: True)
            If True, prints info of the running status.

        kwargs: kwargs
            Any additional parameters to pass to the `orientation_fn`.

        Returns
        -------
        pgmpy.base.DAG: A DAG representing the learned causal structure.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model, llm_pairwise_orient, manual_pairwise_orient
        >>> from pgmpy.estimators import ExpertInLoop
        >>> model = get_example_model('cancer')
        >>> df = model.simulate(int(1e3))

        >>> # Using manual orientation
        >>> dag = ExpertInLoop(df).estimate(
        ...     effect_size_threshold=0.0001,
        ...     orientation_fn=manual_pairwise_orient
        ... )

        >>> # Using LLM-based orientation
        >>> variable_descriptions = {
        ...     "Smoker": "A binary variable representing whether a person smokes or not.",
        ...     "Cancer": "A binary variable representing whether a person has cancer.",
        ...     "Xray": "A binary variable representing the result of an X-ray test.",
        ...     "Pollution": "A binary variable representing whether the person is in a high-pollution area or not.",
        ...     "Dyspnoea": "A binary variable representing whether a person has shortness of breath."
        ... }
        >>> dag = ExpertInLoop(df).estimate(
        ...     effect_size_threshold=0.0001,
        ...     orientation_fn=llm_pairwise_orient,
        ...     variable_descriptions=variable_descriptions,
        ...     llm_model="gemini/gemini-1.5-flash"
        ... )
        >>> dag.edges()
        OutEdgeView([('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea'), ('Pollution', 'Cancer')])

        >>> # Using a custom orientation function
        >>> def my_orientation_func(var1, var2, **kwargs):
        ...     # Custom logic to determine edge orientation
        ...     if var1 == "Pollution" and var2 == "Cancer":
        ...         return ("Pollution", "Cancer")  # Pollution -> Cancer
        ...     elif var1 == "Cancer" and var2 == "Pollution":
        ...         return ("Pollution", "Cancer")  # Pollution -> Cancer
        ...     elif "Smoker" in (var1, var2) and "Cancer" in (var1, var2):
        ...         return ("Smoker", "Cancer")  # Smoker -> Cancer
        ...     # For edges involving Xray, always orient from other variable to Xray
        ...     elif "Xray" in (var1, var2):
        ...         if var1 == "Xray":
        ...             return (var2, var1)
        ...         else:
        ...             return (var1, var2)
        ...     # Default: use alphabetical ordering
        ...     return (var1, var2) if var1 < var2 else (var2, var1)
        >>> dag = ExpertInLoop(df).estimate(
        ...     effect_size_threshold=0.0001,
        ...     orientation_fn=my_orientation_func
        ... )
        >>> dag.edges()
        OutEdgeView([('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea'), ('Pollution', 'Cancer')])
        """
        # Step 0: Create a new DAG on all the variables
        nodes = list(self.data.columns)
        dag = DAG()
        dag.add_nodes_from(nodes)

        # Add required edges from expert knowledge
        if self.expert_knowledge is not None:
            if show_progress:
                logger.info("Adding required edges from expert knowledge...")
            for edge in self.expert_knowledge.required_edges:
                dag.add_edge(*edge)

        blacklisted_edges = []
        while True:
            # Step 1: Compute effects and p-values between every combination of variables
            all_effects = self.test_all(dag)

            # Step 2: Remove any edges between variables that are not sufficiently associated
            edge_effects = all_effects[all_effects.edge_present == True]
            edge_effects = edge_effects[
                (edge_effects.effect < effect_size_threshold)
                | (edge_effects.p_val > pval_threshold)
            ]
            edges_to_remove = []
            for _, row in edge_effects.iterrows():
                edge = (row["u"], row["v"])
                # Don't remove required edges even if they don't meet thresholds
                if self.expert_knowledge is not None and (
                    edge in self.expert_knowledge.required_edges
                    or (edge[1], edge[0]) in self.expert_knowledge.required_edges
                ):
                    continue
                edges_to_remove.append(edge)

            if len(edges_to_remove) > 0:
                if show_progress:
                    logger.info(
                        f"Removing edges with insufficient association: {edges_to_remove}"
                    )
                for edge in edges_to_remove:
                    try:
                        dag.remove_edge(*edge)
                    except Exception:
                        try:
                            dag.remove_edge(edge[1], edge[0])
                        except Exception:
                            pass
                continue

            # Step 3: Add edges between variables that are sufficiently associated
            no_edge_effects = all_effects[all_effects.edge_present == False]
            no_edge_effects = no_edge_effects[
                (no_edge_effects.effect > effect_size_threshold)
                & (no_edge_effects.p_val < pval_threshold)
            ]

            if len(no_edge_effects) == 0:
                break

            # Get the edge with maximum effect size
            max_effect_idx = no_edge_effects.effect.argmax()
            u, v = (
                no_edge_effects.iloc[max_effect_idx].u,
                no_edge_effects.iloc[max_effect_idx].v,
            )

            # Skip if this edge has been blacklisted
            if (u, v) in blacklisted_edges or (v, u) in blacklisted_edges:
                continue

            # Step 4: Get edge orientation
            if (u, v) in orientations:
                edge = (u, v)
            elif (v, u) in orientations:
                edge = (v, u)
            else:
                # Check cache first if use_cache is True
                cache_hit = False
                if use_cache:
                    for e in self.orientation_cache:
                        if set([u, v]) == set(e):
                            edge = e
                            cache_hit = True
                            break

                if not cache_hit:
                    edge = self._resolve_edge_direction(u, v, orientation_fn, **kwargs)
                    if edge is not None and use_cache:
                        self.orientation_cache.add(edge)

            if edge is None:
                blacklisted_edges.append((u, v))
                continue

            # Step 5: Add the edge if it doesn't create a cycle
            try:
                if show_progress:
                    logger.info(f"Adding edge: {edge}")
                dag.add_edge(*edge)
            except Exception:
                blacklisted_edges.append(edge)
                if show_progress:
                    logger.warning(
                        f"Adding edge {edge} would create a cycle. Blacklisting it."
                    )

        return dag
