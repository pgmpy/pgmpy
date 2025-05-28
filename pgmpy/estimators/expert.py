import sys
from itertools import combinations

import networkx as nx
import pandas as pd

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.CITests import pillai_trace
from pgmpy.global_vars import logger
from pgmpy.utils import llm_pairwise_orient, manual_pairwise_orient


class ExpertInLoop(StructureEstimator):
    def __init__(self, data=None, **kwargs):
        super(ExpertInLoop, self).__init__(data=data, **kwargs)
        self.orientation_cache = set([])

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
        Estimates a DAG from the data by utilizing expert knowledge.

        The method iteratively adds and removes edges between variables
        (similar to Greedy Equivalence Search (GES) algorithm) based on a
        global score metric that improves the model's fit in each iteration.
        The score metric used is based on conditional independence testing.
        When adding an edge to the model, the method asks for expert knowledge
        to decide the orientation of the edge. Alternatively, an LLM can used
        to decide the orientation of the edge.

        Parameters
        ----------
        pval_threshold: float
            The p-value threshold to use for the test to determine whether
            there is a significant association between the variables or not.

        effect_size_threshold: float
            The effect size threshold to use to suggest a new edge. If the
            conditional effect size between two variables is greater than the
            threshold, the algorithm would suggest to add an edge between them.
            And if the effect size for an edge is less than the threshold,
            would suggest to remove the edge.

        orientation_fn: callable (default: pgmpy.utils.llm_pairwise_orient)
            A function to determine edge orientation. The function should at
            least take two arguments (the names of the two variables) and
            return either a tuple (source, target) representing the directed
            edge from source to target or None representing no edge between the
            variables. Any additional keyword arguments passed to estimate()
            will be forwarded to this function.

            Built-in functions that can be used:

            - `pgmpy.utils.manual_pairwise_orient`: Prompts the user to specify the direction
              between two variables by presenting options and taking input.

            - `pgmpy.utils.llm_pairwise_orient`: Uses a Large Language Model to determine direction.
              Requires additional parameters:

              * variable_descriptions: dict of {var_name: description} for context
              * llm_model: name of the LLM model (default: "gemini/gemini-1.5-flash")
              * system_prompt: optional custom system prompt

            Custom functions can be provided that implement any desired logic
            for determining edge orientation, including using local LLMs or
            domain-specific heuristics.

        orientations: set
            Users can specify a set of edges which would be used as the
            preferred orientation for edges over the output of orientation_fn.

        use_cache: bool
            If True, the method will cache the results returned by
            `orientation_fn` and reuse it in future calls of the `estimate`
            method instead of calling the `orientation_fn`.

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
        # Step 0: Create a new DAG on all the variables with no edge.
        nodes = list(self.data.columns)
        dag = DAG()
        dag.add_nodes_from(nodes)

        blacklisted_edges = []
        while True:
            # Step 1: Compute effects and p-values between every combination of variables.
            all_effects = self.test_all(dag)

            # Step 2: Remove any edges between variables that are not sufficiently associated.
            edge_effects = all_effects[all_effects.edge_present == True]
            edge_effects = edge_effects[
                (edge_effects.effect < effect_size_threshold)
                & (edge_effects.p_val > pval_threshold)
            ]
            remove_edges = list(edge_effects.loc[:, ("u", "v")].to_records(index=False))
            for edge in remove_edges:
                dag.remove_edge(edge[0], edge[1])

            # Step 3: Add edge between variables which have significant association.
            # Step 3.1: Find edges that are not present in the DAG but have significant association.
            nonedge_effects = all_effects[all_effects.edge_present == False]
            nonedge_effects = nonedge_effects[
                (nonedge_effects.effect >= effect_size_threshold)
                & (nonedge_effects.p_val <= pval_threshold)
            ]

            # Step 3.2: Remove any pair of variables that are blacklisted.
            if len(blacklisted_edges) > 0:
                blacklisted_edges_us = [edge[0] for edge in blacklisted_edges]
                blacklisted_edges_vs = [edge[1] for edge in blacklisted_edges]
                nonedge_effects = nonedge_effects.loc[
                    ~(
                        (
                            nonedge_effects.u.isin(blacklisted_edges_us)
                            & nonedge_effects.v.isin(blacklisted_edges_vs)
                        )
                        | (
                            nonedge_effects.u.isin(blacklisted_edges_vs)
                            & nonedge_effects.v.isin(blacklisted_edges_us)
                        )
                    ),
                    :,
                ]

            # Step 3.3: Exit loop if all correlations in data are explained by the model.
            if (edge_effects.shape[0] == 0) and (nonedge_effects.shape[0] == 0):
                break

            # Step 3.4: Find for the pair of variable with the highest effect size.
            selected_edge = nonedge_effects.iloc[nonedge_effects.effect.argmax()]

            # Step 3.5: Find the edge orientation for the selected pair of variables.
            #
            # 1. If `orientations` are provided, use them.
            # 2. Otherwise, try to use cached orientations if `use_cache=True`
            # 3. If no cached orientation, call the orientation_fn and validate result
            #    - Validate that it returns a valid edge direction tuple
            #    - Cache the orientation and add the edge to the DAG

            if (selected_edge.u, selected_edge.v) in orientations:
                edge_direction = (selected_edge.u, selected_edge.v)
            elif (selected_edge.v, selected_edge.u) in orientations:
                edge_direction = (selected_edge.v, selected_edge.u)
            elif (
                use_cache
                and (selected_edge.u, selected_edge.v) in self.orientation_cache
            ):
                edge_direction = (selected_edge.u, selected_edge.v)
            elif (
                use_cache
                and (selected_edge.v, selected_edge.u) in self.orientation_cache
            ):
                edge_direction = (selected_edge.v, selected_edge.u)
            else:
                edge_direction = orientation_fn(
                    selected_edge.u, selected_edge.v, **kwargs
                )
                if use_cache is True:
                    self.orientation_cache.add(edge_direction)

                if config.SHOW_PROGRESS and show_progress:
                    logger.info(
                        f"\rQueried for edge orientation between"
                        "{selected_edge.u} and {selected_edge.v}. Got:"
                        "{edge_direction[0]} -> {edge_direction[1]}"
                    )

            # Step 3.6: Try adding the edge to the DAG. If edge creates a
            #           cycle, add the reversed edge, and blacklist the original edge.
            if edge_direction is None:
                logger.info(
                    f"Orientation function returned None for edge {selected_edge.u} - {selected_edge.v}. "
                    "Skipping this edge."
                )
                blacklisted_edges.append((selected_edge.u, selected_edge.v))

            elif nx.has_path(dag, edge_direction[1], edge_direction[0]):
                blacklisted_edges.append(edge_direction)
                dag.add_edges_from([(edge_direction[1], edge_direction[0])])
            else:
                dag.add_edges_from([edge_direction])

        # Step 4: Return the final DAG.
        return dag
