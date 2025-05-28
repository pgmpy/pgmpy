import sys
from itertools import combinations

import networkx as nx
import pandas as pd
from six import iteritems

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import ExpertKnowledge, StructureEstimator
from pgmpy.estimators.CITests import pillai_trace
from pgmpy.global_vars import logger
from pgmpy.utils import llm_pairwise_orient, manual_pairwise_orient


class ExpertInLoop(StructureEstimator):
    def __init__(self, data=None, expert_knowledge=None, **kwargs):
        """
        Initialize the ExpertInLoop estimator for causal discovery with expert knowledge support.

        Parameters
        ----------
        data: pandas.DataFrame, optional
            DataFrame containing the dataset for causal discovery.
        expert_knowledge: pgmpy.estimators.ExpertKnowledge, optional
            Expert knowledge specifying forbidden_edges, required_edges, and temporal_order.
        **kwargs:
            Additional arguments passed to the parent class.
        """
        super(ExpertInLoop, self).__init__(data=data, **kwargs)
        self.orientation_cache = set([])
        self.expert_knowledge = expert_knowledge
        self.blacklisted_edges = (
            set(expert_knowledge.forbidden_edges)
            if expert_knowledge and hasattr(expert_knowledge, "forbidden_edges")
            else set([])
        )

        # Initialize DAG with required_edges
        self.dag = DAG()
        if self.data is not None:
            self.dag.add_nodes_from(self.data.columns)
        if expert_knowledge and hasattr(expert_knowledge, "required_edges"):
            for edge in expert_knowledge.required_edges:
                self.dag.add_edge(*edge)

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
        Estimates a DAG from the data by utilizing expert knowledge and interactive edge orientation.

        The method iteratively adds and removes edges between variables
        (similar to Greedy Equivalence Search (GES) algorithm) based on a
        global score metric that improves the model's fit in each iteration.
        The score metric used is based on conditional independence testing.
        When adding an edge to the model, the method first checks the temporal order
        specified in expert_knowledge (if provided) to decide the orientation, then
        uses provided orientations, cached orientations, or queries an orientation function.

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
        orientations: set
            Users can specify a set of edges which would be used as the
            preferred orientation for edges over the output of orientation_fn.
        use_cache: bool
            If True, the method will cache the results returned by
            `orientation_fn` and reuse it in future calls of the `estimate`
            method instead of calling the `orientation_fn`.
        show_progress: bool (default: True)
            If True, prints info of the running status.
        **kwargs:
            Any additional parameters to pass to the `orientation_fn`.

        Returns
        -------
        pgmpy.base.DAG: A DAG representing the learned causal structure.

        Notes
        -----
        - If expert_knowledge is provided, forbidden_edges are never added to the DAG
          and are included in blacklisted_edges.
        - Required edges are included in the initial DAG but may be removed if their
          effect size falls below effect_size_threshold or p-value exceeds pval_threshold.
        - Temporal order (if provided) is used to orient edges before checking orientations,
          cache, or calling orientation_fn.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model, llm_pairwise_orient
        >>> from pgmpy.estimators import ExpertInLoop, ExpertKnowledge
        >>> model = get_example_model('cancer')
        >>> df = model.simulate(int(1e3))
        >>> expert_knowledge = ExpertKnowledge(
        ...     forbidden_edges=[('Pollution', 'Cancer')],
        ...     required_edges=[('Smoker', 'Cancer')],
        ...     temporal_order=['Pollution', 'Smoker', 'Cancer', 'Xray', 'Dyspnoea']
        ... )
        >>> estimator = ExpertInLoop(df, expert_knowledge=expert_knowledge)
        >>> dag = estimator.estimate(
        ...     effect_size_threshold=0.0001,
        ...     orientation_fn=llm_pairwise_orient,
        ...     variable_descriptions={
        ...         "Smoker": "Whether a person smokes or not.",
        ...         "Cancer": "Whether a person has cancer.",
        ...         "Xray": "Result of an X-ray test.",
        ...         "Pollution": "Whether in a high-pollution area.",
        ...         "Dyspnoea": "Whether a person has shortness of breath."
        ...     }
        ... )
        >>> dag.edges()
        OutEdgeView([('Smoker', 'Cancer'), ('Cancer', 'Xray'), ('Cancer', 'Dyspnoea')])
        """
        # Step 0: Initialize DAG (already includes required_edges from __init__)
        dag = self.dag.copy()

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

            # Step 3.2: Remove any pair of variables that are blacklisted (includes forbidden_edges).
            if len(self.blacklisted_edges) > 0:
                blacklisted_edges_us = [edge[0] for edge in self.blacklisted_edges]
                blacklisted_edges_vs = [edge[1] for edge in self.blacklisted_edges]
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

            # Step 3.4: Find the pair of variables with the highest effect size.
            selected_edge = nonedge_effects.iloc[nonedge_effects.effect.argmax()]

            # Step 3.5: Find the edge orientation for the selected pair of variables.
            # 1. Check temporal_order (if provided)
            # 2. Use provided orientations
            # 3. Use cached orientations if use_cache=True
            # 4. Call orientation_fn and validate result
            edge_direction = None
            if self.expert_knowledge and hasattr(
                self.expert_knowledge, "temporal_order"
            ):
                node1_idx = (
                    self.expert_knowledge.temporal_order.index(selected_edge.u)
                    if selected_edge.u in self.expert_knowledge.temporal_order
                    else float("inf")
                )
                node2_idx = (
                    self.expert_knowledge.temporal_order.index(selected_edge.v)
                    if selected_edge.v in self.expert_knowledge.temporal_order
                    else float("inf")
                )
                if node1_idx < node2_idx:
                    edge_direction = (selected_edge.u, selected_edge.v)
                elif node2_idx < node1_idx:
                    edge_direction = (selected_edge.v, selected_edge.u)

            if edge_direction is None:
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
                            f"\rQueried for edge orientation between "
                            f"{selected_edge.u} and {selected_edge.v}. Got: "
                            f"{edge_direction[0]} -> {edge_direction[1]}"
                        )

            # Step 3.6: Try adding the edge to the DAG. If edge creates a
            #           cycle, add the reversed edge, and blacklist the original edge.
            if edge_direction is None:
                logger.info(
                    f"Orientation function returned None for edge {selected_edge.u} - {selected_edge.v}. "
                    "Skipping this edge."
                )
                self.blacklisted_edges.add((selected_edge.u, selected_edge.v))

            elif nx.has_path(dag, edge_direction[1], edge_direction[0]):
                self.blacklisted_edges.add(edge_direction)
                dag.add_edges_from([(edge_direction[1], edge_direction[0])])
            else:
                dag.add_edges_from([edge_direction])

        # Step 4: Return the final DAG.
        return dag
