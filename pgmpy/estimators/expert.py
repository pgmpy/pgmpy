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
        self.nodes = list(data.columns) if data is not None else []

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
        return pd.DataFrame(cis, columns=["u", "v", "z", "edge_present", "effect", "p_val"])

    def prune(self, dag, pval_threshold=0.05, effect_size_threshold=0.05):
        """
        Prunes edges from the DAG based on conditional independence tests.

        Edges with effect size below `effect_size_threshold` and p-value above
        `pval_threshold` are removed. Required edges may be removed if unsupported by data.

        Parameters
        ----------
        dag : pgmpy.base.DAG
            The DAG to prune.
        pval_threshold : float
            P-value threshold for CI tests.
        effect_size_threshold : float
            Effect size threshold for edge retention.

        Returns
        -------
        pgmpy.base.DAG
            The pruned DAG.
        """
        all_effects = self.test_all(dag)
        edge_effects = all_effects[all_effects.edge_present == True]
        edge_effects = edge_effects[
            (edge_effects.effect < effect_size_threshold) & (edge_effects.p_val > pval_threshold)
        ]
        remove_edges = list(edge_effects.loc[:, ("u", "v")].to_records(index=False))
        for u, v in remove_edges:
            dag.remove_edge(u, v)
            logger.info(f"Pruned edge {u}->{v} based on CI test.")
        return dag

    def estimate(
        self,
        pval_threshold=0.05,
        effect_size_threshold=0.05,
        orientation_fn=llm_pairwise_orient,
        orientations=None,
        expert_knowledge=None,
        use_cache=True,
        show_progress=True,
        **kwargs,
    ):
        """
        Estimates a DAG from data using expert knowledge and interactive edge orientation.

        Initializes the DAG with required edges, respects forbidden edges, and uses
        temporal order to orient edges before querying user/LLM. Required edges may be
        removed during pruning if unsupported by data.

        Parameters
        ----------
        pval_threshold : float
            P-value threshold for CI tests to determine significant associations.
        effect_size_threshold : float
            Effect size threshold for adding/removing edges.
        orientation_fn : callable, default llm_pairwise_orient
            Function to determine edge orientation, taking at least two arguments (var1, var2)
            and returning a tuple (source, target) or None.
        orientations : set, optional
            Set of preferred edge orientations to override orientation_fn.
        expert_knowledge : ExpertKnowledge, optional
            Expert knowledge specifying forbidden_edges, required_edges, and temporal_order.
        use_cache : bool, default True
            If True, caches orientation_fn results for reuse.
        show_progress : bool, default True
            If True, logs progress information.
        **kwargs : additional arguments
            Passed to orientation_fn.

        Returns
        -------
        pgmpy.base.DAG
            The learned causal structure.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.estimators import ExpertInLoop, ExpertKnowledge
        >>> model = get_example_model('cancer')
        >>> data = model.simulate(n_samples=1000)
        >>> expert_knowledge = ExpertKnowledge(
        ...     forbidden_edges=[('Cancer', 'Pollution')],
        ...     required_edges=[('Smoker', 'Cancer')],
        ...     temporal_order=[['Pollution', 'Smoker'], ['Cancer'], ['Xray', 'Dyspnoea']]
        ... )
        >>> est = ExpertInLoop(data)
        >>> dag = est.estimate(
        ...     expert_knowledge=expert_knowledge,
        ...     orientation_fn=manual_pairwise_orient,
        ...     effect_size_threshold=0.0001
        ... )
        >>> dag.edges()
        """
        # Initialize DAG with nodes
        nodes = list(self.data.columns)
        dag = DAG()
        dag.add_nodes_from(nodes)

        # Handle expert knowledge
        blacklisted_edges = set()
        temporal_ordering = {}
        if expert_knowledge:
            blacklisted_edges = expert_knowledge.forbidden_edges
            temporal_ordering = expert_knowledge.temporal_ordering
            # Add required edges, checking for acyclicity
            for u, v in expert_knowledge.required_edges:
                if (u, v) not in blacklisted_edges and (v, u) not in blacklisted_edges:
                    dag.add_edge(u, v)
                    if not dag.is_acyclic():
                        dag.remove_edge(u, v)
                        logger.warning(f"Ignoring required edge {u}->{v}: creates a cycle.")
                    else:
                        logger.info(f"Added required edge {u}->{v} to initial DAG.")

        orientations = orientations or set()

        while True:
            dag = self.prune(dag, pval_threshold, effect_size_threshold)

            all_effects = self.test_all(dag)

            nonedge_effects = all_effects[all_effects.edge_present == False]
            nonedge_effects = nonedge_effects[
                (nonedge_effects.effect >= effect_size_threshold) &
                (nonedge_effects.p_val <= pval_threshold)
            ]

            if blacklisted_edges:
                nonedge_effects = nonedge_effects[
                    ~nonedge_effects.apply(
                        lambda row: (row.u, row.v) in blacklisted_edges or (row.v, row.u) in blacklisted_edges,
                        axis=1
                    )
                ]

            if nonedge_effects.empty:
                break

            selected_edge = nonedge_effects.iloc[nonedge_effects.effect.argmax()]
            u, v = selected_edge.u, selected_edge.v

            edge_direction = None

            # Check temporal order first
            if temporal_ordering and u in temporal_ordering and v in temporal_ordering:
                u_tier = temporal_ordering[u]
                v_tier = temporal_ordering[v]
                if u_tier < v_tier:
                    edge_direction = (u, v)
                    logger.info(f"Oriented {u}->{v} based on temporal order.")
                elif v_tier < u_tier:
                    edge_direction = (v, u)
                    logger.info(f"Oriented {v}->{u} based on temporal order.")

            # Check provided orientations or cache
            if not edge_direction:
                if (u, v) in orientations:
                    edge_direction = (u, v)
                elif (v, u) in orientations:
                    edge_direction = (v, u)
                elif use_cache and (u, v) in self.orientation_cache:
                    edge_direction = (u, v)
                elif use_cache and (v, u) in self.orientation_cache:
                    edge_direction = (v, u)
                else:
                    edge_direction = orientation_fn(u, v, **kwargs)
                    if use_cache and edge_direction:
                        self.orientation_cache.add(edge_direction)
                        if show_progress:
                            logger.info(
                                f"Queried orientation for {u}-{v}. Got: {edge_direction[0]}->{edge_direction[1]}"
                            )

            if edge_direction is None:
                logger.info(f"No orientation for {u}-{v}. Blacklisting both directions.")
                blacklisted_edges.add((u, v))
                blacklisted_edges.add((v, u))
            elif nx.has_path(dag, edge_direction[1], edge_direction[0]):
                logger.info(f"Edge {edge_direction[0]}->{edge_direction[1]} creates cycle. Trying reverse.")
                reverse_edge = (edge_direction[1], edge_direction[0])
                if reverse_edge not in blacklisted_edges:
                    dag.add_edge(*reverse_edge)
                    blacklisted_edges.add(edge_direction)
                    logger.info(f"Added {reverse_edge[0]}->{reverse_edge[1]}.")
                else:
                    logger.info(f"Reverse edge {reverse_edge[0]}->{reverse_edge[1]} blacklisted. Skipping.")
            else:
                dag.add_edge(*edge_direction)
                logger.info(f"Added {edge_direction[0]}->{edge_direction[1]}.")

            if config.SHOW_PROGRESS and show_progress:
                logger.info(f"Current DAG edges: {list(dag.edges())}")

        if expert_knowledge:
            pdag = PDAG(directed_ebunch=dag.edges(), undirected_ebunch=[])
            pdag = expert_knowledge.apply_expert_knowledge(pdag)
            dag = DAG(pdag.directed_edges)

        return dag
