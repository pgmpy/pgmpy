import sys
from itertools import combinations

import networkx as nx
import pandas as pd

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.CITests import pillai_trace
from pgmpy.utils import llm_pairwise_orient, manual_pairwise_orient


class ExpertInLoop(StructureEstimator):
    def __init__(self, data=None, **kwargs):
        super(ExpertInLoop, self).__init__(data=data, **kwargs)
        self.orientations_llm = set([])

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
        use_llm=True,
        llm_model="gemini/gemini-1.5-flash",
        variable_descriptions=None,
        show_progress=True,
        orientations=set([]),
        use_cache=True,
        **kwargs,
    ):
        """
        Estimates a DAG from the data by utilizing expert knowledge.

        The method iteratively adds and removes edges between variables
        (similar to Greedy Equivalence Search algorithm) based on a score
        metric that improves the model's fit to the data the most. The score
        metric used is based on conditional independence testing. When adding
        an edge to the model, the method asks for expert knowledge to decide
        the orientation of the edge. Alternatively, an LLM can used to decide
        the orientation of the edge.

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

        use_llm: bool
            Whether to use a Large Language Model for edge orientation. If
            False, prompts the user to specify the direction between the edges.

        llm_model: str (default: gemini/gemini-1.5-flash)
            The LLM model to use. Please refer to litellm documentation (https://docs.litellm.ai/docs/providers)
            for available model options. Default is gemini-1.5-flash

        variable_descriptions: dict
            A dict of the form {var: description} giving a text description of
            each variable in the model.

        show_progress: bool (default: True)
            If True, prints info of the running status.

        orientations: set
            preferred orientation for edges

        use_cache: bool
            If False, ask LLM (the same question multiple times)

        kwargs: kwargs
            Any additional parameters to pass to litellm.completion method.
            Please refer documentation at: https://docs.litellm.ai/docs/completion/input#input-params-1

        Returns
        -------
        pgmpy.base.DAG: A DAG representing the learned causal structure.

        Examples
        --------
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.estimators import ExpertInLoop
        >>> model = get_example_model('cancer')
        >>> df = model.simulate(int(1e3))
        >>> variable_descriptions = {
        ...     "Smoker": "A binary variable representing whether a person smokes or not.",
        ...     "Cancer": "A binary variable representing whether a person has cancer. ",
        ...     "Xray": "A binary variable representing the result of an X-ray test.",
        ...     "Pollution": "A binary variable representing whether the person is in a high-pollution area or not."
        ...     "Dyspnoea": "A binary variable representing whether a person has shortness of breath. "}
        >>> dag = ExpertInLoop(df).estimate(
        ...                 effect_size_threshold=0.0001,
        ...                 use_llm=True,
        ...                 variable_descriptions=variable_descriptions)
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
            nonedge_effects = all_effects[all_effects.edge_present == False]
            nonedge_effects = nonedge_effects[
                (nonedge_effects.effect >= effect_size_threshold)
                & (nonedge_effects.p_val <= pval_threshold)
            ]

            # Step 3.2: Else determine the edge direction and add it if not in blacklisted_edges.
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

            # Step 3.1: Exit loop if all correlations in data are explained by the model.
            if (edge_effects.shape[0] == 0) and (nonedge_effects.shape[0] == 0):
                break

            selected_edge = nonedge_effects.iloc[nonedge_effects.effect.argmax()]
            edge_direction = None
            if use_llm:
                if use_cache:
                    if (selected_edge.u, selected_edge.v) in self.orientations_llm:
                        edge_direction = (selected_edge.u, selected_edge.v)
                    elif (selected_edge.v, selected_edge.u) in self.orientations_llm:
                        edge_direction = (selected_edge.v, selected_edge.u)
                if edge_direction is None:
                    edge_direction = llm_pairwise_orient(
                        selected_edge.u,
                        selected_edge.v,
                        variable_descriptions,
                        llm_model=llm_model,
                        **kwargs,
                    )
                    self.orientations_llm.add(edge_direction)

                if config.SHOW_PROGRESS and show_progress:
                    sys.stdout.write(
                        f"\rQueried for edge orientation between {selected_edge.u} and {selected_edge.v}. Got: {edge_direction[0]} -> {edge_direction[1]}"
                    )
                    sys.stdout.flush()
            elif orientations:
                if (selected_edge.u, selected_edge.v) in orientations:
                    edge_direction = (selected_edge.u, selected_edge.v)
                elif (selected_edge.v, selected_edge.u) in orientations:
                    edge_direction = (selected_edge.v, selected_edge.u)
            if edge_direction is None:
                edge_direction = manual_pairwise_orient(
                    selected_edge.u, selected_edge.v
                )
                orientations.add(edge_direction)

            # Step 3.3: If the edge creates a cycle add the reverse edge. If no cycle, add the original edge.
            if nx.has_path(dag, edge_direction[1], edge_direction[0]):
                blacklisted_edges.append(edge_direction)
                dag.add_edges_from([(edge_direction[1], edge_direction[0])])
            else:
                dag.add_edges_from([edge_direction])
        return dag
