from itertools import combinations

import networkx as nx
import pandas as pd

from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.CITests import ci_pillai
from pgmpy.utils import llm_pairwise_orient


class ExpertInLoop(StructureEstimator):
    def __init__(self, data=None, **kwargs):
        super(ExpertInLoop, self).__init__(data=data, **kwargs)

    def test_all(self, dag):
        """
        Runs CI tests on all possible combinations of variables in the model.

        Parameters
        ----------
        dag: pgmpy.base.DAG
            The DAG on which to run the tests.

        Returns
        -------
        pd.DataFrame: The results of all the tests.
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
            effect, p_value = ci_pillai(
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
        variable_descriptions=None,
    ):
        """
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

        variable_descriptions: dict
            A dict of the form {var: description}.
        """
        nodes = list(self.data.columns)
        dag = DAG()
        dag.add_nodes_from(nodes)

        blacklisted_edges = []
        while True:
            all_effects = self.test_all(dag)

            edge_effects = all_effects[all_effects.edge_present == True]
            edge_effects = edge_effects[
                (edge_effects.effect < effect_size_threshold)
                & (edge_effects.p_val > pval_threshold)
            ]
            remove_edges = list(edge_effects.loc[:, ("u", "v")].to_records(index=False))
            # print(f"Removing edges: {remove_edges}")
            for edge in remove_edges:
                dag.remove_edge(edge[0], edge[1])

            nonedge_effects = all_effects[all_effects.edge_present == False]
            nonedge_effects = nonedge_effects[
                (nonedge_effects.effect >= effect_size_threshold)
                & (nonedge_effects.p_val <= pval_threshold)
            ]

            if (edge_effects.shape[0] == 0) and (nonedge_effects.shape[0] == 0):
                break

            if len(blacklisted_edges) > 0:
                nonedge_effects = nonedge_effects.loc[
                    (
                        (nonedge_effects.u in [edge[0] for edge in blacklisted_edges])
                        & (nonedge_effects.v in [edge[1] for edge in blacklisted_edges])
                    )
                    or (
                        (nonedge_effects.u in [edge[1] for edge in blacklisted_edges])
                        & (nonedge_effects.v in [edge[0] for edge in blacklisted_edges])
                    ),
                    :,
                ]
            if len(blacklisted_edges) > 0:
                import ipdb

                ipdb.set_trace()
            selected_edge = nonedge_effects.iloc[nonedge_effects.effect.argmax()]
            # print(f"Adding: {selected_edge.u} -- {selected_edge.v}")
            edge_direction = llm_pairwise_orient(
                selected_edge.u, selected_edge.v, variable_descriptions
            )
            if nx.has_path(dag, edge_direction[1], edge_direction[0]):
                # print(f"Blacklisting: {edge_direction}")
                blacklisted_edges.append(edge_direction)
            else:
                dag.add_edges_from([edge_direction])

        return dag
