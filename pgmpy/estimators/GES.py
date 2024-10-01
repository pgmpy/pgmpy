from itertools import combinations

import networkx as nx
import numpy as np

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import (
    AICScore,
    AICScoreGauss,
    BDeuScore,
    BDsScore,
    BicScore,
    BicScoreGauss,
    K2Score,
    ScoreCache,
    StructureEstimator,
    StructureScore,
)
from pgmpy.global_vars import logger


class GES(StructureEstimator):
    def __init__(self, data, use_cache=True, **kwargs):
        self.use_cache = use_cache

        super(GES, self).__init__(data=data, **kwargs)

    def _legal_edge_additions(self, current_model):
        """
        Find all edges that can be added to the graph such that it remains a DAG.
        """
        edges = []
        for u, v in combinations(current_model.nodes(), 2):
            if not (current_model.has_edge(u, v) or current_model.has_edge(v, u)):
                if not nx.has_path(current_model, v, u):
                    edges.append((u, v))
                if not nx.has_path(current_model, u, v):
                    edges.append((v, u))
        return edges

    def _legal_edge_flips(self, current_model):
        potential_flips = []
        edges = list(current_model.edges())
        for u, v in edges:
            current_model.remove_edge(u, v)
            if not nx.has_path(current_model, u, v):
                potential_flips.append((v, u))

            # Restore the edge to get to the original model
            current_model.add_edge(u, v)
        return potential_flips

    def estimate(self, scoring_method="bic", debug=False):
        # Step 1: Initial checks and setup for arguments
        # Step 1.1: Check scoring_method
        supported_methods = {
            "k2": K2Score,
            "bdeu": BDeuScore,
            "bds": BDsScore,
            "bic": BicScore,
            "aic": AICScore,
            "aic-g": AICScoreGauss,
            "bic-g": BicScoreGauss,
        }
        if isinstance(scoring_method, str):
            if scoring_method.lower() in [
                "k2score",
                "bdeuscore",
                "bdsscore",
                "bicscore",
                "aicscore",
            ]:
                raise ValueError(
                    f"The scoring method names have been changed. Please refer the documentation."
                )
            elif scoring_method.lower() not in list(supported_methods.keys()):
                raise ValueError(
                    f"Unknown scoring method. Please refer documentation for a list of supported score metrics."
                )
        elif not isinstance(scoring_method, StructureScore):
            raise ValueError(
                "scoring_method should either be one of k2score, bdeuscore, bicscore, bdsscore, aicscore, or an instance of StructureScore"
            )

        if isinstance(scoring_method, str):
            score = supported_methods[scoring_method.lower()](data=self.data)
        else:
            score = scoring_method

        if self.use_cache:
            score_fn = ScoreCache(score, self.data).local_score
        else:
            score_fn = score.local_score

        current_model = DAG()
        current_model.add_nodes_from(list(self.data.columns))
        # Step 2: Forward step: Iteratively add edges till score stops improving.
        while True:
            potential_edges = self._legal_edge_additions(current_model)
            score_deltas = np.zeros(len(potential_edges))
            for index, (u, v) in enumerate(potential_edges):
                current_parents = current_model.get_parents(v)
                score_delta = score_fn(v, current_parents + [u]) - score_fn(
                    v, current_parents
                )
                score_deltas[index] = score_delta

            if (len(potential_edges) == 0) or (np.all(score_deltas <= 0)):
                break

            edge_to_add = potential_edges[np.argmax(score_deltas)]
            current_model.add_edge(edge_to_add[0], edge_to_add[1])
            if debug:
                logger.info(
                    f"Adding edge {edge_to_add[0]} -> {edge_to_add[1]}. Improves score by: {score_deltas.max()}"
                )

        # Step 3: Backward Step: Iteratively remove edges till score stops improving.
        while True:
            potential_removals = list(current_model.edges())
            score_deltas = np.zeros(len(potential_removals))

            for index, (u, v) in enumerate(potential_removals):
                current_parents = current_model.get_parents(v)
                score_deltas[index] = score_fn(
                    v, [node for node in current_parents if node != u]
                ) - score_fn(v, current_parents)
            if (len(potential_removals) == 0) or (np.all(score_deltas <= 0)):
                break
            edge_to_remove = potential_removals[np.argmax(score_deltas)]
            current_model.remove_edge(edge_to_remove[0], edge_to_remove[1])
            if debug:
                logger.info(
                    f"Removing edge {edge_to_remove[0]} -> {edge_to_remove[1]}. Improves score by: {score_deltas.max()}"
                )

        # Step 4: Flip Edges: Iteratively try to flip edges till score stops improving.
        while True:
            potential_flips = self._legal_edge_flips(current_model)
            score_deltas = np.zeros(len(potential_flips))
            for index, (u, v) in enumerate(potential_flips):
                v_parents = current_model.get_parents(v)
                u_parents = current_model.get_parents(u)
                score_deltas[index] = (
                    score_fn(v, v_parents + [u]) - score_fn(v, v_parents)
                ) + (
                    score_fn(u, [node for node in u_parents if node != v])
                    - score_fn(u, u_parents)
                )

            if (len(potential_flips) == 0) or (np.all(score_deltas <= 0)):
                break
            edge_to_flip = potential_flips[np.argmax(score_deltas)]
            current_model.remove_edge(edge_to_flip[1], edge_to_flip[0])
            current_model.add_edge(edge_to_flip[0], edge_to_flip[1])
            if debug:
                logger.info(
                    f"Fliping edge {edge_to_flip[1]} -> {edge_to_flip[0]}. Improves score by: {score_deltas.max()}"
                )

        return current_model
