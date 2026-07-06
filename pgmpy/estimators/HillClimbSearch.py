"""Deprecated compatibility shim for :class:`pgmpy.causal_discovery.HillClimbSearch`."""

import warnings
from types import SimpleNamespace

import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery import HillClimbSearch as _HillClimbSearch
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.ExpertKnowledge import ExpertKnowledge
from pgmpy.structure_score import BaseStructureScore


class HillClimbSearch(StructureEstimator):
    """
    Deprecated: use :class:`pgmpy.causal_discovery.HillClimbSearch` instead.

    Class for heuristic hill climb searches for DAGs, to learn network
    structure from data. This class delegates to the canonical implementation
    in `pgmpy.causal_discovery`.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.

    use_cache: bool (default: True)
        Retained for backwards compatibility; canonical structure scores cache
        local scores internally.
    """

    def __init__(self, data: pd.DataFrame, use_cache: bool = True, **kwargs):
        warnings.warn(
            """HillClimbSearch is deprecated and will be removed in v2.0. Please use
            pgmpy.causal_discovery.HillClimbSearch instead.""",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(data, **kwargs)

    def _legal_operations(
        self,
        model,
        score,
        structure_score,
        tabu_list,
        max_indegree,
        forbidden_edges,
        required_edges,
    ):
        """Legacy forwarder to the canonical `_legal_operations_dag`.

        `score` and `structure_score` are the legacy callables
        (`local_score(variable, parents)` and `structure_prior_ratio(operation)`).
        """
        scoring_method = SimpleNamespace(local_score=score, structure_prior_ratio=structure_score)
        est = _HillClimbSearch()
        # The canonical implementation reads the fitted `variables_` attribute.
        est.variables_ = list(model.nodes())
        return est._legal_operations_dag(
            model=model,
            scoring_method=scoring_method,
            tabu_list=tabu_list,
            max_indegree=max_indegree,
            forbidden_edges=forbidden_edges,
            required_edges=required_edges,
        )

    def estimate(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        start_dag: DAG | None = None,
        tabu_length: int = 100,
        max_indegree: int | None = None,
        expert_knowledge: ExpertKnowledge | None = None,
        epsilon: float = 1e-4,
        max_iter: int = int(1e6),
        show_progress: bool = True,
    ) -> DAG:
        """
        Performs local hill climb search to estimate the `DAG` structure with
        optimal score. Delegates to
        :class:`pgmpy.causal_discovery.HillClimbSearch`; refer to its
        documentation for parameter details.

        Returns
        -------
        Estimated model: pgmpy.base.DAG
            A `DAG` at a (local) score maximum.
        """
        est = _HillClimbSearch(
            scoring_method=scoring_method,
            start_dag=start_dag,
            tabu_length=tabu_length,
            max_indegree=max_indegree,
            expert_knowledge=expert_knowledge,
            return_type="dag",
            epsilon=epsilon,
            max_iter=max_iter,
            show_progress=show_progress,
        )
        return est.fit(self.data).causal_graph_
