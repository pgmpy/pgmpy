"""Deprecated compatibility shim for :class:`pgmpy.causal_discovery.PC`."""

import warnings
from collections.abc import Callable, Hashable

import networkx as nx
import pandas as pd

from pgmpy.base import DAG, PDAG, UndirectedGraph
from pgmpy.causal_discovery import PC as _PC
from pgmpy.estimators.BaseConstraintEstimator import BaseConstraintEstimator
from pgmpy.estimators.ExpertKnowledge import ExpertKnowledge
from pgmpy.independencies import Independencies


class PC(BaseConstraintEstimator):
    """
    Deprecated: use :class:`pgmpy.causal_discovery.PC` instead.

    Class for constraint-based estimation of DAGs using the PC algorithm.
    This class delegates to the canonical implementation in
    `pgmpy.causal_discovery`; refer to its documentation for details.

    Parameters
    ----------
    data: pandas DataFrame object, optional
        dataframe object where each column represents one variable.

    independencies: pgmpy.independencies.Independencies, optional
        Known independence assertions to estimate the structure from instead
        of (or in addition to) data.
    """

    def __init__(
        self,
        data: pd.DataFrame | None = None,
        independencies: Independencies | None = None,
        **kwargs,
    ) -> None:
        warnings.warn(
            "PC is deprecated and will be removed in v2.0. Please use pgmpy.causal_discovery.PC instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(data=data, independencies=independencies, **kwargs)

    @staticmethod
    def orient_colliders(
        skeleton: UndirectedGraph,
        separating_sets: dict[frozenset, set],
        temporal_ordering: dict[Hashable, int] = dict(),
    ) -> PDAG:
        """
        Orients the edges that form v-structures in a graph skeleton based on
        information from `separating_sets` to form a DAG pattern (PDAG).
        Delegates to :meth:`pgmpy.causal_discovery.PC._orient_colliders`.
        """
        est = _PC()
        est.skeleton_ = skeleton
        est.separating_sets_ = separating_sets
        return est._orient_colliders(temporal_ordering=temporal_ordering)

    def estimate(
        self,
        variant: str = "parallel",
        ci_test: str | Callable | None = None,
        return_type: str = "pdag",
        significance_level: float = 0.01,
        max_cond_vars: int = 5,
        expert_knowledge: ExpertKnowledge | None = None,
        enforce_expert_knowledge: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True,
        **kwargs,
    ) -> DAG | PDAG | tuple[nx.Graph, dict[tuple[str, str], set[str]]]:
        """
        Estimates a DAG/PDAG from the data or independencies by delegating to
        :class:`pgmpy.causal_discovery.PC`; refer to its documentation for
        parameter details.

        Notes
        -----
        `enforce_expert_knowledge` is retained for backwards compatibility but
        no longer changes behavior: the canonical implementation always
        enforces expert knowledge (required/forbidden edges restrict the
        skeleton search and orientations are applied as hard constraints).

        Returns
        -------
        Estimated model: pgmpy.base.DAG, pgmpy.base.PDAG, or tuple(networkx.UndirectedGraph, dict)
            The estimated model structure. For `return_type="skeleton"`,
            returns a tuple of the skeleton and the separating sets.
        """
        if self.data is None:
            raise ValueError(
                "Estimating the structure from independencies alone (without `data`) is no "
                "longer supported. Please provide `data` to the constructor."
            )

        if return_type.lower() == "skeleton":
            return self.build_skeleton(
                variant=variant,
                ci_test=ci_test,
                significance_level=significance_level,
                max_cond_vars=max_cond_vars,
                expert_knowledge=expert_knowledge,
                n_jobs=n_jobs,
                show_progress=show_progress,
                **kwargs,
            )

        est = _PC(
            variant=variant,
            ci_test=ci_test,
            return_type=return_type.lower(),
            significance_level=significance_level,
            max_cond_vars=max_cond_vars,
            expert_knowledge=expert_knowledge,
            n_jobs=n_jobs,
            show_progress=show_progress,
        )
        return est.fit(self.data, independencies=self.independencies).causal_graph_
