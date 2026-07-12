"""Deprecated compatibility shim over :class:`pgmpy.causal_discovery`'s constraint-based machinery."""

import warnings
from collections.abc import Callable

from sklearn.base import clone

from pgmpy.base import UndirectedGraph
from pgmpy.estimators import StructureEstimator


class BaseConstraintEstimator(StructureEstimator):
    """
    Deprecated: use the constraint-based estimators in
    `pgmpy.causal_discovery` (e.g. :class:`pgmpy.causal_discovery.PC`)
    instead.

    Base class for constraint-based structure learning algorithms. The
    skeleton discovery step delegates to the canonical implementation in
    `pgmpy.causal_discovery`.

    Parameters
    ----------
    data: pandas.DataFrame or array-like, optional
        The dataset on which to perform structure learning. Each column
        corresponds to a variable.

    independencies: pgmpy.independencies.Independencies instance, optional
        A set of pre-specified conditional independence assertions. If provided,
        the estimator will use these instead of testing conditional
        independencies from data.

    **kwargs:
        Additional keyword arguments passed to `StructureEstimator`.
    """

    def __init__(self, data=None, independencies=None, **kwargs):
        # Subclass shims (e.g. the deprecated PC) emit their own warning.
        if type(self) is BaseConstraintEstimator:
            warnings.warn(
                "BaseConstraintEstimator is deprecated and will be removed in v2.0. "
                "Please use the constraint-based estimators in pgmpy.causal_discovery instead.",
                FutureWarning,
                stacklevel=2,
            )
        super().__init__(data, independencies, **kwargs)

    def build_skeleton(
        self,
        variant: str = "stable",
        ci_test: str | Callable | None = None,
        significance_level: float = 0.01,
        max_cond_vars: int = 5,
        expert_knowledge=None,
        enforce_expert_knowledge: bool = False,
        n_jobs: int = -1,
        show_progress: bool = True,
        **kwargs,
    ) -> tuple[UndirectedGraph, dict[tuple[str, str], set[str]]]:
        """
        Estimates a graph skeleton (UndirectedGraph) from data or
        independencies using (the first part of) the PC algorithm. Delegates
        to the canonical skeleton search in `pgmpy.causal_discovery`; refer to
        its documentation for parameter details.

        Notes
        -----
        `enforce_expert_knowledge` is retained for backwards compatibility but
        no longer changes behavior: the canonical implementation always
        enforces expert knowledge during the skeleton search.

        Returns
        -------
        skeleton: UndirectedGraph
            An estimate for the undirected graph skeleton of the BN underlying the data.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set of variables that makes them conditionally
            independent.
        """
        from pgmpy.causal_discovery import PC as _PC

        if self.data is None:
            raise ValueError(
                "Estimating the skeleton from independencies alone (without `data`) is no "
                "longer supported. Please provide `data` to the constructor."
            )

        if expert_knowledge is not None:
            # The canonical skeleton search consumes fitted (`*_`) expert
            # knowledge attributes; fit on a fresh copy so the user's object
            # is not mutated.
            expert_knowledge = clone(expert_knowledge)
            expert_knowledge.fit(self.data)

        return _PC()._build_skeleton(
            data=self.data,
            independencies=self.independencies,
            variant=variant,
            ci_test=ci_test,
            significance_level=significance_level,
            max_cond_vars=max_cond_vars,
            expert_knowledge=expert_knowledge,
            n_jobs=n_jobs,
            show_progress=show_progress,
            **kwargs,
        )
