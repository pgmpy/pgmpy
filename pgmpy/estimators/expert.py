"""Deprecated compatibility shims for :class:`pgmpy.causal_discovery.ExpertInLoop`
and :class:`pgmpy.causal_discovery.LLMPairwise`."""

import warnings
from collections.abc import Callable, Hashable

import pandas as pd

from pgmpy.base import DAG
from pgmpy.causal_discovery import ExpertInLoop as _ExpertInLoop
from pgmpy.causal_discovery import LLMPairwise as _LLMPairwise
from pgmpy.ci_tests import BaseCITest, get_ci_test
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.ExpertKnowledge import ExpertKnowledge


def llm_pairwise_orient(
    x,
    y,
    descriptions,
    system_prompt=None,
    llm_model="gemini/gemini-2.5-flash",
    **kwargs,
):
    """
    Deprecated: use :class:`pgmpy.causal_discovery.LLMPairwise` instead.

    Asks a Large Language Model (LLM) for the orientation of an edge between
    `x` and `y`. Refer to `pgmpy.causal_discovery.LLMPairwise` for details on
    the parameters.

    Returns
    -------
    tuple or None
        The suggested edge direction `(source, target)`, or None if the LLM
        response could not be parsed.
    """
    warnings.warn(
        """`llm_pairwise_orient` is deprecated and will be removed in v2.0. Please use
        pgmpy.causal_discovery.LLMPairwise instead.""",
        FutureWarning,
        stacklevel=2,
    )
    est = _LLMPairwise(
        descriptions=descriptions,
        system_prompt=system_prompt,
        llm_model=llm_model,
        llm_kwargs=kwargs or None,
    )
    return est._parse_response(est._query_llm(est._build_prompt(x, y)), x, y)


class _LegacyCITestAdapter:
    """
    Adapts a legacy CI-test callable `(X, Y, Z, data=..., boolean=False, ...)`
    to the canonical `BaseCITest` calling convention (`run_test`,
    `is_independent`).
    """

    def __init__(self, fn, data):
        self.fn = fn
        self.data = data

    def run_test(self, X, Y, Z):
        result = self.fn(X, Y, Z, data=self.data, boolean=False)
        self.statistic_, self.p_value_ = result[0], result[1]
        return self.statistic_, self.p_value_

    def is_independent(self, X, Y, Z=(), significance_level=0.05):
        self.run_test(X, Y, list(Z))
        return bool(self.p_value_ >= significance_level)


def _resolve_legacy_ci_test(ci_test, data):
    """Resolve names/instances canonically; wrap legacy callables."""
    resolved = get_ci_test(test=ci_test, data=data)
    if callable(resolved) and not isinstance(resolved, BaseCITest):
        return _LegacyCITestAdapter(resolved, data)
    return resolved


class _LegacyOrientationEstimator:
    """
    Adapts a legacy `orientation_fn` callable (plus preferred `orientations`
    and an optional shared cache) to the canonical pairwise-estimator protocol
    (`fit(X[[u, v]])` populating `causal_graph_`).
    """

    def __init__(self, orientation_fn, orientations, fn_kwargs, cache):
        self.orientation_fn = orientation_fn
        self.orientations = set(orientations)
        self.fn_kwargs = dict(fn_kwargs)
        self.cache = cache

    def fit(self, X):
        x, y = list(X.columns)
        if (x, y) in self.orientations:
            direction = (x, y)
        elif (y, x) in self.orientations:
            direction = (y, x)
        elif self.cache is not None and (x, y) in self.cache:
            direction = (x, y)
        elif self.cache is not None and (y, x) in self.cache:
            direction = (y, x)
        else:
            direction = self.orientation_fn(x, y, **self.fn_kwargs)
            if self.cache is not None and direction is not None:
                self.cache.add(tuple(direction))

        self.causal_graph_ = DAG()
        self.causal_graph_.add_nodes_from([x, y])
        if direction is not None:
            self.causal_graph_.add_edge(*direction)
        return self


class ExpertInLoop(StructureEstimator):
    """
    Deprecated: use :class:`pgmpy.causal_discovery.ExpertInLoop` instead.

    Estimates a DAG from the data by iterating between a CI-test driven
    search and pairwise orientation queries (e.g. to an LLM or a human
    expert). This class delegates to the canonical implementation in
    `pgmpy.causal_discovery`.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
    """

    def __init__(self, data: pd.DataFrame | None = None, **kwargs):
        warnings.warn(
            """ExpertInLoop is deprecated and will be removed in v2.0. Please use pgmpy.causal_discovery.ExpertInLoop
            instead.""",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(data, **kwargs)
        self.orientation_cache = set()

    def test_all(self, ci_test, dag: DAG) -> pd.DataFrame:
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
        return _ExpertInLoop()._test_all(
            ci_test=_resolve_legacy_ci_test(ci_test, self.data),
            dag=dag,
            data=self.data,
        )

    def estimate(
        self,
        pval_threshold: float = 0.05,
        effect_size_threshold: float = 0.05,
        ci_test: str | None = None,
        orientation_fn: Callable[..., tuple[Hashable, Hashable] | None] = llm_pairwise_orient,
        orientations: set[tuple[str, str]] = set(),
        expert_knowledge: ExpertKnowledge | None = None,
        use_cache: bool = True,
        show_progress: bool = True,
        **kwargs,
    ) -> DAG:
        """
        Estimates a DAG from the data by delegating to
        :class:`pgmpy.causal_discovery.ExpertInLoop`; refer to its
        documentation for details.

        The legacy `orientation_fn` callable (with any additional `kwargs`),
        the preferred `orientations` set, and the `use_cache` flag are adapted
        onto the canonical `pairwise_estimator` interface.

        Returns
        -------
        pgmpy.base.DAG: A DAG representing the learned causal structure.
        """
        pairwise_estimator = _LegacyOrientationEstimator(
            orientation_fn=orientation_fn,
            orientations=orientations,
            fn_kwargs=kwargs,
            cache=self.orientation_cache if use_cache else None,
        )
        est = _ExpertInLoop(
            pval_threshold=pval_threshold,
            effect_size_threshold=effect_size_threshold,
            ci_test=ci_test,
            pairwise_estimator=pairwise_estimator,
            expert_knowledge=expert_knowledge,
            show_progress=show_progress,
        )
        return est.fit(self.data).causal_graph_

    def _break_cycle(self, dag, u, v, ci_test, effect_size_threshold, pval_threshold):
        """Legacy forwarder to the canonical cycle breaker."""
        return _ExpertInLoop()._break_cycle(
            dag,
            u,
            v,
            ci_test=_resolve_legacy_ci_test(ci_test, self.data),
            data=self.data,
            effect_size_threshold=effect_size_threshold,
            pval_threshold=pval_threshold,
        )
