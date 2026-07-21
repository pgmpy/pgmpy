"""Deprecated compatibility shim for :class:`pgmpy.causal_discovery.GES`."""

import warnings
from collections.abc import Iterable
from typing import Any

import pandas as pd

from pgmpy.base import PDAG
from pgmpy.causal_discovery import GES as _GES
from pgmpy.estimators import StructureEstimator
from pgmpy.structure_score import BaseStructureScore


class GES(StructureEstimator):
    """
    Deprecated: use :class:`pgmpy.causal_discovery.GES` instead.

    Implementation of the Greedy Equivalence Search (GES) causal discovery /
    structure learning algorithm. This class delegates to the canonical
    implementation in `pgmpy.causal_discovery`.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.

    use_cache: bool (default: False)
        Retained for backwards compatibility; canonical structure scores cache
        local scores internally.
    """

    def __init__(self, data: pd.DataFrame, use_cache: bool = False, **kwargs):
        warnings.warn(
            "GES is deprecated and will be removed in v2.0. Please use pgmpy.causal_discovery.GES instead.",
            FutureWarning,
            stacklevel=2,
        )
        super().__init__(data, **kwargs)

    def _legal_edge_deletions(self, model: PDAG):
        """Legacy forwarder to the canonical implementation."""
        return _GES()._legal_edge_deletions(model)

    def insert(self, u: Any, v: Any, T: Iterable[Any], current_model: PDAG):
        """
        Perform insert(u -> v) with conditioning set T. Delegates to
        :meth:`pgmpy.causal_discovery.GES.insert`.
        """
        return _GES().insert(u, v, T, current_model)

    def delete(self, u: Any, v: Any, H: set[Any], current_model: PDAG):
        """
        Perform delete(u - v) or delete(u -> v) with conditioning set H.
        Delegates to :meth:`pgmpy.causal_discovery.GES.delete`.
        """
        return _GES().delete(u, v, H, current_model)

    def turn(self, u: Any, v: Any, C: Iterable[Any], current_model: PDAG):
        """
        Perform turn operation (reverse or orient edge between u and v) with
        set C. Delegates to :meth:`pgmpy.causal_discovery.GES.turn`.
        """
        return _GES().turn(u, v, C, current_model)

    def estimate(
        self,
        scoring_method: str | BaseStructureScore | None = None,
        min_improvement: float = 1e-6,
        debug: bool = False,
    ) -> PDAG:
        """
        Estimates the PDAG from the data by delegating to
        :class:`pgmpy.causal_discovery.GES`.

        Parameters
        ----------
        scoring_method: str or BaseStructureScore instance
            The score to be optimized during structure estimation.  Supported
            structure scores: k2, bdeu, bds, bic-d, aic-d, ll-g, aic-g, bic-g,
            ll-cg, aic-cg, bic-cg. Also accepts a custom score, but it should
            be an instance of `BaseStructureScore`.

        min_improvement: float
            The operation (edge addition, removal, or turning) would only be performed if the
            model score improves by atleast `min_improvement`.

        debug: bool
            Retained for backwards compatibility; ignored by the canonical
            implementation.

        Returns
        -------
        Estimated model: pgmpy.base.PDAG
            A `PDAG` at a (local) score maximum.
        """
        est = _GES(
            scoring_method=scoring_method,
            return_type="pdag",
            min_improvement=min_improvement,
        )
        return est.fit(self.data).causal_graph_
