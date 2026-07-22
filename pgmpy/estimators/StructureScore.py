"""Deprecated compatibility shims for :mod:`pgmpy.structure_score`.

This module is deprecated and will be removed in v2.0. Every class defined
here is a thin wrapper that delegates to its canonical implementation in
:mod:`pgmpy.structure_score`; no scoring logic lives in this module anymore.

Notes
-----
- ``StructureScore`` (the legacy base class) is an alias of
  :class:`pgmpy.structure_score.BaseStructureScore`, so ``isinstance`` checks
  against either name work for both legacy and new-style score instances.
"""

from __future__ import annotations

import warnings

import pandas as pd

from pgmpy.structure_score import AIC as _AIC
from pgmpy.structure_score import BIC as _BIC
from pgmpy.structure_score import K2 as _K2
from pgmpy.structure_score import AICCondGauss as _AICCondGauss
from pgmpy.structure_score import AICGauss as _AICGauss
from pgmpy.structure_score import BaseStructureScore
from pgmpy.structure_score import BDeu as _BDeu
from pgmpy.structure_score import BDs as _BDs
from pgmpy.structure_score import BICCondGauss as _BICCondGauss
from pgmpy.structure_score import BICGauss as _BICGauss
from pgmpy.structure_score import LogLikelihood as _LogLikelihood
from pgmpy.structure_score import LogLikelihoodCondGauss as _LogLikelihoodCondGauss
from pgmpy.structure_score import LogLikelihoodGauss as _LogLikelihoodGauss
from pgmpy.structure_score import get_scoring_method as _get_scoring_method

# Legacy base class. Aliased to the canonical base so that isinstance checks
# against `pgmpy.estimators.StructureScore` keep working for legacy shims,
# new-style scores, and user-defined subclasses alike.
StructureScore = BaseStructureScore


def _warn_deprecated(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"`pgmpy.estimators.{old_name}` is deprecated and will be removed in v2.0. "
        f"Use `pgmpy.structure_score.{new_name}` instead.",
        FutureWarning,
        stacklevel=3,
    )


class K2(_K2):
    """Deprecated: use :class:`pgmpy.structure_score.K2` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("K2", "K2")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


class BDeu(_BDeu):
    """Deprecated: use :class:`pgmpy.structure_score.BDeu` instead."""

    def __init__(
        self,
        data: pd.DataFrame,
        equivalent_sample_size: int = 10,
        state_names: dict | None = None,
        max_cache_size: int | None = 10000,
    ):
        _warn_deprecated("BDeu", "BDeu")
        super().__init__(
            data,
            equivalent_sample_size=equivalent_sample_size,
            state_names=state_names,
            max_cache_size=max_cache_size,
        )


class BDs(_BDs):
    """Deprecated: use :class:`pgmpy.structure_score.BDs` instead."""

    def __init__(
        self,
        data: pd.DataFrame,
        equivalent_sample_size: int = 10,
        state_names: dict | None = None,
        max_cache_size: int | None = 10000,
    ):
        _warn_deprecated("BDs", "BDs")
        super().__init__(
            data,
            equivalent_sample_size=equivalent_sample_size,
            state_names=state_names,
            max_cache_size=max_cache_size,
        )


class LogLikelihood(_LogLikelihood):
    """Deprecated: use :class:`pgmpy.structure_score.LogLikelihood` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("LogLikeliHood", "LogLikelihood")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


# The legacy class name carried a typo; keep it importable until removal.
LogLikeliHood = LogLikelihood


class BIC(_BIC):
    """Deprecated: use :class:`pgmpy.structure_score.BIC` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("BIC", "BIC")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


class AIC(_AIC):
    """Deprecated: use :class:`pgmpy.structure_score.AIC` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("AIC", "AIC")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


class LogLikelihoodGauss(_LogLikelihoodGauss):
    """Deprecated: use :class:`pgmpy.structure_score.LogLikelihoodGauss` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("LogLikelihoodGauss", "LogLikelihoodGauss")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


class BICGauss(_BICGauss):
    """Deprecated: use :class:`pgmpy.structure_score.BICGauss` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("BICGauss", "BICGauss")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


class AICGauss(_AICGauss):
    """Deprecated: use :class:`pgmpy.structure_score.AICGauss` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("AICGauss", "AICGauss")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


class LogLikelihoodCondGauss(_LogLikelihoodCondGauss):
    """Deprecated: use :class:`pgmpy.structure_score.LogLikelihoodCondGauss` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("LogLikelihoodCondGauss", "LogLikelihoodCondGauss")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


class BICCondGauss(_BICCondGauss):
    """Deprecated: use :class:`pgmpy.structure_score.BICCondGauss` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("BICCondGauss", "BICCondGauss")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


class AICCondGauss(_AICCondGauss):
    """Deprecated: use :class:`pgmpy.structure_score.AICCondGauss` instead."""

    def __init__(self, data: pd.DataFrame, state_names: dict | None = None, max_cache_size: int | None = 10000):
        _warn_deprecated("AICCondGauss", "AICCondGauss")
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)


def get_scoring_method(
    scoring_method: str | BaseStructureScore | None,
    data: pd.DataFrame,
    use_cache: bool,
    **kwargs,
) -> tuple[BaseStructureScore, BaseStructureScore]:
    """Deprecated: use :func:`pgmpy.structure_score.get_scoring_method` instead.

    Legacy adapter preserving the old ``(score, cached_score)`` tuple contract.
    Canonical scores cache internally (see ``max_cache_size``), so both tuple
    elements refer to the same instance.
    """
    _warn_deprecated("get_scoring_method", "get_scoring_method")
    score = _get_scoring_method(scoring_method, data)
    if kwargs and isinstance(scoring_method, str):
        # The legacy API forwarded extra kwargs to the score constructor.
        score = type(score)(data, **kwargs)
    return score, score
