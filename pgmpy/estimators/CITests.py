"""Deprecated compatibility shims for :mod:`pgmpy.ci_tests`.

The CI test functions in this module are deprecated and will be removed in
v2.0. Each function is a thin wrapper delegating to its canonical class in
:mod:`pgmpy.ci_tests`; no test logic lives in this module anymore.

Legacy call convention: ``test(X, Y, Z, data, boolean=True, **kwargs)``. With
``boolean=True`` the functions return a bool (requiring a ``significance_level``
kwarg); with ``boolean=False`` they return the raw test results (see each
function).
"""

import warnings
from collections.abc import Callable

import pandas as pd

import pgmpy.ci_tests as _ci
from pgmpy.utils import get_dataset_type


class CITestRegistry:
    """
    Registry to manage Conditional Independence (CI) Test Strategies.

    Allows looking up tests by name or inferring suitable tests based on data type.
    """

    def __init__(self):
        self._registry: dict[str, Callable] = {}
        self._tags: dict[str, list[str]] = {}
        self._defaults: dict[str, str] = {
            "continuous": "pearsonr",
            "discrete": "chi_square",
            "mixed": "pillai",
        }

    def register(self, name: str, data_types: list[str]):
        """
        Decorator to register a CI test strategy.

        Parameters
        ----------
        name : str
            The name of the test (case-insensitive).

        data_types : list of str
            List of data types this test supports (e.g., ['continuous', 'discrete']).
        """

        def decorator(func: Callable):
            clean_name = name.lower()
            self._registry[clean_name] = func
            self._tags[clean_name] = data_types

            return func

        return decorator

    def list_all(self, data_type=None) -> list[str]:
        """
        Lists all registered CI test strategies.

        Parameters
        ----------
        data_type : str, optional
            If provided, filters tests that support the given data type.

        Returns
        -------
        list of str
            Names of all registered CI tests.
        """
        if data_type:
            return [name for name, types in self._tags.items() if data_type in types]

        return list(self._registry.keys())

    def get_test(self, test: str | None | Callable, data: pd.DataFrame | None = None) -> Callable:
        """
        Retrieves a CI test strategy.

        Parameters
        ----------
        test : str, callable or None
            The name of the test, a callable function, or None.

        data : pandas.DataFrame, optional
            The dataframe used to infer the test type if `test` is None.

        Returns
        -------
        callable
            The CI test function.

        Raises
        ------
        ValueError
            If `test` is None and `data` is None, or if the test name is not found.
        """
        # Case 1: Test is already a function/strategy
        if callable(test):
            return test

        # Case 2: Test is None, infer from data
        if test is None:
            if data is None:
                raise ValueError("Cannot determine a suitable CI test as data is None. Please specify CI test to use.")
            var_type = get_dataset_type(data)
            test_name = self._defaults.get(var_type)
            return self._registry[test_name]

        # Case 3: Test is a string name
        if isinstance(test, str):
            clean_name = test.lower()
            if clean_name in self._registry:
                return self._registry[clean_name]
            else:
                raise ValueError(
                    f"`ci_test` must either be one of {list(self._registry.keys())}, or a callable. Got: {test}"
                )


ci_registry = CITestRegistry()


def _warn_deprecated(old_name: str, new_name: str) -> None:
    warnings.warn(
        f"`{old_name}` is deprecated and will be removed in v2.0. Please use `pgmpy.ci_tests.{new_name}` instead.",
        FutureWarning,
        stacklevel=3,
    )


@ci_registry.register(
    name="independence_match",
    data_types=["discrete", "continuous", "mixed"],
)
def independence_match(X, Y, Z, independencies, **kwargs):
    """
    Deprecated: use :class:`pgmpy.ci_tests.IndependenceMatch` instead.

    Check if `X ⟂ Y | Z` is in `independencies`.

    Returns
    -------
    bool
        True if the independence assertion is present in `independencies`, else False.
    """
    _warn_deprecated("independence_match", "IndependenceMatch")
    return _ci.IndependenceMatch(independencies=independencies, use_cache=False).is_independent(X, Y, Z)


@ci_registry.register(name="pearsonr", data_types=["continuous"])
def pearsonr(X, Y, Z, data, boolean=True, **kwargs):
    """
    Deprecated: use :class:`pgmpy.ci_tests.Pearsonr` instead.

    (Partial) correlation test for continuous data.

    Returns
    -------
    bool or (coef, p_value)
        If boolean=True, returns True if p_value >= `significance_level`
        (required kwarg), else False. Otherwise returns the correlation
        coefficient and the p-value.
    """
    _warn_deprecated("pearsonr", "Pearsonr")
    test = _ci.Pearsonr(data, use_cache=False)
    if boolean:
        return test.is_independent(X, Y, Z, significance_level=kwargs["significance_level"])
    return test.run_test(X, Y, Z)


@ci_registry.register(name="power_divergence", data_types=["discrete"])
def power_divergence(X, Y, Z, data, boolean=True, lambda_="cressie-read", **kwargs):
    """
    Deprecated: use :class:`pgmpy.ci_tests.PowerDivergence` instead.

    Power divergence family of CI tests for discrete data.

    Returns
    -------
    bool or (statistic, p_value, dof)
        If boolean=True, returns True if p_value >= `significance_level`
        (required kwarg), else False. Otherwise returns the test statistic,
        the p-value, and the degrees of freedom.
    """
    _warn_deprecated("power_divergence", "PowerDivergence")
    test = _ci.PowerDivergence(data, lambda_=lambda_, use_cache=False)
    if boolean:
        return test.is_independent(X, Y, Z, significance_level=kwargs["significance_level"])
    statistic, p_value = test.run_test(X, Y, Z)
    return statistic, p_value, test.dof_


def _power_divergence_shim(new_cls, X, Y, Z, data, boolean, **kwargs):
    test = new_cls(data, use_cache=False)
    if boolean:
        return test.is_independent(X, Y, Z, significance_level=kwargs["significance_level"])
    statistic, p_value = test.run_test(X, Y, Z)
    return statistic, p_value, test.dof_


@ci_registry.register(name="chi_square", data_types=["discrete"])
def chi_square(X, Y, Z, data, boolean=True, **kwargs):
    """
    Deprecated: use :class:`pgmpy.ci_tests.ChiSquare` instead.

    Chi-square CI test for discrete data. Same return contract as
    :func:`power_divergence`.
    """
    _warn_deprecated("chi_square", "ChiSquare")
    return _power_divergence_shim(_ci.ChiSquare, X, Y, Z, data, boolean, **kwargs)


@ci_registry.register(name="g_sq", data_types=["discrete"])
def g_sq(X, Y, Z, data, boolean=True, **kwargs):
    """
    Deprecated: use :class:`pgmpy.ci_tests.GSq` instead.

    G-squared CI test for discrete data. Same return contract as
    :func:`power_divergence`.
    """
    _warn_deprecated("g_sq", "GSq")
    return _power_divergence_shim(_ci.GSq, X, Y, Z, data, boolean, **kwargs)


@ci_registry.register(name="log_likelihood", data_types=["discrete"])
def log_likelihood(X, Y, Z, data, boolean=True, **kwargs):
    """
    Deprecated: use :class:`pgmpy.ci_tests.LogLikelihood` instead.

    Log-likelihood ratio CI test for discrete data. Same return contract as
    :func:`power_divergence`.
    """
    _warn_deprecated("log_likelihood", "LogLikelihood")
    return _power_divergence_shim(_ci.LogLikelihood, X, Y, Z, data, boolean, **kwargs)


@ci_registry.register(name="modified_log_likelihood", data_types=["discrete"])
def modified_log_likelihood(X, Y, Z, data, boolean=True, **kwargs):
    """
    Deprecated: use :class:`pgmpy.ci_tests.ModifiedLogLikelihood` instead.

    Modified log-likelihood CI test for discrete data. Same return contract as
    :func:`power_divergence`.
    """
    _warn_deprecated("modified_log_likelihood", "ModifiedLogLikelihood")
    return _power_divergence_shim(_ci.ModifiedLogLikelihood, X, Y, Z, data, boolean, **kwargs)


@ci_registry.register(name="pillai", data_types=["discrete", "continuous", "mixed"])
def pillai_trace(X, Y, Z, data, boolean=True, **kwargs):
    """
    Deprecated: use :class:`pgmpy.ci_tests.PillaiTrace` instead.

    Pillai's trace based residualization CI test for mixed data.

    Note: the canonical implementation residualizes with a RandomForest
    estimator by default (the legacy function used XGBoost); pass
    ``estimator=`` for a custom sklearn-compatible estimator.

    Returns
    -------
    bool or (coef, p_value)
        If boolean=True, returns True if p_value >= `significance_level`
        (required kwarg), else False. Otherwise returns the canonical
        correlation coefficient and the p-value.
    """
    _warn_deprecated("pillai_trace", "PillaiTrace")
    test = _ci.PillaiTrace(data, estimator=kwargs.get("estimator"), use_cache=False)
    if boolean:
        return test.is_independent(X, Y, Z, significance_level=kwargs["significance_level"])
    return test.run_test(X, Y, Z)


@ci_registry.register(name="gcm", data_types=["continuous"])
def gcm(X, Y, Z, data, boolean=True, **kwargs):
    """
    Deprecated: use :class:`pgmpy.ci_tests.GCM` instead.

    Generalised Covariance Measure CI test.

    Note: the canonical implementation residualizes with a RandomForest
    estimator by default (the legacy function used XGBoost); pass
    ``estimator=`` for a custom sklearn-compatible estimator.

    Returns
    -------
    bool or (statistic, p_value)
        If boolean=True, returns True if p_value >= `significance_level`
        (required kwarg), else False. Otherwise returns the test statistic
        and the p-value.
    """
    _warn_deprecated("gcm", "GCM")
    test = _ci.GCM(data, estimator=kwargs.get("estimator"), use_cache=False)
    if boolean:
        return test.is_independent(X, Y, Z, significance_level=kwargs["significance_level"])
    return test.run_test(X, Y, Z)


@ci_registry.register(name="pearsonr_equivalence", data_types=["continuous"])
def pearsonr_equivalence(X, Y, Z, data, boolean=True, delta_threshold=0.1, **kwargs) -> tuple | bool:
    """
    Deprecated: use :class:`pgmpy.ci_tests.PearsonrEquivalence` instead.

    Equivalence (TOST) variant of the pearsonr CI test. Note the inverted
    boolean semantics: with boolean=True, returns True (independent) if
    p_value < `significance_level` (default 0.05).

    Returns
    -------
    bool or (coef, p_value)
        If boolean=True, returns True if p_value < `significance_level`,
        else False. Otherwise returns the partial correlation coefficient
        and the TOST p-value.
    """
    _warn_deprecated("pearsonr_equivalence", "PearsonrEquivalence")
    test = _ci.PearsonrEquivalence(data, delta_threshold=delta_threshold, use_cache=False)
    if boolean:
        return test.is_independent(X, Y, Z, significance_level=kwargs.get("significance_level", 0.05))
    return test.run_test(X, Y, Z)
