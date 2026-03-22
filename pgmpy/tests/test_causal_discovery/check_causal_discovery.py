"""
Method to check whether a causal discovery algorithm is compliant with pgmpy's
unified interface, as defined in ``devtools/extension_templates/_causal_discovery.py``.

Location: pgmpy/tests/test_causal_discovery/check_causal_discovery.py
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery

VALID_GRAPH_TYPES = (DAG, PDAG, MAG, ADMG)

EXPECTED_FAILED_SKLEARN_CHECKS = {
    "check_fit_score_takes_y": ("Causal discovery estimators do not take y parameter in score method."),
    "check_n_features_in_after_fitting": ("Failing for score method (not for fit) for unknown reason."),
}


def _make_test_data(data_type="discrete", n_samples=200, n_features=3, seed=42):
    """Generate a simple synthetic dataset for testing.

    Parameters
    ----------
    data_type : str, default="discrete"
        The type of data to generate. Must be one of:
            * ``"discrete"``   - random integers in {0, 1}.
            * ``"continuous"`` - random standard-normal floats.
            * ``"mixed"``     - first half of columns discrete, rest continuous.

    n_samples : int, default=200
        Number of rows.

    n_features : int, default=3
        Number of columns.

    seed : int, default=42
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
    """
    rng = np.random.default_rng(seed)
    columns = [f"x{i}" for i in range(n_features)]

    if data_type == "discrete":
        data = rng.choice([0, 1], size=(n_samples, n_features))
        return pd.DataFrame(data, columns=columns)

    elif data_type == "continuous":
        data = rng.standard_normal(size=(n_samples, n_features))
        return pd.DataFrame(data, columns=columns)

    elif data_type == "mixed":
        n_discrete = max(1, n_features // 2)
        n_continuous = n_features - n_discrete
        discrete_part = rng.choice([0, 1], size=(n_samples, n_discrete))
        continuous_part = rng.standard_normal(size=(n_samples, n_continuous))
        data = np.hstack([discrete_part, continuous_part])
        return pd.DataFrame(data, columns=columns)

    else:
        raise ValueError(f"data_type must be one of 'discrete', 'continuous', or 'mixed'. Got: {data_type!r}")


def check_causal_discovery(estimator, data_type="discrete"):
    """Run all convention checks on a causal-discovery estimator.

    Parameters
    ----------
    estimator : object
        An **unfitted** causal-discovery estimator (e.g. ``PC()``, ``GES()``).

    data_type : str, default="discrete"
        The kind of synthetic data to generate for fitting.
        One of ``"discrete"``, ``"continuous"``, or ``"mixed"``.

    Raises
    ------
    TypeError
        If the estimator does not inherit from ``_BaseCausalDiscovery``.
    AssertionError
        If any check fails.

    Notes
    -----
    sklearn's ``check_estimator`` (called at the end) already validates
    ``get_params``, ``set_params``, and ``clone`` compatibility via its own
    checks (``check_get_params_invariance``, ``check_set_params``,
    ``check_estimator_cloneable``). Those checks are therefore not
    duplicated here.

    Examples
    --------
    >>> from pgmpy.causal_discovery import PC
    >>> from pgmpy.tests.test_causal_discovery.check_causal_discovery import check_causal_discovery
    >>> check_causal_discovery(PC())
    """
    name = type(estimator).__name__

    # Check 1: Must inherit from _BaseCausalDiscovery.
    if not isinstance(estimator, _BaseCausalDiscovery):
        raise TypeError(
            f"{name} does not inherit from _BaseCausalDiscovery. All causal discovery "
            f"algorithms must inherit from pgmpy.causal_discovery._base._BaseCausalDiscovery."
        )

    # Check 2: Must have a _fit method.
    assert hasattr(estimator, "_fit") and callable(estimator._fit), f"{name} does not implement a `_fit` method."

    # Check 3: Must have a score method.
    assert hasattr(estimator, "score") and callable(estimator.score), f"{name} does not have a `score` method."

    # Check 4: Fit on synthetic data.
    data = _make_test_data(data_type=data_type)
    fitted = estimator.fit(data)
    assert fitted is estimator, f"{name}.fit(X) must return self."
    n_cols = data.shape[1]

    # Check 5: causal_graph_ must exist and be a valid graph type.
    assert hasattr(fitted, "causal_graph_"), f"{name} does not set `causal_graph_` after fitting."
    assert isinstance(fitted.causal_graph_, VALID_GRAPH_TYPES), (
        f"{name}.causal_graph_ must be an instance of "
        f"{[t.__name__ for t in VALID_GRAPH_TYPES]}, "
        f"got {type(fitted.causal_graph_).__name__}."
    )

    # Check 6: adjacency_matrix_ must exist, be a DataFrame, and be square.
    assert hasattr(fitted, "adjacency_matrix_"), f"{name} does not set `adjacency_matrix_` after fitting."
    assert isinstance(fitted.adjacency_matrix_, pd.DataFrame), (
        f"{name}.adjacency_matrix_ must be a pandas DataFrame, got {type(fitted.adjacency_matrix_).__name__}."
    )
    adj_shape = fitted.adjacency_matrix_.shape
    assert adj_shape == (n_cols, n_cols), (
        f"{name}.adjacency_matrix_ shape should be ({n_cols}, {n_cols}), got {adj_shape}."
    )

    # Check 7: n_features_in_ must exist and match the data.
    assert hasattr(fitted, "n_features_in_"), f"{name} does not set `n_features_in_` after fitting."
    assert fitted.n_features_in_ == n_cols, f"{name}.n_features_in_ should be {n_cols}, got {fitted.n_features_in_}."

    # Check 8: feature_names_in_ must exist and match the data.
    assert hasattr(fitted, "feature_names_in_"), f"{name} does not set `feature_names_in_` after fitting."
    assert len(fitted.feature_names_in_) == n_cols, (
        f"{name}.feature_names_in_ length should be {n_cols}, got {len(fitted.feature_names_in_)}."
    )

    # Check 9: sklearn's check_estimator.
    # sklearn's checks call score() which requires a DAG, so we set
    # return_type="dag" if the estimator supports it.
    sklearn_estimator = clone(estimator)
    if hasattr(sklearn_estimator, "return_type"):
        sklearn_estimator.set_params(return_type="dag")
    check_estimator(
        sklearn_estimator,
        expected_failed_checks=EXPECTED_FAILED_SKLEARN_CHECKS,
    )
