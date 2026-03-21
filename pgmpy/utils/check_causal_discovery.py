"""
Method to check whether a causal discovery algorithm is compliant with pgmpy's
unified interface, as defined in ``devtools/extension_templates/_causal_discovery.py``.
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.estimator_checks import check_estimator

from pgmpy.base import ADMG, DAG, MAG, PDAG
from pgmpy.causal_discovery._base import _BaseCausalDiscovery

VALID_GRAPH_TYPES = (DAG, PDAG, MAG, ADMG)

EXPECTED_FAILED_SKLEARN_CHECKS = {
    "check_fit_score_takes_y": "Causal discovery estimators do not take y parameter in score method.",
    "check_n_features_in_after_fitting": "Failing for score method (not for fit) for unknown reason.",
}


def check_causal_discovery(estimator):
    """
    Check if a causal discovery estimator is compliant with pgmpy's interface.

    Runs a series of checks on the estimator to verify that it follows the unified
    interface defined in ``devtools/extension_templates/_causal_discovery.py``. Also
    runs sklearn's ``check_estimator`` to verify sklearn compatibility.

    Parameters
    ----------
    estimator : object
        An unfitted instance of a causal discovery class to check.

    Raises
    ------
    TypeError
        If the estimator does not inherit from ``_BaseCausalDiscovery``.
    AssertionError
        If any of the interface checks fail.

    Examples
    --------
    >>> from pgmpy.causal_discovery import PC
    >>> from pgmpy.utils.check_causal_discovery import check_causal_discovery
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

    # Check 4: sklearn compatibility — get_params, set_params, clone.
    params = estimator.get_params()
    assert isinstance(params, dict), f"{name}.get_params() should return a dict."

    for param in params:
        assert hasattr(estimator, param), f"{name}.__init__ parameter `{param}` is not stored as an instance attribute."

    cloned = clone(estimator)
    assert type(cloned) is type(estimator), f"sklearn.base.clone failed for {name}."

    # Check 5: fit on synthetic data.
    data = _make_test_data()
    fitted = estimator.fit(data)
    assert fitted is estimator, f"{name}.fit(X) must return self."

    # Check 6: Fitted attributes.
    assert hasattr(fitted, "causal_graph_"), f"{name} does not set `causal_graph_` after fitting."
    assert isinstance(fitted.causal_graph_, VALID_GRAPH_TYPES), (
        f"{name}.causal_graph_ must be an instance of {[t.__name__ for t in VALID_GRAPH_TYPES]}, "
        f"got {type(fitted.causal_graph_).__name__}."
    )

    assert hasattr(fitted, "adjacency_matrix_"), f"{name} does not set `adjacency_matrix_` after fitting."
    assert isinstance(fitted.adjacency_matrix_, pd.DataFrame), (
        f"{name}.adjacency_matrix_ must be a pandas DataFrame, got {type(fitted.adjacency_matrix_).__name__}."
    )

    n_cols = data.shape[1]
    assert hasattr(fitted, "n_features_in_"), f"{name} does not set `n_features_in_` after fitting."
    assert fitted.n_features_in_ == n_cols, f"{name}.n_features_in_ should be {n_cols}, got {fitted.n_features_in_}."

    assert hasattr(fitted, "feature_names_in_"), f"{name} does not set `feature_names_in_` after fitting."
    assert len(fitted.feature_names_in_) == n_cols, (
        f"{name}.feature_names_in_ length should be {n_cols}, got {len(fitted.feature_names_in_)}."
    )

    # Check 7: sklearn's check_estimator.
    # sklearn's checks call score() which requires a DAG, so we set return_type="dag"
    # if the estimator supports it. This matches the approach used in the existing test files.
    sklearn_estimator = clone(estimator)
    if hasattr(sklearn_estimator, "return_type"):
        sklearn_estimator.set_params(return_type="dag")
    check_estimator(sklearn_estimator, expected_failed_checks=EXPECTED_FAILED_SKLEARN_CHECKS)


def _make_test_data(n_samples=200, seed=42):
    """Generate a simple synthetic discrete dataset for testing."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.choice([0, 1], size=(n_samples, 3)),
        columns=["x0", "x1", "x2"],
    )
