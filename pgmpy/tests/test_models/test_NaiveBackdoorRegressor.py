"""
Tests for Naive Backdoor Regressor.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.models.NaiveBackdoorRegressor import NaiveBackdoorRegressor


def make_estimator():
    """Create a valid estimator for sklearn compatibility tests."""

    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": "Z"},
    )

    return NaiveBackdoorRegressor(dag=dag, base_estimator=DummyRegressor())


@parametrize_with_checks([make_estimator()])
def test_sklearn_compatibility(estimator, check):
    """Test sklearn compatibility using parametrize_with_checks."""
    check(estimator)


def test_basic_functionality_with_adjustment():
    """Test basic fit and predict functionality with synthetic causal data."""

    # Synthetic causal data: Z -> X, Z -> Y, X -> Y
    np.random.seed(42)
    n_samples = 1000

    # Generate confounder Z
    Z = np.random.normal(0, 1, n_samples)

    # Generate exposure X influenced by Z
    X = 0.5 * Z + np.random.normal(0, 0.5, n_samples)

    # Generate outcome Y influenced by both X and Z (confounded relationship)
    Y = 2.0 * X + 1.5 * Z + np.random.normal(0, 0.3, n_samples)

    data = pd.DataFrame(
        {
            "X": X,  # exposure
            "Y": Y,  # note: outcome (this will be y in fit, not used in X)
            "Z": Z,  # adjustment/confounder
        }
    )

    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": "Z"},
    )

    # Test with different base estimators
    estimators = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=10, random_state=42),
        DummyRegressor(),
    ]

    for base_est in estimators:
        regressor = NaiveBackdoorRegressor(dag=dag, base_estimator=base_est)

        # Split data
        train_size = int(0.7 * n_samples)
        # Features should include exposure + adjustment variables
        X_train = data[["X", "Z"]].iloc[:train_size]
        y_train = data["Y"].iloc[:train_size]  # outcome
        X_test = data[["X", "Z"]].iloc[train_size:]
        y_test = data["Y"].iloc[train_size:]

        # Fit and predict
        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_test)

        # Basic validation
        assert len(predictions) == len(y_test)
        assert regressor.exposure_var_ == "X"
        assert regressor.outcome_var_ == "Y"
        assert regressor.adjustment_vars_ == ["Z"]

        # Check feature names
        feature_names = regressor.get_feature_names_out()
        expected_features = ["X", "Z"]  # exposure + adjustment
        assert list(feature_names) == expected_features


def test_no_adjustment_variables():
    """Test case where there are no adjustment variables (no confounders)."""

    dag = DAG(ebunch=[("X", "Y")], roles={"exposure": "X", "outcome": "Y"})

    # Generate simple causal data without confounders
    np.random.seed(42)
    n_samples = 100
    X = np.random.normal(0, 1, n_samples)
    Y = 2.0 * X + np.random.normal(0, 0.3, n_samples)

    data = pd.DataFrame({"X": X, "Y": Y})

    regressor = NaiveBackdoorRegressor(dag=dag)
    regressor.fit(data[["X"]], data["Y"])
    predictions = regressor.predict(data[["X"]])

    assert len(predictions) == n_samples
    assert regressor.adjustment_vars_ == []
    assert list(regressor.get_feature_names_out()) == ["X"]
    assert isinstance(regressor.estimator_, LinearRegression)  # default estimator


def test_multiple_adjustment_variables():
    """Test with multiple adjustment variables."""

    # DAG with multiple confounders: U1 -> X, U1 -> Y, U2 -> X, U2 -> Y, X -> Y
    dag = DAG(
        ebunch=[("U1", "X"), ("U1", "Y"), ("U2", "X"), ("U2", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["U1", "U2"]},
    )

    np.random.seed(42)
    n_samples = 200
    U1 = np.random.normal(0, 1, n_samples)
    U2 = np.random.normal(0, 1, n_samples)
    X = 0.3 * U1 + 0.4 * U2 + np.random.normal(0, 0.5, n_samples)
    Y = 1.5 * X + 0.6 * U1 + 0.7 * U2 + np.random.normal(0, 0.3, n_samples)

    data = pd.DataFrame({"X": X, "Y": Y, "U1": U1, "U2": U2})

    regressor = NaiveBackdoorRegressor(dag=dag, base_estimator=LinearRegression())
    regressor.fit(data[["X", "U1", "U2"]], data["Y"])
    predictions = regressor.predict(data[["X", "U1", "U2"]])

    assert len(predictions) == n_samples
    assert regressor.exposure_var_ == "X"
    assert set(regressor.adjustment_vars_) == {"U1", "U2"}
    assert set(regressor.get_feature_names_out()) == {"X", "U1", "U2"}


def test_error_handling():
    """Test various error conditions and validation."""

    # Test missing required roles
    dag_no_outcome = DAG(
        ebunch=[("X", "Y")], roles={"exposure": "X"}  # Missing outcome role
    )
    regressor = NaiveBackdoorRegressor(dag=dag_no_outcome)

    with pytest.raises(ValueError, match="no 'outcome' role was defined"):
        regressor.fit(pd.DataFrame({"X": [1, 2], "Y": [3, 4]}), [5, 6])

    # Test multiple exposure variables (should fail)
    dag_multi_exposure = DAG(
        ebunch=[("X1", "Y"), ("X2", "Y")],
        roles={"exposure": ["X1", "X2"], "outcome": "Y"},
    )
    regressor = NaiveBackdoorRegressor(dag=dag_multi_exposure)

    with pytest.raises(
        ValueError, match="Exactly one exposure variable must be defined"
    ):
        regressor.fit(pd.DataFrame({"X1": [1, 2], "X2": [3, 4], "Y": [5, 6]}), [7, 8])

    # Test missing required columns in data
    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": "Z"},
    )
    regressor = NaiveBackdoorRegressor(dag=dag)

    # Data missing required column Z
    incomplete_data = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})

    with pytest.raises(ValueError, match="Missing required columns"):
        regressor.fit(incomplete_data, [5, 6])


def test_numpy_array_input():
    """Test that regressor works with numpy array inputs."""

    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": "Z"},
    )

    regressor = NaiveBackdoorRegressor(dag=dag)

    # Create data as numpy arrays (columns should be in order: X, Z)
    np.random.seed(42)
    n_samples = 50
    X_array = np.random.normal(0, 1, (n_samples, 2))  # 2 features: X, Z
    y_array = np.random.normal(0, 1, n_samples)

    # This should work - regressor should map columns to required features
    regressor.fit(X_array, y_array)
    predictions = regressor.predict(X_array)

    assert len(predictions) == n_samples
    assert regressor.feature_columns_ == ["X", "Z"]


def test_sample_weight_support():
    """Test that sample_weight parameter is properly passed to base estimator."""

    dag = DAG(ebunch=[("X", "Y")], roles={"exposure": "X", "outcome": "Y"})

    # estimator that supports sample_weight
    regressor = NaiveBackdoorRegressor(dag=dag, base_estimator=LinearRegression())

    data = pd.DataFrame({"X": [1, 2, 3, 4], "Y": [2, 4, 6, 8]})
    sample_weights = np.array([1, 1, 2, 2])  # Give more weight to last two samples

    regressor.fit(data[["X"]], data["Y"], sample_weight=sample_weights)
    predictions = regressor.predict(data[["X"]])

    assert len(predictions) == len(data)


def test_dag_roles_validation():
    """Test that DAG roles are properly validated using pgmpy's built-in methods."""

    # Test valid causal structure
    dag_valid = DAG(ebunch=[("X", "Y")], roles={"exposure": "X", "outcome": "Y"})

    regressor = NaiveBackdoorRegressor(dag=dag_valid)

    # This should work without errors
    exposure, outcome, adjustment = regressor._validate_dag_and_extract_roles()
    assert exposure == "X"
    assert outcome == "Y"
    assert adjustment == []

    # Test that pgmpy's validation catches invalid structures
    dag_no_roles = DAG(ebunch=[("X", "Y")])  # No roles defined
    regressor_invalid = NaiveBackdoorRegressor(dag=dag_no_roles)

    with pytest.raises(ValueError):
        regressor_invalid._validate_dag_and_extract_roles()
