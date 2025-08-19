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
from pgmpy.prediction.NaiveBackdoorRegressor import NaiveBackdoorRegressor


def make_estimator():
    """Create a valid estimator for sklearn compatibility tests."""
    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    )
    return NaiveBackdoorRegressor(causal_graph=dag, base_estimator=LinearRegression())


@parametrize_with_checks([make_estimator()])
def test_sklearn_compatibility(estimator, check):
    """Test sklearn compatibility using parametrize_with_checks."""
    check(estimator)


def test_basic_functionality_with_adjustment():
    """Test basic fit and predict functionality with synthetic causal data."""

    # Create LinearGaussianBayesianNetwork using dagitty with specif coeff
    lgbn = DAG.from_dagitty(
        "dag { Z -> X [beta=0.5] X -> Y [beta=2.0] Z -> Y [beta=1.5] }"
    )

    # Generate synthetic data
    np.random.seed(42)
    data = lgbn.simulate(1000)

    # Create DAG with roles for the NaiveBackdoorRegressor
    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    )

    # Test with different base estimators
    estimators = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=10, random_state=42),
        DummyRegressor(),
    ]

    for base_est in estimators:
        regressor = NaiveBackdoorRegressor(causal_graph=dag, base_estimator=base_est)

        # Split data
        train_size = int(0.7 * len(data))
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

    # simple X -> Y relationship
    lgbn = DAG.from_dagitty("dag { X -> Y [beta=2.0] }")

    # Generate synthetic data
    np.random.seed(42)
    data = lgbn.simulate(100)

    dag = DAG(
        ebunch=[("X", "Y")],
        roles={
            "exposure": "X",
            "outcome": "Y",
            "adjustment": [],
        },  # Explicit empty adjustment
    )

    regressor = NaiveBackdoorRegressor(causal_graph=dag)
    regressor.fit(data[["X"]], data["Y"])
    predictions = regressor.predict(data[["X"]])

    assert len(predictions) == len(data)
    assert regressor.adjustment_vars_ == []
    assert list(regressor.get_feature_names_out()) == ["X"]
    assert isinstance(regressor.estimator_, LinearRegression)  # default estimator


def test_multiple_adjustment_variables():
    """Test with multiple adjustment variables."""

    # complex causal structure
    lgbn = DAG.from_dagitty(
        "dag { U1 -> X [beta=0.3] U1 -> Y [beta=0.6] U2 -> X [beta=0.4] U2 -> Y [beta=0.7] X -> Y [beta=1.5] }"
    )

    # Generate synthetic data
    np.random.seed(42)
    data = lgbn.simulate(200)

    # DAG with multiple confounders: U1 -> X, U1 -> Y, U2 -> X, U2 -> Y, X -> Y
    dag = DAG(
        ebunch=[("U1", "X"), ("U1", "Y"), ("U2", "X"), ("U2", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["U1", "U2"]},
    )

    regressor = NaiveBackdoorRegressor(
        causal_graph=dag, base_estimator=LinearRegression()
    )
    regressor.fit(data[["X", "U1", "U2"]], data["Y"])
    predictions = regressor.predict(data[["X", "U1", "U2"]])

    assert len(predictions) == len(data)
    assert regressor.exposure_var_ == "X"
    assert set(regressor.adjustment_vars_) == {"U1", "U2"}
    assert set(regressor.get_feature_names_out()) == {"X", "U1", "U2"}


def test_error_handling():
    """Test various error conditions and validation."""

    # Test missing required roles
    dag_no_outcome = DAG(
        ebunch=[("X", "Y")], roles={"exposure": "X"}  # Missing outcome role
    )
    regressor = NaiveBackdoorRegressor(causal_graph=dag_no_outcome)

    with pytest.raises(ValueError, match="no 'outcome' role was defined"):
        regressor.fit(pd.DataFrame({"X": [1, 2], "Y": [3, 4]}), [5, 6])

    # Test multiple exposure variables (should fail)
    dag_multi_exposure = DAG(
        ebunch=[("X1", "Y"), ("X2", "Y")],
        roles={"exposure": ["X1", "X2"], "outcome": "Y"},
    )
    regressor = NaiveBackdoorRegressor(causal_graph=dag_multi_exposure)

    with pytest.raises(
        ValueError, match="Exactly one exposure variable must be defined"
    ):
        regressor.fit(pd.DataFrame({"X1": [1, 2], "X2": [3, 4], "Y": [5, 6]}), [7, 8])

    # Test missing required columns in data
    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    )
    regressor = NaiveBackdoorRegressor(causal_graph=dag)

    # Data missing required column Z
    incomplete_data = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})

    with pytest.raises(ValueError, match="Missing required columns"):
        regressor.fit(incomplete_data, [5, 6])


def test_numpy_array_input():
    """Test that regressor works with numpy array inputs."""

    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    )

    regressor = NaiveBackdoorRegressor(causal_graph=dag)

    # Create data as numpy arrays (columns should be in order: X, Z)
    np.random.seed(42)
    n_samples = 50
    X_array = np.random.normal(0, 1, (n_samples, 2))  # 2 features: X, Z
    y_array = np.random.normal(0, 1, n_samples)

    # Now arrays require explicit feature names for safety
    regressor.fit(X_array, y_array, feature_names=["X", "Z"])
    predictions = regressor.predict(X_array, feature_names=["X", "Z"])

    assert len(predictions) == n_samples
    assert regressor.feature_columns_ == ["X", "Z"]


def test_sample_weight_support():
    """Test that sample_weight parameter is properly passed to base estimator."""

    # for realistic causal relationship
    lgbn = DAG.from_dagitty("dag { X -> Y [beta=2.0] }")

    # small dataset
    np.random.seed(42)
    data = lgbn.simulate(4)

    dag = DAG(
        ebunch=[("X", "Y")],
        roles={
            "exposure": "X",
            "outcome": "Y",
            "adjustment": [],
        },  # Explicit empty adjustment
    )

    # estimator that supports sample_weight
    regressor = NaiveBackdoorRegressor(
        causal_graph=dag, base_estimator=LinearRegression()
    )

    sample_weights = np.array([1, 1, 2, 2])  # Give more weight to last two samples

    regressor.fit(data[["X"]], data["Y"], sample_weight=sample_weights)
    predictions = regressor.predict(data[["X"]])

    assert len(predictions) == len(data)


def test_dag_roles_validation():
    """Test that DAG roles are properly validated using pgmpy's built-in methods."""

    # Test valid causal structure
    dag_valid = DAG(
        ebunch=[("X", "Y")], roles={"exposure": "X", "outcome": "Y", "adjustment": []}
    )

    regressor = NaiveBackdoorRegressor(causal_graph=dag_valid)

    # This should work without errors
    exposure, outcome, adjustment, pretreatment = (
        regressor._validate_dag_and_extract_roles()
    )
    assert exposure == "X"
    assert outcome == "Y"
    assert adjustment == []
    assert pretreatment == []

    # Test that pgmpy's validation catches invalid structures
    dag_no_roles = DAG(ebunch=[("X", "Y")])  # No roles defined
    regressor_invalid = NaiveBackdoorRegressor(causal_graph=dag_no_roles)

    with pytest.raises(ValueError):
        regressor_invalid._validate_dag_and_extract_roles()


def test_array_input_requires_feature_names():
    """Test that array input requires explicit feature names."""
    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    )

    regressor = NaiveBackdoorRegressor(causal_graph=dag)

    # Array input without feature_names should raise error
    X_array = np.random.normal(0, 1, (50, 2))
    y_array = np.random.normal(0, 1, 50)

    with pytest.raises(ValueError, match="must provide explicit feature names"):
        regressor.fit(X_array, y_array)

    # Should work with feature_names
    regressor.fit(X_array, y_array, feature_names=["X", "Z"])
    predictions = regressor.predict(X_array, feature_names=["X", "Z"])
    assert len(predictions) == 50


def test_adjustment_role_required():
    """Test that adjustment role must be explicitly defined."""
    # Missing adjustment role should raise error
    dag_no_adj = DAG(
        ebunch=[("X", "Y")],
        roles={"exposure": "X", "outcome": "Y"},  # Missing adjustment role
    )

    regressor = NaiveBackdoorRegressor(causal_graph=dag_no_adj)

    with pytest.raises(
        ValueError, match="adjustment.*role.*must be explicitly defined"
    ):
        regressor.fit(pd.DataFrame({"X": [1, 2], "Y": [3, 4]}), [5, 6])


def test_empty_adjustment_role_explicit():
    """Test that explicit empty adjustment role works correctly."""
    dag = DAG(
        ebunch=[("X", "Y")],
        roles={
            "exposure": "X",
            "outcome": "Y",
            "adjustment": [],
        },  # Explicit empty adjustment
    )

    regressor = NaiveBackdoorRegressor(causal_graph=dag)
    regressor.fit(pd.DataFrame({"X": [1, 2], "Y": [3, 4]}), [5, 6])
    assert regressor.adjustment_vars_ == []


def test_pretreatment_variables():
    """Test support for pretreatment variables."""
    dag = DAG(
        ebunch=[("P", "X"), ("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={
            "exposure": "X",
            "outcome": "Y",
            "adjustment": ["Z"],
            "pretreatment": ["P"],
        },
    )

    regressor = NaiveBackdoorRegressor(causal_graph=dag)

    # Data should include pretreatment variable
    data = pd.DataFrame(
        {"X": [1, 2, 3, 4], "Y": [2, 4, 6, 8], "Z": [0, 1, 0, 1], "P": [1, 1, 0, 0]}
    )

    regressor.fit(data[["X", "Z", "P"]], data["Y"])

    # Feature columns should include pretreatment
    assert set(regressor.feature_columns_) == {"X", "Z", "P"}
    assert regressor.pretreatment_vars_ == ["P"]


def test_multi_output_not_implemented():
    """Test that multi_output raises NotImplementedError."""
    dag = DAG(
        ebunch=[("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": []},
    )

    with pytest.raises(
        NotImplementedError, match="Multiple outcome support is planned"
    ):
        NaiveBackdoorRegressor(causal_graph=dag, multi_output=True)
