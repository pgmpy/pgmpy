"""
Tests for Naive Adjustment Regressor.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.models import LinearGaussianBayesianNetwork as LGBN
from pgmpy.prediction.NaiveAdjustmentRegressor import NaiveAdjustmentRegressor


def make_estimator():
    """Create a valid estimator for sklearn compatibility tests."""
    dag = DAG()
    dag.add_nodes_from([0, 1])  # exposure=0, outcome=1 (dummy)

    # assign roles
    dag = dag.with_role("exposure", [0])
    dag = dag.with_role("outcome", [1])
    dag = dag.with_role("adjustment", [])
    return NaiveAdjustmentRegressor(causal_graph=dag, estimator=LinearRegression())


@parametrize_with_checks([make_estimator()])
def test_sklearn_compatibility(estimator, check):
    """Test sklearn compatibility using parametrize_with_checks."""
    check(estimator)


def test_basic_functionality_with_adjustment():
    """Test basic fit and predict functionality with synthetic causal data."""

    lgbn = DAG.from_dagitty(
        "dag { Z -> X [beta=0.5] X -> Y [beta=2.0] Z -> Y [beta=1.5] }"
    )

    data = lgbn.simulate(1000, seed=42)

    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    )

    estimators = [
        LinearRegression(),
        RandomForestRegressor(n_estimators=10, random_state=42),
        DummyRegressor(),
    ]

    for base_est in estimators:
        regressor = NaiveAdjustmentRegressor(causal_graph=dag, estimator=base_est)

        train_size = int(0.7 * len(data))
        X_train = data[["X", "Z"]].iloc[:train_size]
        y_train = data["Y"].iloc[:train_size]
        X_test = data[["X", "Z"]].iloc[train_size:]
        y_test = data["Y"].iloc[train_size:]

        regressor.fit(X_train, y_train)
        predictions = regressor.predict(X_test)

        assert len(predictions) == len(y_test)
        assert regressor.exposure_var_ == "X"
        assert regressor.outcome_var_ == "Y"
        assert regressor.adjustment_vars_ == ["Z"]

        if isinstance(base_est, LinearRegression):
            # Expected coefficients: [X_coef=2.0, Z_coef=1.5] from simulation
            # Feature order is [X, Z] (exposure + adjustment)
            expected_coefs = [2.0, 1.5]  # beta coefficients from DAG simulation
            actual_coefs = regressor.estimator_.coef_

            # Check coefficients are close to expected values (allowing for noise)
            np.testing.assert_allclose(actual_coefs, expected_coefs, atol=0.05)

            # Check intercept is close to 0 (no baseline effect in simulation)
            assert abs(regressor.estimator_.intercept_) < 0.1

        feature_names = regressor.get_feature_names_out()
        expected_features = ["X", "Z"]
        assert list(feature_names) == expected_features


def test_dataframe_input_for_both_x_and_y():
    """Test that regressor works when both X and y are DataFrames."""

    lgbn = LGBN([("Z", "X"), ("Z", "Y"), ("X", "Y")])

    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    )
    cpd_x = LinearGaussianCPD("X", beta=[0, 0.5], std=0.01, evidence=["Z"])
    cpd_y = LinearGaussianCPD("Y", beta=[0, 0.3, 0.2], std=0.01, evidence=["X", "Z"])
    cpd_z = LinearGaussianCPD("Z", beta=[0], std=0.01)
    lgbn.add_cpds(cpd_x, cpd_y, cpd_z)
    data = lgbn.simulate(10000, seed=42)

    regressor = NaiveAdjustmentRegressor(causal_graph=dag)

    X_df = data[["X", "Z"]]
    y_df = data[["Y"]]

    regressor.fit(X_df, y_df)
    predictions = regressor.predict(X_df)

    assert len(predictions) == len(data)
    assert regressor.exposure_var_ == "X"
    assert regressor.adjustment_vars_ == ["Z"]
    assert list(regressor.get_feature_names_out()) == ["X", "Z"]

    true_values = data["Y"].values
    np.testing.assert_allclose(predictions, true_values, atol=0.1)


def test_no_adjustment_variables():
    """Test case where there are no adjustment variables (no confounders)."""
    lgbn = DAG.from_dagitty("dag { X -> Y [beta=2.0] }")

    data = lgbn.simulate(100, seed=42)

    dag = DAG(
        ebunch=[("X", "Y")],
        roles={
            "exposure": "X",
            "outcome": "Y",
            "adjustment": [],
        },
    )

    regressor = NaiveAdjustmentRegressor(causal_graph=dag)
    regressor.fit(data[["X"]], data["Y"])
    predictions = regressor.predict(data[["X"]])

    assert len(predictions) == len(data)
    assert regressor.adjustment_vars_ == []
    assert list(regressor.get_feature_names_out()) == ["X"]
    assert isinstance(regressor.estimator_, LinearRegression)


def test_multiple_adjustment_variables():
    """Test with multiple adjustment variables."""
    lgbn = DAG.from_dagitty(
        "dag { U1 -> X [beta=0.3] U1 -> Y [beta=0.6] U2 -> X [beta=0.4] U2 -> Y [beta=0.7] X -> Y [beta=1.5] }"
    )

    data = lgbn.simulate(200, seed=42)

    np.random.seed(42)
    data["noise1"] = np.random.normal(0, 1, len(data))
    data["noise2"] = np.random.normal(0, 1, len(data))

    # DAG with multiple confounders: U1 -> X, U1 -> Y, U2 -> X, U2 -> Y, X -> Y
    dag = DAG(
        ebunch=[("U1", "X"), ("U1", "Y"), ("U2", "X"), ("U2", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["U1", "U2"]},
    )

    regressor = NaiveAdjustmentRegressor(causal_graph=dag, estimator=LinearRegression())

    X_with_noise = data[["X", "U1", "U2", "noise1", "noise2"]]
    regressor.fit(X_with_noise, data["Y"])
    predictions = regressor.predict(X_with_noise)

    assert len(predictions) == len(data)
    assert regressor.exposure_var_ == "X"
    assert set(regressor.adjustment_vars_) == {"U1", "U2"}
    assert set(regressor.get_feature_names_out()) == {"X", "U1", "U2"}

    assert (
        regressor.n_features_in_ == 5
    )  # Input has 5 columns: X, U1, U2, noise1, noise2
    assert len(regressor.get_feature_names_out()) == 3  # Only X, U1, U2 are used


def test_error_handling():
    """Test various error conditions and validation."""

    # Test missing required roles
    dag_no_outcome = DAG(ebunch=[("X", "Y")], roles={"exposure": "X"})
    regressor = NaiveAdjustmentRegressor(causal_graph=dag_no_outcome)

    with pytest.raises(
        ValueError, match="Exactly one outcome variable must be defined"
    ):
        regressor.fit(pd.DataFrame({"X": [1, 2], "Y": [3, 4]}), [5, 6])

    # Test multiple exposure variables (should fail)
    dag_multi_exposure = DAG(
        ebunch=[("X1", "Y"), ("X2", "Y")],
        roles={"exposure": ["X1", "X2"], "outcome": "Y"},
    )
    regressor = NaiveAdjustmentRegressor(causal_graph=dag_multi_exposure)

    with pytest.raises(
        ValueError, match="Exactly one exposure variable must be defined"
    ):
        regressor.fit(pd.DataFrame({"X1": [1, 2], "X2": [3, 4], "Y": [5, 6]}), [7, 8])

    # Test missing required columns in data
    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    )

    regressor = NaiveAdjustmentRegressor(causal_graph=dag)

    incomplete_data = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})

    with pytest.raises(ValueError, match="Missing required columns"):
        regressor.fit(incomplete_data, [5, 6])


def test_numpy_array_input():
    """Test that regressor works with numpy array inputs."""
    # Use integer column names for numpy array input
    dag = DAG(
        ebunch=[
            (1, 0),
            (1, 2),
            (0, 2),
        ],  # Column 1 -> Column 0, Column 1 -> Column 2, Column 0 -> Column 2
        roles={"exposure": [0], "outcome": [2], "adjustment": [1]},
    )

    regressor = NaiveAdjustmentRegressor(causal_graph=dag)

    np.random.seed(42)
    n_samples = 50
    X_array = np.random.normal(
        0, 1, (n_samples, 2)
    )  # Columns 0 and 1 (exposure and adjustment)
    y_array = np.random.normal(0, 1, n_samples)  # Target (outcome column 2)

    regressor.fit(X_array, y_array)
    predictions = regressor.predict(X_array)

    assert len(predictions) == n_samples
    assert regressor.feature_columns_ == [0, 1]  # exposure + adjustment


def test_sample_weight_support():
    """Test that sample_weight parameter is properly passed to base estimator."""
    lgbn = DAG.from_dagitty("dag { X -> Y [beta=2.0] }")
    data = lgbn.simulate(4, seed=42)

    dag = DAG(
        ebunch=[("X", "Y")],
        roles={
            "exposure": "X",
            "outcome": "Y",
            "adjustment": [],
        },
    )

    regressor = NaiveAdjustmentRegressor(causal_graph=dag, estimator=LinearRegression())

    sample_weights = np.array([1, 1, 2, 2])

    regressor.fit(data[["X"]], data["Y"], sample_weight=sample_weights)
    predictions = regressor.predict(data[["X"]])

    assert len(predictions) == len(data)


def test_dag_roles_validation():
    """Test that DAG roles are properly validated during fit."""
    dag_valid = DAG(
        ebunch=[("X", "Y")], roles={"exposure": "X", "outcome": "Y", "adjustment": []}
    )

    regressor = NaiveAdjustmentRegressor(causal_graph=dag_valid)
    exposure_vars = list(regressor.causal_graph.get_role("exposure"))
    outcome_vars = list(regressor.causal_graph.get_role("outcome"))
    adjustment_vars = list(regressor.causal_graph.get_role("adjustment"))
    pretreatment_vars = list(
        regressor.causal_graph.get_role("pretreatment")
        if regressor.causal_graph.has_role("pretreatment")
        else []
    )

    assert len(exposure_vars) == 1, f"Expected 1 exposure, got {len(exposure_vars)}"
    assert len(outcome_vars) == 1, f"Expected 1 outcome, got {len(outcome_vars)}"
    assert exposure_vars[0] == "X"
    assert outcome_vars[0] == "Y"
    assert adjustment_vars == []
    assert pretreatment_vars == []

    # Invalid DAG (no roles) should fail
    dag_no_roles = DAG(ebunch=[("X", "Y")])
    regressor_invalid = NaiveAdjustmentRegressor(causal_graph=dag_no_roles)

    exposure_vars_invalid = list(regressor_invalid.causal_graph.get_role("exposure"))
    outcome_vars_invalid = list(regressor_invalid.causal_graph.get_role("outcome"))
    assert len(exposure_vars_invalid) == 0
    assert len(outcome_vars_invalid) == 0

    with pytest.raises(
        ValueError, match="Exactly one exposure variable must be defined"
    ):
        test_data = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
        regressor_invalid.fit(test_data[["X"]], test_data["Y"])


def test_array_input_with_integer_dag_variables():
    """Test that array input works with integer DAG variable names."""
    # DAG with integer column names matching array structure
    dag = DAG(
        ebunch=[
            (1, 0),
            (1, 2),
            (0, 2),
        ],  # Column 1 -> Column 0, Column 1 -> Column 2, Column 0 -> Column 2
        roles={"exposure": [0], "outcome": [2], "adjustment": [1]},
    )

    regressor = NaiveAdjustmentRegressor(causal_graph=dag)

    np.random.seed(42)
    X_array = np.random.normal(0, 1, (50, 2))  # Columns 0 and 1
    y_array = np.random.normal(0, 1, 50)  # Target

    regressor.fit(X_array, y_array)
    predictions = regressor.predict(X_array)
    assert len(predictions) == 50


def test_adjustment_role_behavior():
    """Test that missing and empty adjustment roles behave identically."""
    dag_missing_adj = DAG(
        ebunch=[("X", "Y")],
        roles={"exposure": "X", "outcome": "Y"},
    )

    dag_empty_adj = DAG(
        ebunch=[("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": []},
    )

    regressor1 = NaiveAdjustmentRegressor(causal_graph=dag_missing_adj)
    regressor2 = NaiveAdjustmentRegressor(causal_graph=dag_empty_adj)

    data = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})

    regressor1.fit(data[["X"]], [5, 6])
    regressor2.fit(data[["X"]], [5, 6])

    assert regressor1.adjustment_vars_ == []
    assert regressor2.adjustment_vars_ == []


def test_empty_adjustment_role_explicit():
    """Test that explicit empty adjustment role works correctly."""
    dag = DAG(
        ebunch=[("X", "Y")],
        roles={
            "exposure": "X",
            "outcome": "Y",
            "adjustment": [],
        },
    )

    regressor = NaiveAdjustmentRegressor(causal_graph=dag)
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

    regressor = NaiveAdjustmentRegressor(causal_graph=dag)

    data = pd.DataFrame(
        {"X": [1, 2, 3, 4], "Y": [2, 4, 6, 8], "Z": [0, 1, 0, 1], "P": [1, 1, 0, 0]}
    )

    regressor.fit(data[["X", "Z", "P"]], data["Y"])

    assert set(regressor.feature_columns_) == {"X", "Z", "P"}
    assert regressor.pretreatment_vars_ == ["P"]
