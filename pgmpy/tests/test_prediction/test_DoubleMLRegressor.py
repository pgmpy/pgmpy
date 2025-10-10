import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base.DAG import DAG
from pgmpy.prediction.DoubleMLRegressor import DoubleMLRegressor


@pytest.fixture
def dag():
    return DAG(
        ebunch=[("Z1", "D"), ("Z2", "D"), ("D", "Y"), ("Z1", "Y"), ("Z2", "Y")],
        roles={"exposure": "D", "adjustment": ("Z1", "Z2"), "outcome": "Y"},
    )


def estimator_for_sklearn_checks():
    G = DAG([(0, 3), (0, 1), (0, 2)], roles={"exposure": [0], "outcome": [3]})

    est = DoubleMLRegressor(
        causal_graph=G,
        nuisance_estimators=(LinearRegression(), LinearRegression()),
        effect_estimator=LinearRegression(),
        n_folds=1,
        seed=0,
    )
    return est


@parametrize_with_checks([estimator_for_sklearn_checks()])
def test_sklearn_compatibility(estimator, check):
    """Run sklearn's compatibility checks."""
    check(estimator)


def make_simulated_plr(n=500, effect=0.6, nuisance_scale=0.5, seed=42):
    """Simulate a simple : Z -> D, Z -> Y, and D -> Y with linear relationships."""
    rng = np.random.RandomState(seed)
    Z1 = rng.normal(size=n)
    Z2 = rng.normal(size=n)

    D = 0.4 * Z1 - 0.3 * Z2 + rng.normal(scale=nuisance_scale, size=n)
    Y = effect * D + 0.6 * Z1 + 0.2 * Z2 + rng.normal(scale=nuisance_scale, size=n)

    df = pd.DataFrame({"D": D, "Z1": Z1, "Z2": Z2, "Y": Y})

    return df.loc[:, ["D", "Z1", "Z2"]], df.loc[:, ["Y"]]


def test_dataframe_input_for_both_x_and_y(dag):
    """Test that regressor works when both X and y are DataFrames (y as DataFrame column)."""
    X, y = make_simulated_plr(n=1000, seed=1)

    model = DoubleMLRegressor(
        causal_graph=dag,
        nuisance_estimators=LinearRegression(),
        effect_estimator=LinearRegression(),
        n_folds=3,
        seed=0,
    )

    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)

    assert model.exposure_var_ == "D"
    assert model.outcome_var_ == "Y"
    assert set(model.adjustment_vars_) == {"Z1", "Z2"}
    assert set(model.feature_columns_) == {"D", "Z1", "Z2"}
    assert model.n_samples_ == 1000

    assert len(model.treatment_est_) == 3
    assert len(model.outcome_est_) == 3


def test_numpy_array_input_with_integer_dag_variables():
    """Test that regressor works with numpy array inputs when DAG uses integer-string column names."""
    # Construct DAG with stringified integer names to match DataFrame conversion behavior
    dag = DAG(
        ebunch=[(1, 0), (1, 2), (0, 2)],
        roles={"exposure": [0], "outcome": [2], "adjustment": [1]},
    )

    model = DoubleMLRegressor(
        causal_graph=dag,
        nuisance_estimators=LinearRegression(),
        effect_estimator=LinearRegression(),
        n_folds=1,
    )

    n_samples = 50
    # Build array columns: column 0 = exposure (0), column 1 = adjustment (1)
    X_array = np.random.normal(0, 1, (n_samples, 2))
    y_array = np.random.normal(0, 1, n_samples)

    _ = model.fit(X_array, y_array)
    preds = model.predict(X_array)
    assert len(preds) == n_samples
    assert model.feature_columns_ == [0, 1]


def test_no_adjustment_variables():
    """Test case where there are no adjustment variables (no confounders)."""
    # Create simple DAG D -> Y with no adjustments
    # Use small simulation via dagitty to create data
    lgbn = DAG.from_dagitty("dag { D -> Y [beta=0.6] }")
    data = lgbn.simulate(n_samples=200, seed=42)

    dag = DAG(
        ebunch=[("D", "Y")], roles={"exposure": "D", "outcome": "Y", "adjustment": []}
    )
    model = DoubleMLRegressor(
        causal_graph=dag,
        nuisance_estimators=LinearRegression(),
        effect_estimator=LinearRegression(),
        n_folds=1,
    )

    # Fit with only exposure column in X (no adjustments)
    _ = model.fit(data[["D"]], data["Y"])
    preds = model.predict(data[["D"]])
    assert len(preds) == len(data)
    assert model.adjustment_vars_ == []
    assert list(model.feature_columns_) == ["D"]
    assert hasattr(model, "effect_est_")


def test_multiple_adjustment_variables_and_noise_columns():
    """Test with multiple adjustment variables and extra noise columns in X."""
    lgbn = DAG.from_dagitty(
        "dag { U1 -> X [beta=0.3] U1 -> Y [beta=0.2] U2 -> X [beta=0.3] U2 -> Y [beta=0.4] X -> Y [beta=0.6] }"
    )
    data = lgbn.simulate(n_samples=300, seed=42)
    # add unrelated noise columns
    rng = np.random.RandomState(42)
    data["noise1"] = rng.normal(size=len(data))
    data["noise2"] = rng.normal(size=len(data))

    dag = DAG(
        ebunch=[("U1", "X"), ("U1", "Y"), ("U2", "X"), ("U2", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["U1", "U2"]},
    )

    model = DoubleMLRegressor(
        causal_graph=dag,
        nuisance_estimators=LinearRegression(),
        effect_estimator=LinearRegression(),
        n_folds=3,
    )

    X_with_noise = data[["X", "U1", "U2", "noise1", "noise2"]]
    _ = model.fit(X_with_noise, data["Y"])

    # check attributes
    assert model.exposure_var_ == "X"
    assert set(model.adjustment_vars_) == {"U1", "U2"}
    # feature_columns_ should include exposure + adjustments + pretreatment (if any)
    assert model.feature_columns_[:3] == ["X", "U1", "U2"]
    # n_features_in_ counts total columns passed to fit
    assert model.n_features_in_ == X_with_noise.shape[1]


def test_error_handling_missing_roles_and_multiple_exposure():
    """Test various error conditions and validation."""
    # missing outcome role
    dag_no_outcome = DAG(ebunch=[("X", "Y")], roles={"exposure": "X"})
    model1 = DoubleMLRegressor(
        causal_graph=dag_no_outcome,
        nuisance_estimators=LinearRegression(),
        effect_estimator=LinearRegression(),
    )
    with pytest.raises(Exception):
        model1.fit(pd.DataFrame({"X": [1, 2], "Y": [3, 4]}), [5, 6])

    # multiple exposures should raise
    dag_multi_exposure = DAG(
        ebunch=[("X1", "Y"), ("X2", "Y")],
        roles={"exposure": ["X1", "X2"], "outcome": "Y"},
    )
    model2 = DoubleMLRegressor(
        causal_graph=dag_multi_exposure,
        nuisance_estimators=LinearRegression(),
        effect_estimator=LinearRegression(),
    )
    with pytest.raises(Exception):
        model2.fit(pd.DataFrame({"X1": [1, 2], "X2": [3, 4], "Y": [5, 6]}), [7, 8])

    # missing required columns in data
    dag = DAG(
        ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
        roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    )
    model3 = DoubleMLRegressor(
        causal_graph=dag,
        nuisance_estimators=LinearRegression(),
        effect_estimator=LinearRegression(),
    )
    incomplete = pd.DataFrame({"X": [1, 2], "Y": [3, 4]})
    with pytest.raises(ValueError):
        model3.fit(incomplete, [5, 6])


def test_sample_weight_support_and_shapes(dag):
    """Test that sample_weight parameter is accepted and shape-validated."""
    X, y = make_simulated_plr(n=150, seed=5)
    # dag = make_simple_dag_roles()
    model = DoubleMLRegressor(
        causal_graph=dag,
        nuisance_estimators=LinearRegression(),
        effect_estimator=LinearRegression(),
        n_folds=1,
    )

    # list-like sample weight accepted
    sw_list = [1.0] * X.shape[0]
    _ = model.fit(X, y, sample_weight=sw_list)

    # pandas Series accepted
    sw_ser = pd.Series([1.0] * X.shape[0])
    _ = model.fit(X, y, sample_weight=sw_ser)

    # wrong-length should raise
    with pytest.raises(ValueError):
        model.fit(X, y, sample_weight=[1.0] * (X.shape[0] - 1))


def test_dag_roles_validation_and_pretreatment_support():
    """Test role extraction and pretreatment variable handling."""
    dag = DAG(
        ebunch=[("P", "D"), ("Z", "D"), ("Z", "Y"), ("D", "Y")],
        roles={
            "exposure": "D",
            "outcome": "Y",
            "adjustment": ["Z"],
            "pretreatment": ["P"],
        },
    )
    model = DoubleMLRegressor(
        causal_graph=dag,
        nuisance_estimators=LinearRegression(),
        effect_estimator=LinearRegression(),
        n_folds=1,
    )
    # Before fit the roles are accessible via DAG; check that role lists are non-empty
    exposure_vars = list(model.causal_graph.get_role("exposure"))
    outcome_vars = list(model.causal_graph.get_role("outcome"))
    adjustment_vars = list(model.causal_graph.get_role("adjustment"))
    pretreat_vars = list(model.causal_graph.get_role("pretreatment"))
    assert exposure_vars and outcome_vars
    assert adjustment_vars == ["Z"]
    assert pretreat_vars == ["P"]

    # Now fit with matching DataFrame and verify feature columns include pretreatment
    rng = np.random.RandomState(2)
    P = rng.normal(size=50)
    Z = rng.normal(size=50)
    D = 0.5 * P + 0.4 * Z + rng.normal(scale=0.2, size=50)
    Y = 1.2 * D + 0.3 * Z + rng.normal(scale=0.2, size=50)
    df = pd.DataFrame({"D": D, "Z": Z, "P": P})
    _ = model.fit(df, pd.Series(Y, name="Y"))
    assert set(model.feature_columns_) >= {"D", "Z", "P"}


def test_doubleml_recovers_theta_with_RF():
    """Use pgmpy DAG + simulator to generate linear-Gaussian data and check theta recovery."""

    lgbn = DAG.from_dagitty(
        "dag { U1 -> X [beta=0.3] U1 -> Y [beta=0.2] U2 -> X [beta=0.3] U2 -> Y [beta=0.4] X -> Y [beta=0.6] }"
    )

    data = lgbn.simulate(1000, seed=42)  # returns a pandas DataFrame

    df = data.loc[:, ["X", "U1", "U2"]]
    df = (df - df.mean(axis=0)) / df.std(axis=0)

    y = data["Y"]

    G = DAG(
        lgbn.edges(),
        roles={"exposure": "X", "adjustment": ("U1", "U2"), "outcome": "Y"},
    )

    est = DoubleMLRegressor(
        causal_graph=G,
        nuisance_estimators=(
            RandomForestRegressor(),
            RandomForestRegressor(),
        ),
        effect_estimator=LinearRegression(),
        n_folds=3,
        seed=0,
    )

    est.fit(df, y)

    assert est.effect_est_.coef_.round(1)[0] == 0.6

    preds = est.predict(df)
    assert preds.shape[0] == df.shape[0]
    mse = np.mean((preds - y.to_numpy()) ** 2)
    assert mse < 0.5


def test_doubleml_recovers_theta_high_dim():
    """Use pgmpy DAG + simulator to generate linear-Gaussian data and check theta recovery in high-dim setting."""

    dag = DAG.from_dagitty(
        """dag { D -> Y [beta=0.6]
               Z1 -> D [beta=0.4]
               Z1 -> Y [beta=0.6]
               Z2 -> D [beta=-0.3]
               Z2 -> Y [beta=0.2]
               Z3 -> D [beta=0.2]
               Z3 -> Y [beta=0.1]
               Z4 -> D [beta=-0.1]
               Z4 -> Y [beta=0.3]
               Z5 -> D [beta=0.3]
               Z5 -> Y [beta=-0.2]
               Z6 -> D [beta=0.1]
               Z6 -> Y [beta=0.2]
               Z7 -> D [beta=-0.2]
               Z7 -> Y [beta=0.1]
               Z8 -> D [beta=0.3]
               Z8 -> Y [beta=-0.3]
               Z9 -> D [beta=0.2]
               Z9 -> Y [beta=0.2]
               Z10 -> D [beta=-0.3]
               Z10 -> Y [beta=0.1]}"""
    )

    data = dag.simulate(10000, seed=42)

    df = data.loc[:, list(set(dag.nodes()).difference({"Y"}))]
    df = df - df.mean(axis=0)

    y = data["Y"]

    G = DAG(
        dag.edges(),
        roles={
            "exposure": "D",
            "adjustment": dag.get_role("adjustment"),
            "outcome": "Y",
        },
    )

    est = DoubleMLRegressor(
        causal_graph=G,
        nuisance_estimators=(
            RandomForestRegressor(),
            RandomForestRegressor(),
        ),
        effect_estimator=LinearRegression(),
        n_folds=3,
        seed=0,
    )

    est.fit(df, y)

    assert est.effect_est_.coef_.round(1)[0] == 0.6

    preds = est.predict(df)
    assert preds.shape[0] == df.shape[0]
