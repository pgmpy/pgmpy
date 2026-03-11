import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.prediction.NaiveIVRegressor import NaiveIVRegressor


@pytest.fixture
def dag():
    return DAG(
        ebunch=[("Z1", "X"), ("X", "Y"), ("Z2", "X")],
        roles={
            "exposures": ["X"],
            "outcomes": ["Y"],
            "instrument": ["Z1", "Z2"],
        },
    )


def make_estimator():
    """Create a valid estimator for sklearn compatibility tests."""
    G = DAG(
        [(0, 1), (1, 2), (3, 2)],
        roles={
            "exposures": [1],
            "outcomes": [2],
            "instrument": [0],
        },
    )
    return NaiveIVRegressor(
        causal_graph=G,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )


def make_simulated_plr(n=500, effect=0.6, nuisance_scale=0.5, seed=42):
    """Simulate a simple : Z -> X, X -> Y with linear relationships."""
    rng = np.random.RandomState(seed)
    Z1 = rng.normal(size=n)
    Z2 = rng.normal(size=n)

    X = 0.4 * Z1 - 0.3 * Z2 + rng.normal(scale=nuisance_scale, size=n)
    Y = effect * X + 0.6 * Z1 + 0.2 * Z2 + rng.normal(scale=nuisance_scale, size=n)

    df = pd.DataFrame({"X": X, "Z1": Z1, "Z2": Z2, "Y": Y})

    return df.loc[:, ["X", "Z1", "Z2"]], df.loc[:, ["Y"]]


@parametrize_with_checks([make_estimator()])
def test_sklearn_compatibility(estimator, check):
    """Test sklearn compatibility using parametrize_with_checks."""
    check(estimator)


def test_dataframe_input_for_both_x_and_y(dag):
    X, y = make_simulated_plr(n=1000, seed=1)

    model = NaiveIVRegressor(
        causal_graph=dag,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )
    model.fit(X, y)
    preds = model.predict(X)

    assert len(preds) == len(X)

    assert model.exposure_var_ == "X"
    assert model.outcome_var_ == "Y"
    assert model.instrument_vars_ == ["Z1", "Z2"]
    assert set(model.feature_columns_fit_) == {"X", "Z1", "Z2"}
    assert set(model.feature_columns_predict_) == {"X"}

    assert model.n_features_in_ == 3

    assert model.stage1_est_.coef_.shape == (2,)
    assert model.stage2_est_.coef_.shape == (1, 1)


def test_numpy_array_input_with_integer_dag_variables():
    """Test that regressor works with numpy array inputs when DAG uses integer-string column names."""
    # Construct DAG with stringified integer names to match DataFrame conversion behavior
    dag = DAG(
        ebunch=[(1, 0), (0, 2)],
        roles={"exposures": [0], "outcomes": [2], "instrument": [1]},
    )

    model = NaiveIVRegressor(
        causal_graph=dag,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )

    n_samples = 50
    # Build array columns: column 0 = exposure (0), column 1 = instrument (1)
    X_array = np.random.normal(0, 1, (n_samples, 2))
    y_array = np.random.normal(0, 1, n_samples)

    _ = model.fit(X_array, y_array)
    preds = model.predict(X_array)

    assert len(preds) == n_samples
    assert model.feature_columns_fit_ == [0, 1]
    assert model.feature_columns_predict_ == [0]

    assert model.n_features_in_ == 2
    assert model.stage1_est_.coef_.shape == (1,)
    assert model.stage2_est_.coef_.shape == (1,)


def test_sample_weight_support_and_shapes(dag):
    """Test that sample_weight parameter is accepted and shape-validated."""
    X, y = make_simulated_plr(n=150, seed=5)

    model = NaiveIVRegressor(
        causal_graph=dag,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
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


def test_naiveiv_recovers_theta_with_LR():
    """Use pgmpy DAG + simulator to generate linear-Gaussian data and check theta recovery."""

    lgbn = DAG.from_dagitty(
        "dag { Z1 -> X [beta=0.2] Z2 -> X [beta=0.2] X -> Y [beta=0.3]}"
    )

    data = lgbn.simulate(1000, seed=42)  # returns a pandas DataFrame

    df = data.loc[:, ["X", "Z1", "Z2"]]
    df = (df - df.mean(axis=0)) / df.std(axis=0)

    y = data["Y"]

    G = DAG(
        lgbn.edges(),
        roles={"exposures": "X", "instrument": ("Z1", "Z2"), "outcomes": "Y"},
    )

    model = NaiveIVRegressor(
        causal_graph=G,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )

    model.fit(df, y)

    assert model.stage2_est_.coef_.round(1)[0] == 0.3

    preds = model.predict(df)
    assert preds.shape[0] == df.shape[0]
    mse = np.mean((preds - y.to_numpy()) ** 2)
    assert mse < 0.98


def test_dag_roles_validation_and_pretreatment_support():
    """Test role extraction and pretreatment variable handling."""
    G = DAG(
        ebunch=[("Z", "E"), ("E", "Y"), ("P", "Y")],
        roles={
            "exposures": ["E"],
            "outcomes": ["Y"],
            "instrument": ["Z"],
            "pretreatment": ["P"],
        },
    )

    model = NaiveIVRegressor(
        causal_graph=G,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )
    # Before fit the roles are accessible via DAG; check that role lists are non-empty
    exposure_vars = model.causal_graph.get_role("exposures")
    outcome_vars = model.causal_graph.get_role("outcomes")
    instrument_vars_ = model.causal_graph.get_role("instrument")
    pretreatment_vars = model.causal_graph.get_role("pretreatment")
    assert exposure_vars and outcome_vars
    assert instrument_vars_ == ["Z"]
    assert pretreatment_vars == ["P"]

    # Now fit with matching DataFrame and verify feature columns include pretreatment
    rng = np.random.RandomState(2)
    P = rng.normal(size=50)
    Z = rng.normal(size=50)
    E = 0.4 * Z + rng.normal(scale=0.2, size=50)
    Y = 1.2 * E + 0.3 * P + rng.normal(scale=0.2, size=50)

    df = pd.DataFrame({"E": E, "Z": Z, "P": P})
    _ = model.fit(df, pd.Series(Y, name="Y"))
    assert set(model.feature_columns_predict_) == {"E", "P"}


def test_error_handling_missing_roles_():
    """Test various error conditions and validation."""
    # missing outcome role
    dag_no_outcome = DAG(
        ebunch=[("X", "Y"), ("Z", "X")], roles={"exposures": "X", "instrument": "Z"}
    )
    model1 = NaiveIVRegressor(
        causal_graph=dag_no_outcome,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )
    with pytest.raises(Exception):
        model1.fit(pd.DataFrame({"X": [1, 2], "Y": [3, 4], "Z": [5, 6]}), [7, 8])

    # missing instrument role
    dag_no_instrument = DAG(
        ebunch=[("X", "Y")], roles={"exposures": "X", "outcomes": "Y"}
    )
    model1 = NaiveIVRegressor(
        causal_graph=dag_no_instrument,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )
    with pytest.raises(Exception):
        model1.fit(pd.DataFrame({"X": [1, 2], "Y": [3, 4], "Z": [5, 6]}), [7, 8])


def test_multiple_instrument_variables_and_noise_columns():
    """Test with multiple instrument variables and extra noise columns in X."""
    lgbn = DAG.from_dagitty(
        "dag { U1 -> X [beta=0.3] U2 -> X [beta=0.2] U3 -> X [beta=0.3] U4 -> X [beta=0.4] X -> Y [beta=0.6] }"
    )
    data = lgbn.simulate(n_samples=300, seed=42)
    # add unrelated noise columns
    rng = np.random.RandomState(42)
    data["noise1"] = rng.normal(size=len(data))
    data["noise2"] = rng.normal(size=len(data))

    dag = DAG(
        ebunch=[("U1", "X"), ("U2", "X"), ("U3", "X"), ("U4", "X"), ("X", "Y")],
        roles={"exposures": "X", "outcomes": "Y", "instrument": ["U1", "U2"]},
    )

    model = NaiveIVRegressor(
        causal_graph=dag,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )

    X_with_noise = data[["X", "U1", "U2", "noise1", "noise2"]]
    _ = model.fit(X_with_noise, data["Y"])

    # check attributes
    assert model.exposure_var_ == "X"
    assert set(model.instrument_vars_) == {"U1", "U2"}
    # feature_columns_fit_ should include exposure + adjustments + pretreatment (if any)
    # feature columns_predict_ should include exposure + pretreatment (if any)
    assert model.feature_columns_fit_[:3] == ["X", "U1", "U2"]
    assert model.feature_columns_predict_[0] == "X"
    # n_features_in_ counts total columns passed to fit
    assert model.n_features_in_ == X_with_noise.shape[1]


def test_naiveiv_recovers_theta_high_dim():
    """Use pgmpy DAG + simulator to generate linear-Gaussian data and check theta recovery in high-dim setting."""

    dag = DAG.from_dagitty(
        """dag { D -> Y [beta=0.6]
               Z1 -> D [beta=0.4]
               Z2 -> D [beta=-0.3]
               Z3 -> D [beta=0.2]
               Z4 -> D [beta=-0.1]
               Z5 -> D [beta=0.3]
               Z6 -> D [beta=0.1]
               Z7 -> D [beta=-0.2]
               Z8 -> D [beta=0.3]
               Z9 -> D [beta=0.2]
               Z10 -> D [beta=-0.3]}"""
    )

    data = dag.simulate(10000, seed=42)

    df = data.loc[:, list(set(dag.nodes()).difference({"Y"}))]
    df = df - df.mean(axis=0)

    y = data["Y"]

    G = DAG(
        dag.edges(),
        roles={
            "exposures": "D",
            "instrument": [(f"Z{i}") for i in range(1, 11)],
            "outcomes": "Y",
        },
    )

    est = NaiveIVRegressor(
        causal_graph=G,
        stage1_estimator=RandomForestRegressor(),
        stage2_estimator=LinearRegression(),
    )

    est.fit(df, y)

    assert np.isclose(est.stage2_est_.coef_.round(1)[0], 0.7, rtol=0.2)

    preds = est.predict(df)
    assert preds.shape[0] == df.shape[0]

    mse = np.mean((preds - y.to_numpy()) ** 2)
    assert mse < 1.07


def test_naiveIV_no_estimators(dag):
    """Test whether NaiveIVRegressor works when no estimators are provided (defaults to LinearRegression)."""
    X, y = make_simulated_plr(n=150, seed=5)

    model = NaiveIVRegressor(
        causal_graph=dag,
    )

    model.fit(X, y)

    assert hasattr(model, "stage1_est_")
    assert hasattr(model, "stage2_est_")

    assert isinstance(model.stage1_est_, LinearRegression)
    assert isinstance(model.stage2_est_, LinearRegression)

    assert getattr(model.stage2_est_, "coef_", None) is not None

    preds = model.predict(X)
    assert len(preds) == len(X)

    assert model.n_features_in_ == X.shape[1]
