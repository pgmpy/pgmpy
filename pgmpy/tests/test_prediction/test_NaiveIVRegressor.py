import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base import DAG
from pgmpy.prediction.NaiveIVRegressor import NaiveIVRegressor


def make_estimator():
    """Create a valid estimator for sklearn compatibility tests."""
    G = DAG(
        [(0, 1), (1, 2), (3, 2)],
        roles={
            "exposure": [1],
            "outcome": [2],
            "instrument": [0],
        },
    )
    return NaiveIVRegressor(
        causal_graph=G,
        stage1_estimator=LinearRegression(),
        stage2_estimator=LinearRegression(),
    )


def make_simulated_plr(n=500, effect=0.6, nuisance_scale=0.5, seed=42):
    """Simulate a simple : Z -> D, Z -> Y, and D -> Y with linear relationships."""
    rng = np.random.RandomState(seed)
    Z1 = rng.normal(size=n)
    Z2 = rng.normal(size=n)

    D = 0.4 * Z1 - 0.3 * Z2 + rng.normal(scale=nuisance_scale, size=n)
    Y = effect * D + 0.6 * Z1 + 0.2 * Z2 + rng.normal(scale=nuisance_scale, size=n)

    df = pd.DataFrame({"D": D, "Z1": Z1, "Z2": Z2, "Y": Y})

    return df.loc[:, ["D", "Z1", "Z2"]], df.loc[:, ["Y"]]


@parametrize_with_checks([make_estimator()])
def test_sklearn_compatibility(estimator, check):
    """Test sklearn compatibility using parametrize_with_checks."""
    check(estimator)


def test_dag_roles_validation_and_pretreatment_support():
    """Test role extraction and pretreatment variable handling."""
    G = DAG(
        ebunch=[("Z", "E"), ("E", "Y"), ("P", "Y")],
        roles={
            "exposure": ["E"],
            "outcome": ["Y"],
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
    exposure_vars = list(model.causal_graph.get_role("exposure"))
    outcome_vars = list(model.causal_graph.get_role("outcome"))
    instrument_vars_ = model.causal_graph.get_role("instrument")
    pretreat_vars = list(model.causal_graph.get_role("pretreatment"))
    assert exposure_vars and outcome_vars
    assert instrument_vars_ == ["Z"]
    assert pretreat_vars == ["P"]

    # Now fit with matching DataFrame and verify feature columns include pretreatment
    rng = np.random.RandomState(2)
    P = rng.normal(size=50)
    Z = rng.normal(size=50)
    E = 0.4 * Z + rng.normal(scale=0.2, size=50)
    Y = 1.2 * E + 0.3 * P + rng.normal(scale=0.2, size=50)

    df = pd.DataFrame({"E": E, "Z": Z, "P": P})
    _ = model.fit(df, pd.Series(Y, name="Y"))
    assert set(model.feature_columns_predict_) >= {"E", "P"}
