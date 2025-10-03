import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base.DAG import DAG
from pgmpy.prediction.DoubleMLRegressor import DoubleMLRegressor


def make_estimator_for_checks():
    """
    Return an unfitted DoubleMLRegressor instance configured to accept numpy arrays.
    Important: passes feature_names so sklearn's tests (which passes ndarrays) can map columns.
    """
    G = DAG([(0, 3), (0, 1), (0, 2)], roles={"exposure": [0], "outcome": [3]})

    est = DoubleMLRegressor(
        causal_graph=G,
        nuisance_estimators=(LinearRegression(), LinearRegression()),
        effect_estimator=LinearRegression(),
        n_folds=1,
        seed=0,
    )
    return est


@parametrize_with_checks([make_estimator_for_checks()])
def test_sklearn_compatibility(estimator, check):
    """Run sklearn's compatibility checks (one check at a time)."""
    check(estimator)


def test_doubleml_recovers_theta_on_simple_plr():
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

    assert est.effect_estimator_.coef_.round(1)[0] == 0.6

    preds = est.predict(df)
    assert preds.shape[0] == df.shape[0]
    mse = np.mean((preds - y.to_numpy()) ** 2)
    assert mse < 0.5
