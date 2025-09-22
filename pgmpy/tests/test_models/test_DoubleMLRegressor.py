# tests/test_doubleml_regressor_sklearn_checks.py
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.utils import check_random_state
from sklearn.utils.estimator_checks import parametrize_with_checks

from pgmpy.base.DAG import DAG
from pgmpy.models.DoubleMLRegressor import DoubleMLRegressor


# -------------------------
# Helper: build a simple DAG with roles matching x0,x1,x2
# -------------------------
def make_role_dag(
    node_names=("x0", "x1", "x2"), exposure="x0", adjustments=("x1", "x2"), outcome="y"
):
    """
    Creates a pgmpy.DAG and assign roles:
      - exposure role: exposure
      - adjustment role: adjustments (set)
      - outcome role: outcome (ensures outcome node exists)
    Returns: DAG instance
    """
    G = DAG()
    # add nodes
    for n in node_names:
        G.add_node(n)
    if outcome is not None and outcome not in G:
        G.add_node(outcome)
    # assign roles via with_role
    G = G.with_role("exposure", exposure)
    if adjustments:
        G = G.with_role("adjustment", set(adjustments))
    if outcome:
        G = G.with_role("outcome", outcome)

    return G


# -------------------------
# Factory for parametrize_with_checks
# -------------------------
def make_estimator_for_checks():
    """
    Return an unfitted DoubleMLRegressor instance configured to accept numpy arrays.
    Important: passes feature_names so sklearn's tests (which passes ndarrays) can map columns.
    """
    G = make_role_dag(
        node_names=("x0", "x1", "x2"),
        exposure="x0",
        adjustments=(),
        outcome="y",
    )
    est = DoubleMLRegressor(
        dag=G,
        estimator_g=DummyRegressor(strategy="mean"),
        estimator_m=DummyRegressor(strategy="mean"),
        n_folds=1,
        seed=0,
        allow_array_unnamed=False,
    )
    return est


@parametrize_with_checks([make_estimator_for_checks()])
def test_sklearn_compatibility(estimator, check):
    """Run sklearn's compatibility checks (one check at a time)."""
    check(estimator)


# -------------------------
# Synthetic-data recovery test
# -------------------------
def test_doubleml_recovers_theta_on_simple_plr():
    """
    Simulate a PLR model:
       Z ~ N(0, I)
       D = g(Z) + nu   (g linear)
       Y = theta * D + m(Z) + eps   (m linear)
    Use linear regressors for nuisances -> DoubleML should recover theta approximately.
    """
    rng = check_random_state(0)
    n = 500
    p_z = 2  # number of adjustments (x1,x2)
    theta_true = 2.5

    # simulate covariates Z = [x1, x2]
    Z = rng.normal(size=(n, p_z))
    # build g(Z) = linear function
    beta_g = np.array([0.8, -0.5])
    g_z = Z.dot(beta_g)
    # treatment with some noise
    D = g_z + rng.normal(scale=0.5, size=n)

    # outcome baseline m(Z)
    beta_m = np.array([1.2, 0.7])
    m_z = Z.dot(beta_m)

    # outcome with treatment effect
    Y = theta_true * D + m_z + rng.normal(scale=0.5, size=n)

    # Construct DataFrame with column names that match DAG roles
    df = pd.DataFrame(
        {
            "x0": D,  # exposure
            "x1": Z[:, 0],
            "x2": Z[:, 1],
        }
    )
    y = pd.Series(Y, name="y")

    # DAG must have exposure/outcome/adjustment role
    G = make_role_dag(
        node_names=("x0", "x1", "x2"),
        exposure="x0",
        adjustments=("x1", "x2"),
        outcome="y",
    )

    # use linear regressors as nuisances (well-specified for this simulation)
    est = DoubleMLRegressor(
        dag=G,
        estimator_g=LinearRegression(),
        estimator_m=LinearRegression(),
        n_folds=3,
        seed=0,
        allow_array_unnamed=False,
    )

    est.fit(df, y)

    # the DML residual method estimates theta; checking it's near truth
    theta_hat = getattr(est, "treatment_effect_", None)
    assert theta_hat is not None, "Estimator did not set treatment_effect_"

    # Allow some tolerance because of finite-sample noise
    assert np.isclose(
        theta_hat, theta_true, atol=0.15
    ), f"theta_hat={theta_hat} not close to true={theta_true}"

    # Also test that predict produces values close to Y on the training data
    preds = est.predict(df)
    assert preds.shape[0] == n
    mse = np.mean((preds - Y) ** 2)
    # Check MSE is reasonable and not super huge
    assert mse < 6.0, f"MSE too large: {mse}"
