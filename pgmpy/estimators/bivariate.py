from collections.abc import Callable

import numpy as np
import pandas as pd
from sklearn.base import clone

from pgmpy.estimators import CITests as ci_tests

# A set of allowed independence tests for this ANM implementation.
# Only tests suitable for continuous or mixed data are included.
ALLOWED_ANM_CI_TESTS = {ci_tests.pearsonr, ci_tests.gcm, ci_tests.pillai_trace}


def ANM(
    x: np.ndarray,
    y: np.ndarray,
    regressor,
    independence_test: Callable,
    significance_level: float = 0.05,
) -> str:
    """Performs bivariate causal discovery using the Additive Noise Model (ANM).

    This function tests for a causal relationship between two variables, `x` and `y`,
    in both directions (`x` -> `y` and `y` -> `x`). It uses a provided regressor
    and a specific set of allowed pgmpy conditional independence tests.

    The ANM principle assumes that for the true causal direction, the residuals of a
    regression are independent of the cause variable. This implementation infers the
    causal direction by comparing the p-values from these independence tests.

    Parameters
    ----------
    x : np.ndarray
        A 1D NumPy array representing the first variable.
    y : np.ndarray
        A 1D NumPy array representing the second variable.
    regressor : sklearn-compatible estimator
        An instance of a regressor model (e.g., from scikit-learn) with
        `.fit()` and `.predict()` methods.
    independence_test : callable
        A function from pgmpy.estimators.CITests to test for independence.
        Allowed tests are 'pearsonr', 'gcm', and 'pillai_trace'.
    significance_level : float, optional
        The significance level for the independence test, used to decide the
        causal direction. Defaults to 0.05.

    Returns
    -------
    str
        The inferred causal orientation. One of 'x->y', 'y->x', or 'x--y'
        (for an undecided relationship).

    Raises
    ------
    ValueError
        If the provided `independence_test` is not in the allowed set of tests.
    """
    # --- Validate that the provided independence test is allowed ---
    if independence_test not in ALLOWED_ANM_CI_TESTS:
        allowed_names = sorted([f.__name__ for f in ALLOWED_ANM_CI_TESTS])
        raise ValueError(
            f"Invalid independence_test. Allowed tests for ANM are: {allowed_names}. "
            f"Got: {independence_test.__name__}"
        )

    # --- Internal helper to run the pgmpy test ---
    def run_independence_test(cause_arr, residual_arr):
        # For pgmpy tests, create a DataFrame and call with named args
        df = pd.DataFrame(
            {"cause": cause_arr.flatten(), "residual": residual_arr.flatten()}
        )
        # pgmpy tests return a tuple (statistic, p_value, ...); we need the p_value
        return independence_test(X="cause", Y="residual", Z=[], data=df, boolean=False)[
            1
        ]

    # Reshape data for scikit-learn
    x_reshaped = x.reshape(-1, 1)
    y_reshaped = y.reshape(-1, 1)

    # --- Test for X -> Y ---
    reg_xy = clone(regressor)
    reg_xy.fit(x_reshaped, y_reshaped.ravel())
    y_pred = reg_xy.predict(x_reshaped)
    residuals_xy = y_reshaped.flatten() - y_pred.flatten()
    p_value_xy = run_independence_test(x, residuals_xy)

    # --- Test for Y -> X ---
    reg_yx = clone(regressor)
    reg_yx.fit(y_reshaped, x_reshaped.ravel())
    x_pred = reg_yx.predict(y_reshaped)
    residuals_yx = x_reshaped.flatten() - x_pred.flatten()
    p_value_yx = run_independence_test(y, residuals_yx)

    # --- Decision Logic ---
    accept_xy = p_value_xy >= significance_level
    accept_yx = p_value_yx >= significance_level

    if accept_xy and not accept_yx:
        return "x->y"
    elif not accept_xy and accept_yx:
        return "y->x"
    else:
        return "x--y"
