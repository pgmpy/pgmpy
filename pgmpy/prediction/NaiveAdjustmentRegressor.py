"""
Naive Adjustment Regressor in sklearn Compatible Design.
"""

from typing import Optional

import numpy as np
from sklearn.base import BaseEstimator, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import (
    check_is_fitted,
    validate_data,
)

from pgmpy.prediction._base import _BaseCausalPrediction


class NaiveAdjustmentRegressor(_BaseCausalPrediction):
    """
    Naive adjustment regressor using causal graph roles for feature selection.

    This estimator concatenates exposure, adjustment, and pretreatment variables
    as features to predict the outcome variable using standard ML algorithms.
    It's "naive" because it uses a simple prediction model with the adjustment
    set and doesn't employ sophisticated causal inference methods like double ML,
    inverse propensity weighting, or other advanced causal estimation techniques.

    Parameters
    ----------
    causal_graph : DAG, PDAG, ADMG, MAG, or PAG
        Causal graph with defined variable roles. Must have exactly one exposure
        and one outcome variable. The adjustment role is optional (can be missing,
        empty or contain variables).
    estimator : sklearn estimator, optional (default=LinearRegression())
        Base estimator for prediction.

    Attributes
    ----------
    `estimator_` : sklearn estimator
        The fitted base estimator.
    `feature_names_in_` : ndarray of shape (n_features,)
        Names of features seen during fit.
    `n_features_in_` : int
        Number of features seen during fit.
    `exposure_var_` : str
        Name of exposure variable extracted from causal graph.
    `adjustment_vars_` : list
        List of adjustment variable names extracted from causal graph.
    `pretreatment_vars_` : list
        List of pretreatment variable names extracted from causal graph.
    `outcome_var_` : str
        Name of outcome variable extracted from causal graph.
    `feature_columns_fit_` : list
        List of feature column names used (exposure + adjustment + pretreatment).
    `explanation_` : str
        Formatted description of the fitted model.

    Examples
    --------
    Basic usage with a simple causal DAG:

    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.base import DAG
    >>> from pgmpy.prediction import NaiveAdjustmentRegressor
    >>> from sklearn.linear_model import LinearRegression
    >>>
    >>> # Create a simple causal DAG: Z -> X, Z -> Y, X -> Y
    >>> # where Z is a confounder, X is exposure, Y is outcome
    >>> dag = DAG(
    ...     ebunch=[("Z", "X"), ("Z", "Y"), ("X", "Y")],
    ...     roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    ... )
    >>>
    >>> # Generate some synthetic data
    >>> np.random.seed(42)
    >>> n = 100
    >>> Z = np.random.normal(0, 1, n)
    >>> X = 0.5 * Z + np.random.normal(0, 0.5, n)
    >>> Y = 2.0 * X + 1.5 * Z + np.random.normal(0, 0.3, n)
    >>>
    >>> data = pd.DataFrame({"X": X, "Y": Y, "Z": Z})
    >>>
    >>> # Fit the regressor
    >>> regressor = NaiveAdjustmentRegressor(causal_graph=dag)
    >>> regressor.fit(data[["X", "Z"]], data["Y"])
    NaiveAdjustmentRegressor(...)
    >>>
    >>> # Make predictions
    >>> predictions = regressor.predict(data[["X", "Z"]])
    >>> print(f"Predictions shape: {predictions.shape}")
    Predictions shape: (100,)

    Using a custom estimator:

    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> # Use Random Forest as the base estimator
    >>> rf_regressor = NaiveAdjustmentRegressor(
    ...     causal_graph=dag,
    ...     estimator=RandomForestRegressor(n_estimators=10, random_state=42),
    ... )
    >>> rf_regressor.fit(data[["X", "Z"]], data["Y"])
    NaiveAdjustmentRegressor(...)

    Example with pretreatment variables:

    >>> # Create DAG with pretreatment variable P -> Y
    >>> dag_with_pretreatment = DAG(
    ...     ebunch=[("P", "Y"), ("Z", "X"), ("Z", "Y"), ("X", "Y")],
    ...     roles={
    ...         "exposure": "X",
    ...         "outcome": "Y",
    ...         "adjustment": ["Z"],
    ...         "pretreatment": ["P"],
    ...     },
    ... )
    >>>
    >>> # Generate data with proper relationships using simulate
    >>> lgbn_with_P = DAG.from_dagitty(
    ...     "dag { P -> Y [beta=0.8] Z -> X [beta=0.5] X -> Y [beta=2.0] Z -> Y [beta=1.5] }"
    ... )
    >>> data_with_P = lgbn_with_P.simulate(100, seed=42)
    >>>
    >>> regressor_with_P = NaiveAdjustmentRegressor(causal_graph=dag_with_pretreatment)
    >>> regressor_with_P.fit(data_with_P[["X", "Z", "P"]], data_with_P["Y"])
    NaiveAdjustmentRegressor(...)
    """

    def __init__(
        self,
        causal_graph,
        estimator: Optional[BaseEstimator] = None,
    ):
        self.causal_graph = causal_graph
        self.estimator = estimator

    def fit(
        self,
        X,
        y,
        sample_weight: Optional[np.ndarray] = None,
    ):
        """
        Fit the Naive Adjustment Regressor.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data. Column names must exactly match variable names in the causal graph.
            - If DataFrame: Column names must match DAG variable names exactly
            - If numpy array: Will be converted to DataFrame with columns [0, 1, 2, ...],
              so DAG should use integer variable names
        y : array-like of shape (n_samples,)
            Target values (outcome variable).
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights for training.

        Returns
        -------
        self : object
            Returns self for method chaining.
        """

        # Step 1: Validate input data
        validate_data(self, X, y, accept_sparse=False, ensure_2d=True, dtype="numeric")

        # Step 2: Extract and validate causal graph roles
        exposure_vars = self.causal_graph.get_role("exposure")
        outcome_vars = self.causal_graph.get_role("outcome")
        adjustment_vars = self.causal_graph.get_role("adjustment")
        pretreatment_vars = self.causal_graph.get_role("pretreatment")

        # Validate exactly one exposure and one outcome variable
        if len(exposure_vars) != 1:
            raise ValueError(
                f"Exactly one exposure variable must be defined. Found {len(exposure_vars)}: {exposure_vars}"
            )

        if len(outcome_vars) != 1:
            raise ValueError(
                f"Exactly one outcome variable must be defined. Found {len(outcome_vars)}: {outcome_vars}"
            )

        # Step 3: Store role variables as instance attributes
        self.exposure_var_ = exposure_vars[0]
        self.outcome_var_ = outcome_vars[0]
        self.adjustment_vars_ = adjustment_vars
        self.pretreatment_vars_ = pretreatment_vars
        self.feature_columns_fit_ = (
            [self.exposure_var_] + adjustment_vars + pretreatment_vars
        )

        # Step 4: Prepare feature DataFrame
        X_features = self._prepare_feature_df(
            X, required_features=self.feature_columns_fit_
        )

        # Step 5: Initialize base estimator
        self.estimator_ = (
            LinearRegression() if self.estimator is None else clone(self.estimator)
        )

        # Step 6: Fit the estimator
        self.estimator_.fit(X_features, y, sample_weight=sample_weight)

        # Step 7: Create explanation
        adj_str = ", ".join(map(str, adjustment_vars)) if adjustment_vars else "none"
        pre_str = (
            ", ".join(map(str, pretreatment_vars)) if pretreatment_vars else "none"
        )
        self.explanation_ = (
            f"NaiveAdjustmentRegressor(exposure={self.exposure_var_}, outcome={self.outcome_var_}, "
            f"adjustment=[{adj_str}], pretreatment=[{pre_str}], "
            f"estimator={type(self.estimator_).__name__})"
        )

        return self

    def predict(self, X):
        """Make predictions using the fitted regressor.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Input data. Column names must exactly match variable names in the causal graph.
            - If DataFrame: Column names must match DAG variable names exactly
            - If numpy array: Will be converted to DataFrame with columns [0, 1, 2, ...],
              so DAG should use integer variable names

        Returns
        -------
        predictions : ndarray of shape (n_samples,)
            Predicted values.
        """
        # Step 1: Validate that estimator is fitted
        check_is_fitted(self, "estimator_")

        validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype="numeric",
            reset=False,
        )
        X_filtered = self._prepare_feature_df(
            X, required_features=self.feature_columns_fit_
        )

        # Step 2: Make predictions and return as 1D array
        predictions = self.estimator_.predict(X_filtered)
        return np.asarray(predictions).ravel()

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        check_is_fitted(self, "estimator_")
        return np.array(self.feature_columns_fit_, dtype=str)
