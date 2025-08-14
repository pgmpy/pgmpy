"""
Naive Backdoor Regressor in sklearn Compatible Design.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted, check_X_y

from pgmpy.base import DAG


class NaiveBackdoorRegressor(BaseEstimator, RegressorMixin):
    """
    A naive backdoor regressor that uses causal graph roles for feature selection.

    This estimator combines exposure and adjustment variables as features to predict
    the outcome variable. The approach is "naive" because it assumes the adjustment
    set is already correctly identified in the DAG roles and simply concatenates
    exposure + adjustment variables as features for standard ML prediction.

    Parameters
    ----------
    dag : DAG
        A pgmpy DAG object with defined roles for exposure, outcome, and optionally
        adjustment variables. Must have exactly one exposure and one outcome variable.
    base_estimator : sklearn estimator, optional (default=LinearRegression())
        The base sklearn estimator to use for prediction.

    Attributes
    ----------
    `estimator_` : sklearn estimator
        The fitted base estimator.
    `feature_names_in_` : ndarray of shape (n_features,)
        Names of features seen during fit.
    `n_features_in_` : int
        Number of features seen during fit.
    `exposure_var_` : str
        Name of exposure variable extracted from DAG.
    `adjustment_vars_` : list
        List of adjustment variable names extracted from DAG.
    `outcome_var_` : str
        Name of outcome variable extracted from DAG.
    `feature_columns_` : list
        List of feature column names used (exposure + adjustment).

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> # Create DAG with roles
    >>> dag = DAG(
    ...     [("Z", "X"), ("Z", "Y"), ("X", "Y")],
    ...     roles={"exposure": "X", "outcome": "Y", "adjustment": "Z"},
    ... )
    >>>
    >>> regressor = NaiveBackdoorRegressor(
    ...     dag=dag, base_estimator=RandomForestRegressor()
    ... )
    >>> regressor.fit(X_train, y_train)
    >>> predictions = regressor.predict(X_test)
    """

    def __init__(self, dag: DAG, base_estimator: Optional[BaseEstimator] = None):
        self.dag = dag
        self.base_estimator = base_estimator

    def _more_tags(self):
        """Additional tags for sklearn compatibility."""
        return {
            "requires_y": True,
            "no_sparse_input": True,
            "requires_positive_X": False,
        }

    def _validate_dag_and_extract_roles(self):
        """
        Validate DAG has required roles and extract variable assignments.

        Returns
        -------
        tuple
            (exposure_var, outcome_var, adjustment_vars)

        Raises
        ------
        ValueError
            If DAG doesn't have required roles or has invalid structure.
        """
        # Validate that DAG has causal structure (exposure and outcome roles)
        self.dag.is_valid_causal_structure()

        # Extract roles
        exposure_vars = self.dag.get_role("exposure")
        outcome_vars = self.dag.get_role("outcome")
        adjustment_vars = (
            self.dag.get_role("adjustment") if self.dag.has_role("adjustment") else []
        )

        # Enforce single exposure and outcome constraint
        if len(exposure_vars) != 1:
            raise ValueError(
                f"Exactly one exposure variable must be defined. "
                f"Found {len(exposure_vars)}: {exposure_vars}"
            )

        if len(outcome_vars) != 1:
            raise ValueError(
                f"Exactly one outcome variable must be defined. "
                f"Found {len(outcome_vars)}: {outcome_vars}"
            )

        return exposure_vars[0], outcome_vars[0], adjustment_vars

    def _extract_feature_columns(self):
        """
        Extract feature column names (exposure + adjustment variables).

        Returns
        -------
        list
            List of column names to use as features.
        """
        exposure_var, _, adjustment_vars = self._validate_dag_and_extract_roles()

        # Combine exposure and adjustment variables as features
        feature_columns = [exposure_var] + adjustment_vars
        return feature_columns

    def _validate_data_columns(self, X_df: pd.DataFrame):
        """
        Validate that required columns exist in the data.

        Parameters
        ----------
        X_df : pd.DataFrame
            Input data as DataFrame.

        Raises
        ------
        ValueError
            If required columns are missing from the data.
        """
        required_columns = self._extract_feature_columns()
        missing_columns = [col for col in required_columns if col not in X_df.columns]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in input data: {missing_columns}. "
                f"Required columns based on DAG roles: {required_columns}. "
                f"Available columns: {list(X_df.columns)}"
            )

    def _ensure_dataframe(self, X) -> pd.DataFrame:
        """
        Convert input to DataFrame with proper column names.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        pd.DataFrame
            DataFrame with proper column names.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()

        # Convert numpy array to DataFrame
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # Generate column names matching expected feature count
        required_features = self._extract_feature_columns()
        if X_arr.shape[1] != len(required_features):
            raise ValueError(
                f"Input data has {X_arr.shape[1]} columns but DAG roles require "
                f"{len(required_features)} features: {required_features}"
            )

        return pd.DataFrame(X_arr, columns=required_features)

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        """
        Fit the Naive Backdoor Regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            Training data containing all variables. If DataFrame, must include
            columns for exposure and adjustment variables as defined in DAG roles.
            If array, assumes columns are ordered as [exposure, adjustment_vars...].
        y : array-like of shape (n_samples,)
            Target values (outcome variable).
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        # Validate inputs
        X_arr, y_arr = check_X_y(
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            force_all_finite=True,
            dtype="numeric",
        )

        # Extract and validate DAG roles
        exposure_var, outcome_var, adjustment_vars = (
            self._validate_dag_and_extract_roles()
        )

        # Store roles
        self.exposure_var_ = exposure_var
        self.outcome_var_ = outcome_var
        self.adjustment_vars_ = adjustment_vars

        # Convert to DataFrame and validate columns exist
        X_df = self._ensure_dataframe(X)
        self._validate_data_columns(X_df)

        # Store sklearn attributes
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)

        # Extract feature columns (exposure + adjustment)
        feature_columns = self._extract_feature_columns()
        X_features = X_df[feature_columns]

        # Store feature columns for prediction
        self.feature_columns_ = feature_columns

        # Initialize base estimator if not provided
        if self.base_estimator is None:
            self.estimator_ = LinearRegression()
        else:
            self.estimator_ = clone(self.base_estimator)

        # Fit the base estimator on selected features
        # fit(exposure + adjustment, outcome)
        if sample_weight is not None:
            self.estimator_.fit(X_features, y_arr, sample_weight=sample_weight)
        else:
            self.estimator_.fit(X_features, y_arr)

        return self

    def predict(self, X):
        """
        Make predictions using the fitted regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            Input data for prediction. Must have same structure as training data.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "estimator_")

        # Convert to DataFrame and validate
        X_df = self._ensure_dataframe(X)
        self._validate_data_columns(X_df)

        # Extract the same feature columns used during fit
        X_features = X_df[self.feature_columns_]

        # Make predictions using the fitted base estimator
        predictions = self.estimator_.predict(X_features)
        return np.asarray(predictions).ravel()

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.

        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Not used, present for API consistency.

        Returns
        -------
        ndarray of shape (n_features_out,), dtype=str
            Feature names used by the estimator.
        """
        check_is_fitted(self, "estimator_")
        return np.array(self.feature_columns_, dtype=str)
