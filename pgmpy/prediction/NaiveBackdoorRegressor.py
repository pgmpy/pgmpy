"""
Naive Backdoor Regressor in sklearn Compatible Design.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted, check_X_y


class NaiveBackdoorRegressor(BaseEstimator, RegressorMixin):
    """
    A naive backdoor regressor that uses causal graph roles for feature selection.

    This estimator combines exposure, adjustment, and pretreatment variables as features
    to predict the outcome variable. The approach is "naive" because it assumes the
    adjustment set is already correctly identified in the causal graph roles and simply
    concatenates these variables as features for standard ML prediction.

    Parameters
    ----------
    causal_graph : DAG, PDAG, or ADMG
        A pgmpy causal graph object (DAG, PDAG, or ADMG) with defined roles for
        exposure, outcome, and adjustment variables. Must have exactly one exposure
        and one outcome variable. The 'adjustment' role must be explicitly defined
        even if empty to prevent accidental omission of confounders.
    base_estimator : sklearn estimator, optional (default=LinearRegression())
        The base sklearn estimator to use for prediction.
    multi_output : bool, optional (default=False)
        Reserved for future multiple outcome support. Currently must be False.

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
    `feature_columns_` : list
        List of feature column names used (exposure + adjustment + pretreatment).

    Examples
    --------
    >>> from pgmpy.base import DAG
    >>> from sklearn.ensemble import RandomForestRegressor
    >>>
    >>> # Create DAG with roles
    >>> dag = DAG(
    ...     [("Z", "X"), ("Z", "Y"), ("X", "Y")],
    ...     roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    ... )
    >>>
    >>> regressor = NaiveBackdoorRegressor(
    ...     causal_graph=dag, base_estimator=RandomForestRegressor()
    ... )
    >>> regressor.fit(X_train, y_train)
    >>> predictions = regressor.predict(X_test)
    """

    def __init__(
        self,
        causal_graph,
        base_estimator: Optional[BaseEstimator] = None,
        multi_output=False,
    ):
        self.causal_graph = causal_graph
        self.base_estimator = base_estimator
        self.multi_output = multi_output

        # Store the original roles specification to validate adjustment was explicitly defined
        self._original_roles = getattr(causal_graph, "_original_roles", None)

        if multi_output:
            raise NotImplementedError(
                "Multiple outcome support is planned for future releases. "
                "Currently only single outcome variables are supported."
            )

    def __sklearn_tags__(self):
        """Tags for sklearn compatibility."""
        tags = super().__sklearn_tags__()
        # Input requirements
        tags.input_tags.sparse = False  # no sparse input
        tags.input_tags.positive_only = False  # don't require positive X
        # Target requirements
        tags.target_tags.required = True  # requires y
        tags.target_tags.single_output = True  # single output only
        tags.target_tags.multi_output = False  # no multi-output support
        return tags

    def _more_tags(self):
        """Additional tags for sklearn compatibility (backward compatibility)."""
        return {
            "requires_y": True,
            "no_sparse_input": True,
            "requires_positive_X": False,
        }

    def _validate_dag_and_extract_roles(self):
        """
        Validate causal graph has required roles and extract variable assignments.

        Returns
        -------
        tuple
            (exposure_var, outcome_var, adjustment_vars, pretreatment_vars)

        Raises
        ------
        ValueError
            If causal graph doesn't have required roles or has invalid structure.
        """
        # Validate that causal graph has causal structure
        self.causal_graph.is_valid_causal_structure()

        # Extract roles
        exposure_vars = self.causal_graph.get_role("exposure")
        outcome_vars = self.causal_graph.get_role("outcome")

        # Check if adjustment role was explicitly defined
        if not self.causal_graph.has_role("adjustment"):
            raise ValueError(
                "The 'adjustment' role must be explicitly defined in the causal graph, "
                "even if no adjustment variables are needed. Use an empty list [] "
                "to indicate no adjustment variables are required. This prevents "
                "accidental omission of confounders in causal effect estimation."
            )

        # Get adjustment and pretreatment variables (may be empty)
        adjustment_vars = self.causal_graph.get_role("adjustment")
        pretreatment_vars = (
            self.causal_graph.get_role("pretreatment")
            if self.causal_graph.has_role("pretreatment")
            else []
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

        return exposure_vars[0], outcome_vars[0], adjustment_vars, pretreatment_vars

    def _extract_feature_columns(self):
        """
        Extract feature column names (exposure + adjustment + pretreatment variables).

        Returns
        -------
        list
            List of column names to use as features.
        """
        exposure_var, _, adjustment_vars, pretreatment_vars = (
            self._validate_dag_and_extract_roles()
        )

        # Combine exposure, adjustment, and pretreatment variables as features
        feature_columns = [exposure_var] + adjustment_vars + pretreatment_vars
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

    def _ensure_dataframe(self, X, feature_names=None) -> pd.DataFrame:
        """
        Convert input to DataFrame with proper column names.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.
        feature_names : list, optional
            Explicit feature names when X is an array. Required for array input.

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

        # For arrays, require explicit feature names
        if feature_names is None:
            raise ValueError(
                "When passing array input to NaiveBackdoorRegressor, you must provide "
                "explicit feature names via feature_names parameter to ensure correct "
                "mapping to causal graph variables. This prevents silent errors in "
                "causal effect estimation."
            )

        required_features = self._extract_feature_columns()
        if len(feature_names) != len(required_features):
            raise ValueError(
                f"feature_names has {len(feature_names)} elements but causal graph roles require "
                f"{len(required_features)} features: {required_features}"
            )

        if X_arr.shape[1] != len(feature_names):
            raise ValueError(
                f"Input data has {X_arr.shape[1]} columns but feature_names specifies "
                f"{len(feature_names)} features: {feature_names}"
            )

        return pd.DataFrame(X_arr, columns=feature_names)

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None, feature_names=None):
        """
        Fit the Naive Backdoor Regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            Training data containing all variables. If DataFrame, must include
            columns for exposure, adjustment, and pretreatment variables as defined
            in causal graph roles. If array, feature_names must be provided.
        y : array-like of shape (n_samples,)
            Target values (outcome variable).
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        feature_names : list, optional
            Feature names when X is an array. Required for array input.

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
            ensure_all_finite=True,
            dtype="numeric",
        )

        # Extract and validate causal graph roles
        exposure_var, outcome_var, adjustment_vars, pretreatment_vars = (
            self._validate_dag_and_extract_roles()
        )

        # Store roles
        self.exposure_var_ = exposure_var
        self.outcome_var_ = outcome_var
        self.adjustment_vars_ = adjustment_vars
        self.pretreatment_vars_ = pretreatment_vars

        # Convert to DataFrame and validate columns exist
        X_df = self._ensure_dataframe(X, feature_names=feature_names)
        self._validate_data_columns(X_df)

        # Store sklearn attributes
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = np.array(X_df.columns, dtype=object)

        # Extract feature columns (exposure + adjustment + pretreatment)
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

    def predict(self, X, feature_names=None):
        """
        Make predictions using the fitted regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            Input data for prediction. Must have same structure as training data.
        feature_names : list, optional
            Feature names when X is an array. Required for array input.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted values.
        """
        check_is_fitted(self, "estimator_")

        # Convert to DataFrame and validate
        X_df = self._ensure_dataframe(X, feature_names=feature_names)
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
