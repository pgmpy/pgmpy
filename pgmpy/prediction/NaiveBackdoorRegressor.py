"""
Naive Backdoor Regressor in sklearn Compatible Design.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

try:
    from sklearn.utils.validation import validate_data
except ImportError:
    validate_data = None


class NaiveBackdoorRegressor(RegressorMixin, BaseEstimator):
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
        Base sklearn estimator for prediction.

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
    `explanation_` : str
        Formatted description of the fitted model.
    """

    def __init__(
        self,
        causal_graph,
        base_estimator: Optional[BaseEstimator] = None,
    ):
        self.causal_graph = causal_graph
        self.base_estimator = base_estimator

        # Cache roles during init to avoid DAG mutation during fit
        # Note: This is a workaround for pgmpy's DAG methods that mutate internal state
        # For sklearn compatibility, we delay validation until fit() is called
        try:
            self._cached_roles = self._extract_roles_safely(causal_graph)
        except (TypeError, AttributeError, ValueError):
            # Allow invalid parameters during init for sklearn compatibility tests
            self._cached_roles = None

    def set_params(self, **params):
        """Set parameters and re-cache roles if causal_graph changes."""
        result = super().set_params(**params)

        if "causal_graph" in params:
            try:
                self._cached_roles = self._extract_roles_safely(self.causal_graph)
            except (TypeError, AttributeError, ValueError):
                # If the new causal_graph is invalid, we'll catch it during validation
                # This allows sklearn's parameter validation tests to work
                self._cached_roles = None

        return result

    def _extract_roles_safely(self, dag):
        """Extract roles from DAG during initialization."""
        if not hasattr(dag, "get_role"):
            raise TypeError(
                f"causal_graph must have 'get_role' method, got {type(dag)}"
            )

        exposure_vars = list(dag.get_role("exposure"))
        outcome_vars = list(dag.get_role("outcome"))
        adjustment_vars = list(dag.get_role("adjustment"))

        pretreatment_vars = list(
            dag.get_role("pretreatment") if dag.has_role("pretreatment") else []
        )

        return {
            "exposure": exposure_vars,
            "outcome": outcome_vars,
            "adjustment": adjustment_vars,
            "pretreatment": pretreatment_vars,
        }

    def __sklearn_tags__(self):
        """Tags for sklearn compatibility."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        tags.regressor_tags.poor_score = True
        return tags

    def _validate_dag_and_extract_roles(self):
        """Validate causal graph has required roles and extract variable assignments."""
        if self._cached_roles is None:
            self._cached_roles = self._extract_roles_safely(self.causal_graph)

        cached_roles = self._cached_roles

        exposure_vars = cached_roles["exposure"]
        outcome_vars = cached_roles["outcome"]
        adjustment_vars = cached_roles["adjustment"]
        pretreatment_vars = cached_roles["pretreatment"]

        # Validation for single exposure/outcome
        if len(exposure_vars) != 1:
            raise ValueError(
                f"Exactly one exposure variable must be defined. Found {len(exposure_vars)}: {exposure_vars}"
            )

        if len(outcome_vars) != 1:
            raise ValueError(
                f"Exactly one outcome variable must be defined. Found {len(outcome_vars)}: {outcome_vars}"
            )

        return exposure_vars[0], outcome_vars[0], adjustment_vars, pretreatment_vars

    def _check_feature_names(self, X, reset):
        """Validate feature names for sklearn compatibility."""
        # This method can be used for additional feature name validation
        # Currently using basic sklearn validation in _validate_data
        pass

    def _ensure_dataframe(self, X, feature_names=None) -> pd.DataFrame:
        """Convert input to DataFrame, generating generic names if needed."""
        if isinstance(X, pd.DataFrame):
            return X.copy()

        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        if feature_names is None:
            # Generate generic names for array inputs, e.g., from sklearn tests
            feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        return pd.DataFrame(X_arr, columns=feature_names)

    def _prepare_feature_df(self, X, feature_names=None) -> pd.DataFrame:
        """
        Ensures input X is a DataFrame with columns matching the causal graph's roles.

        This method handles three cases:
        1. X is a pandas DataFrame: It selects the required feature columns based on names.
        2. X is a NumPy array with `feature_names`: It converts X to a DataFrame
           and selects the required feature columns based on names.
        3. X is a NumPy array without `feature_names` (sklearn compatibility case):
           It assumes the first N columns correspond to the N required features
           and renames them. Workaround for sklearn's
           generic test suite, which does not support named features.
        """
        if hasattr(self, "feature_columns_"):
            required_features = self.feature_columns_
        else:
            exposure_var, _, adjustment_vars, pretreatment_vars = (
                self._validate_dag_and_extract_roles()
            )
            required_features = [exposure_var] + adjustment_vars + pretreatment_vars
            self.feature_columns_ = required_features

        X_df = self._ensure_dataframe(X, feature_names)

        # Case 3: Handle sklearn compatibility for generic inputs
        # Check for feature_0, feature_1, etc. OR integer column names 0, 1, 2, etc.
        is_sklearn_generic_input = all(
            str(col).startswith("feature_") for col in X_df.columns
        ) or all(isinstance(col, (int, np.integer)) for col in X_df.columns)
        if is_sklearn_generic_input:
            if len(X_df.columns) < len(required_features):
                raise ValueError(
                    f"Input has {len(X_df.columns)} features, but the causal model "
                    f"requires {len(required_features)}: {required_features}"
                )
            # Select the first N columns and rename them to match the semantic roles.
            # Workaround for sklearn's check_estimator.
            feature_df = X_df.iloc[:, : len(required_features)].copy()
            feature_df.columns = required_features
            return feature_df

        # Cases 1 & 2: Standard semantic mapping from named columns
        missing_columns = set(required_features) - set(X_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in input data: {list(missing_columns)}. "
                f"Required columns based on DAG roles: {required_features}."
            )

        return X_df[required_features]

    def fit(
        self,
        X,
        y,
        sample_weight: Optional[np.ndarray] = None,
        feature_names=None,
        use_feature_names_out=True,
    ):
        """
        Fit the Naive Backdoor Regressor.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features) or pandas DataFrame
            Training data. If DataFrame, must include columns for exposure,
            adjustment, and pretreatment variables as defined in causal graph roles.
            If array, `feature_names` must be provided unless used within a
            generic sklearn context (like compatibility tests).
        y : array-like of shape (n_samples,)
            Target values (outcome variable).
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights.
        feature_names : list, optional
            Feature names when X is an array.
        use_feature_names_out : bool, optional (default=True)
            If True, feature_names_in_ are set and used for validation.

        Returns
        -------
        self
            Returns self for method chaining.
        """
        if validate_data is not None:
            X_arr, y_arr = validate_data(
                self, X, y, accept_sparse=False, ensure_2d=True, dtype="numeric"
            )
        else:
            # Fallback for older sklearn versions
            X_arr, y_arr = check_X_y(
                X, y, accept_sparse=False, ensure_2d=True, dtype="numeric"
            )

        # Extract and validate causal graph roles and define required features
        exposure_var, outcome_var, adjustment_vars, pretreatment_vars = (
            self._validate_dag_and_extract_roles()
        )
        self.exposure_var_ = exposure_var
        self.outcome_var_ = outcome_var
        self.adjustment_vars_ = adjustment_vars
        self.pretreatment_vars_ = pretreatment_vars
        self.feature_columns_ = [exposure_var] + adjustment_vars + pretreatment_vars

        X_features = self._prepare_feature_df(X, feature_names)

        # self.n_features_in_ = X_features.shape[1]
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = np.array(X_features.columns, dtype=object)

        # Initialize base estimator
        self.estimator_ = (
            LinearRegression()
            if self.base_estimator is None
            else clone(self.base_estimator)
        )

        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self.estimator_.fit(X_features, y_arr, **fit_params)

        # Explanation attribute
        exposure_var, outcome_var, adjustment_vars, pretreatment_vars = (
            self._validate_dag_and_extract_roles()
        )
        adj_str = ", ".join(adjustment_vars) if adjustment_vars else "none"
        pre_str = ", ".join(pretreatment_vars) if pretreatment_vars else "none"
        self.explanation_ = (
            f"NaiveBackdoorRegressor(exposure={exposure_var}, outcome={outcome_var}, "
            f"adjustment=[{adj_str}], pretreatment=[{pre_str}], "
            f"estimator={type(self.estimator_).__name__})"
        )

        return self

    def predict(self, X, feature_names=None):
        """Make predictions using the fitted regressor."""
        check_is_fitted(self, "estimator_")

        if validate_data is not None:
            X = validate_data(
                self,
                X,
                accept_sparse=False,
                ensure_2d=True,
                dtype="numeric",
                reset=False,
            )
        else:
            # Fallback for older sklearn versions
            X = check_array(X, accept_sparse=False, ensure_2d=True, dtype="numeric")

        X_features = self._prepare_feature_df(X, feature_names)

        predictions = self.estimator_.predict(X_features)
        return np.asarray(predictions).ravel()

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        check_is_fitted(self, "estimator_")
        return np.array(self.feature_columns_, dtype=str)
