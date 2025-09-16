"""
Naive Adjustment Regressor in sklearn Compatible Design.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import (  # check_array,; check_X_y,
    check_is_fitted,
    validate_data,
)


class NaiveAdjustmentRegressor(RegressorMixin, BaseEstimator):
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
        and one outcome variable. The adjustment role must be defined (can be empty).
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
    `feature_columns_` : list
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

    >>> # Create DAG with pretreatment variable P
    >>> dag_with_pretreatment = DAG(
    ...     ebunch=[("P", "X"), ("Z", "X"), ("Z", "Y"), ("X", "Y")],
    ...     roles={
    ...         "exposure": "X",
    ...         "outcome": "Y",
    ...         "adjustment": ["Z"],
    ...         "pretreatment": ["P"],
    ...     },
    ... )
    >>>
    >>> # Add pretreatment variable to data
    >>> data_with_P = data.copy()
    >>> data_with_P["P"] = np.random.normal(0, 1, n)
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

        # Cache roles during init to avoid DAG mutation during fit.
        # This is needed because pgmpy's DAG methods can mutate internal state,
        # and sklearn compatibility requires parameter validation to be deferred until fit().
        try:
            self._cached_roles = self._extract_roles_safely(causal_graph)
        except (TypeError, AttributeError, ValueError):
            # Allow invalid parameters during init for sklearn compatibility tests
            self._cached_roles = None

    def set_params(self, **params):
        """Set parameters and re-cache roles if causal_graph changes."""
        result = super().set_params(**params)

        # Re-cache roles if causal_graph parameter was updated
        if "causal_graph" in params:
            try:
                self._cached_roles = self._extract_roles_safely(self.causal_graph)
            except (TypeError, AttributeError, ValueError):
                # Invalid graph will be caught during fit() validation
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
        # Step 1: Extract roles from DAG (use cached if available)
        if self._cached_roles is None:
            self._cached_roles = self._extract_roles_safely(self.causal_graph)

        cached_roles = self._cached_roles
        exposure_vars = cached_roles["exposure"]
        outcome_vars = cached_roles["outcome"]
        adjustment_vars = cached_roles["adjustment"]
        pretreatment_vars = cached_roles["pretreatment"]

        # Step 2: Validate exactly one exposure and one outcome variable
        if len(exposure_vars) != 1:
            raise ValueError(
                f"Exactly one exposure variable must be defined. Found {len(exposure_vars)}: {exposure_vars}"
            )

        if len(outcome_vars) != 1:
            raise ValueError(
                f"Exactly one outcome variable must be defined. Found {len(outcome_vars)}: {outcome_vars}"
            )

        return exposure_vars[0], outcome_vars[0], adjustment_vars, pretreatment_vars

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
        Convert input to DataFrame with columns matching causal graph roles.

        Handles three cases:
        1. DataFrame input: Select required columns by name
        2. Array with `feature_names`: Convert to DataFrame and select by name
        3. Array without `feature_names` (sklearn tests): Map first N columns to roles
        """
        # Step 1: Get required feature columns (set during fit)
        if hasattr(self, "feature_columns_"):
            required_features = self.feature_columns_
        else:
            # This branch is for predict() called before fit()
            exposure_var, _, adjustment_vars, pretreatment_vars = (
                self._validate_dag_and_extract_roles()
            )
            required_features = [exposure_var] + adjustment_vars + pretreatment_vars

        # Step 2: Convert input to DataFrame format
        X_df = self._ensure_dataframe(X, feature_names)

        # Step 3: Handle sklearn compatibility for generic feature names
        has_generic_names = all(str(col).startswith("feature_") for col in X_df.columns)
        has_numeric_names = all(
            isinstance(col, (int, np.integer)) for col in X_df.columns
        )

        if has_generic_names or has_numeric_names:
            # Step 3a: sklearn test case - map first N columns to semantic role names
            if len(X_df.columns) < len(required_features):
                raise ValueError(
                    f"Input has {len(X_df.columns)} features, but the causal model "
                    f"requires {len(required_features)}: {required_features}"
                )
            feature_df = X_df.iloc[:, : len(required_features)].copy()
            feature_df.columns = required_features
            return feature_df

        # Step 4: Standard case - map named columns to causal roles
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
    ):
        """
        Fit the Naive Backdoor Regressor.

        Parameters
        ----------
        X : array-like or DataFrame of shape (n_samples, n_features)
            Training data. If DataFrame, must include columns for exposure,
            adjustment, and pretreatment variables as defined in causal graph.
            If array, `feature_names` should be provided (except for sklearn tests).
        y : array-like of shape (n_samples,)
            Target values (outcome variable).
        sample_weight : array-like of shape (n_samples,), optional
            Sample weights for training.
        feature_names : list, optional
            Feature names when X is an array.

        Returns
        -------
        self : object
            Returns self for method chaining.
        """
        # Step 1: Validate input data using sklearn utilities
        X_arr, y_arr = validate_data(
            self, X, y, accept_sparse=False, ensure_2d=True, dtype="numeric"
        )

        # Step 2: Extract and validate causal graph roles
        exposure_var, outcome_var, adjustment_vars, pretreatment_vars = (
            self._validate_dag_and_extract_roles()
        )

        # Step 3: Store role variables as instance attributes
        self.exposure_var_ = exposure_var
        self.outcome_var_ = outcome_var
        self.adjustment_vars_ = adjustment_vars
        self.pretreatment_vars_ = pretreatment_vars
        self.feature_columns_ = [exposure_var] + adjustment_vars + pretreatment_vars

        # Step 4: Prepare feature DataFrame from input data
        X_features = self._prepare_feature_df(X, feature_names)

        # Step 5: Set sklearn-required attributes
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = np.array(X_features.columns, dtype=object)

        # Step 6: Initialize and configure base estimator
        self.estimator_ = (
            LinearRegression() if self.estimator is None else clone(self.estimator)
        )

        # Step 7: Prepare fitting parameters and fit the estimator
        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight
        self.estimator_.fit(X_features, y_arr, **fit_params)

        # Step 8: Create readable explanation
        adj_str = ", ".join(adjustment_vars) if adjustment_vars else "none"
        pre_str = ", ".join(pretreatment_vars) if pretreatment_vars else "none"
        self.explanation_ = (
            f"NaiveAdjustmentRegressor(exposure={exposure_var}, outcome={outcome_var}, "
            f"adjustment=[{adj_str}], pretreatment=[{pre_str}], "
            f"estimator={type(self.estimator_).__name__})"
        )

        return self

    def predict(self, X, feature_names=None):
        """Make predictions using the fitted regressor."""
        # Step 1: Validate that estimator is fitted
        check_is_fitted(self, "estimator_")

        # Step 2: Validate input data using sklearn utilities
        X = validate_data(
            self,
            X,
            accept_sparse=False,
            ensure_2d=True,
            dtype="numeric",
            reset=False,
        )

        # Step 3: Prepare feature DataFrame with causal graph roles
        X_features = self._prepare_feature_df(X, feature_names)

        # Step 4: Make predictions and return as 1D array
        predictions = self.estimator_.predict(X_features)
        return np.asarray(predictions).ravel()

    def get_feature_names_out(self, input_features=None):
        """Get output feature names for transformation."""
        check_is_fitted(self, "estimator_")
        return np.array(self.feature_columns_, dtype=str)
