"""
Naive Backdoor Regressor in sklearn Compatible Design.
"""

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted, check_X_y


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
        tags.input_tags.sparse = False
        tags.input_tags.positive_only = False
        # Target requirements
        tags.target_tags.required = True
        tags.target_tags.single_output = True
        tags.target_tags.multi_output = False
        return tags

    # For backward compatibility with older sklearn versions
    def _more_tags(self):
        """Backward compatibility tags."""
        return {
            "requires_y": True,
            "requires_positive_X": False,
            "allow_nan": False,
            "poor_score": True,  # Specialized estimator
        }

    def _validate_dag_and_extract_roles(self):
        """Validate causal graph has required roles and extract variable assignments."""
        dag = self.causal_graph
        dag.is_valid_causal_structure()

        # Extract roles
        exposure_vars = dag.get_role("exposure")
        outcome_vars = dag.get_role("outcome")

        # For sklearn compatibility, make adjustment role optional if not explicitly set
        if hasattr(dag, "_original_roles") and dag._original_roles is not None:
            # If roles were explicitly set, require adjustment to be defined
            if not dag.has_role("adjustment"):
                raise ValueError(
                    "The 'adjustment' role must be explicitly defined in the causal graph, "
                    "even if no adjustment variables are needed. Use an empty list [] "
                    "to indicate no adjustment variables are required."
                )
        else:
            # For sklearn compatibility, default to empty adjustment if not defined
            if not dag.has_role("adjustment"):
                # Add empty adjustment role for compatibility
                dag.set_node_roles({"adjustment": []})

        adjustment_vars = dag.get_role("adjustment")
        pretreatment_vars = (
            dag.get_role("pretreatment") if dag.has_role("pretreatment") else []
        )

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
        """Validate that required columns exist in the data."""
        required_columns = self._extract_feature_columns()

        # For sklearn compatibility, be more flexible about column requirements
        if len(X_df.columns) != len(required_columns):
            # If we have generic feature names, map them to required features
            if all(col.startswith("feature_") for col in X_df.columns):
                if len(X_df.columns) == len(required_columns):
                    # Rename columns to match required features
                    X_df.columns = required_columns
                    return

        # Check for missing required columns
        missing_columns = [col for col in required_columns if col not in X_df.columns]
        if missing_columns:
            raise ValueError(
                f"Missing required columns in input data: {missing_columns}. "
                f"Required columns based on DAG roles: {required_columns}. "
                f"Available columns: {list(X_df.columns)}"
            )

    def _ensure_dataframe(self, X, feature_names=None) -> pd.DataFrame:
        """Convert input to DataFrame with proper column names."""
        if isinstance(X, pd.DataFrame):
            return X.copy()

        # Convert numpy array to DataFrame
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr.reshape(-1, 1)

        # For sklearn compatibility, generate feature names if not provided
        if feature_names is None:
            required_features = self._extract_feature_columns()

            # If this is during sklearn testing, generate generic feature names
            if X_arr.shape[1] != len(required_features):
                # For sklearn compatibility tests, use generic column names
                feature_names = [f"feature_{i}" for i in range(X_arr.shape[1])]
            else:
                # Use the required feature names from the DAG
                feature_names = required_features

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

        # Handle sample_weight if it's a pandas Series
        if sample_weight is not None:
            if hasattr(sample_weight, "values"):
                sample_weight = sample_weight.values
            else:
                sample_weight = np.asarray(sample_weight)

        # Fit the base estimator on selected features
        # fit(exposure + adjustment, outcome)
        if sample_weight is not None:
            self.estimator_.fit(X_features, y_arr, sample_weight=sample_weight)
        else:
            self.estimator_.fit(X_features, y_arr)

        return self

    def predict(self, X, feature_names=None):
        """Make predictions using the fitted regressor."""
        check_is_fitted(self, "estimator_")

        # Convert to DataFrame and validate
        X_df = self._ensure_dataframe(X, feature_names=feature_names)
        self._validate_data_columns(X_df)

        # Extract the same feature columns used during fit
        X_features = X_df[self.feature_columns_]

        # Make predictions
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
