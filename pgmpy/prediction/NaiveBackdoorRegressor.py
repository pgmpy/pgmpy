"""
Naive Backdoor Regressor in sklearn Compatible Design.
"""

from typing import List, Optional

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
    >>> import pandas as pd
    >>>
    >>> # Create DAG with roles
    >>> dag = DAG(
    ...     [("Z", "X"), ("Z", "Y"), ("X", "Y")],
    ...     roles={"exposure": "X", "outcome": "Y", "adjustment": ["Z"]},
    ... )
    >>>
    >>> # Create dummy data
    >>> data = pd.DataFrame({"X": [1, 2, 3], "Z": [4, 5, 6], "Y": [7, 8, 9]})
    >>> X_train, y_train = data[["X", "Z"]], data["Y"]
    >>>
    >>> regressor = NaiveBackdoorRegressor(
    ...     causal_graph=dag, base_estimator=RandomForestRegressor()
    ... )
    >>> regressor.fit(X_train, y_train)
    >>> predictions = regressor.predict(X_train)
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

        if multi_output:
            raise NotImplementedError(
                "Multiple outcome support is planned for future releases. "
                "Currently only single outcome variables are supported."
            )

    def _more_tags(self):
        """Tags for sklearn compatibility."""
        # This method is used by sklearn's check_estimator to understand the
        # capabilities of the estimator.
        return {
            "requires_y": True,
            "allow_nan": False,
        }

    def _validate_dag_and_extract_roles(self):
        """Validate causal graph has required roles and extract variable assignments."""
        dag = self.causal_graph
        dag.is_valid_causal_structure()

        # Extract roles
        exposure_vars = dag.get_role("exposure")
        outcome_vars = dag.get_role("outcome")

        if not dag.has_role("adjustment"):
            raise ValueError(
                "The 'adjustment' role must be explicitly defined in the causal graph, "
                "even if no adjustment variables are needed. Use an empty list [] "
                "to indicate no adjustment variables are required."
            )

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

    def _extract_feature_columns(self) -> List[str]:
        """
        Extract and store the required feature column names from the causal graph.
        """
        if hasattr(self, "feature_columns_"):
            return self.feature_columns_

        exposure_var, _, adjustment_vars, pretreatment_vars = (
            self._validate_dag_and_extract_roles()
        )
        # Combine and store
        self.feature_columns_ = [exposure_var] + adjustment_vars + pretreatment_vars
        return self.feature_columns_

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
           and renames them. This is a pragmatic workaround for sklearn's
           generic test suite, which does not support named features.
        """
        required_features = self._extract_feature_columns()
        X_df = self._ensure_dataframe(X, feature_names)

        # Case 3: Handle sklearn compatibility for generic inputs (e.g., 'feature_0')
        is_sklearn_generic_input = all(
            str(col).startswith("feature_") for col in X_df.columns
        )
        if is_sklearn_generic_input:
            if len(X_df.columns) < len(required_features):
                raise ValueError(
                    f"Input has {len(X_df.columns)} features, but the causal model "
                    f"requires {len(required_features)}: {required_features}"
                )
            # Select the first N columns and rename them to match the semantic roles.
            # This is a specific, documented workaround for sklearn's check_estimator.
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

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None, feature_names=None):
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
            Feature names when X is an array. Strongly recommended for array inputs
            to ensure correct semantic mapping.

        Returns
        -------
        self
            Returns self for method chaining.
        """
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
        self.feature_columns_ = self._extract_feature_columns()

        # Prepare the feature DataFrame using the centralized, safe logic
        X_features = self._prepare_feature_df(X, feature_names)

        # Set sklearn attributes. n_features_in_ should reflect the original input shape.
        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = np.array(X_features.columns, dtype=object)

        # Initialize base estimator
        self.estimator_ = (
            LinearRegression()
            if self.base_estimator is None
            else clone(self.base_estimator)
        )

        # Fit the base estimator on the prepared features
        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self.estimator_.fit(X_features, y_arr, **fit_params)

        return self

    def predict(self, X, feature_names=None):
        """Make predictions using the fitted regressor."""
        check_is_fitted(self, "estimator_")

        # Prepare the feature DataFrame using the same safe logic as in fit
        X_features = self._prepare_feature_df(X, feature_names)

        # Make predictions
        predictions = self.estimator_.predict(X_features)
        return np.asarray(predictions).ravel()

    def get_feature_names_out(self, input_features=None):
        """
        Get output feature names for transformation.
        """
        check_is_fitted(self, "estimator_")
        return np.array(self.feature_columns_, dtype=str)
