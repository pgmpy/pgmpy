from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted, validate_data

from pgmpy.base.DAG import DAG


class DoubleMLRegressor(RegressorMixin, BaseEstimator):
    """
    Double-ML skeleton (single-exposure) with cross-fitting and explicit feature_names support.

    Parameters
    ----------
    dag : object
        DAG object
    estimator_g : estimator-like
        Outcome nuisance model prototype (must implement fit/predict).
    estimator_m : estimator-like or None
        Treatment nuisance model prototype (if None, estimator_g is used for both).
    n_folds : int
        Number of folds for cross-fitting (>0).
    seed : int or None
        Random seed for folding.

    Examples
    --------
    TO:DO

    References
    ----------
    TO:DO

    """

    def __init__(
        self,
        dag: DAG,
        estimator_g: Any,
        estimator_m: Optional[Any] = None,
        n_folds: int = 5,
        seed: Optional[int] = None,
    ):

        self.dag = dag
        self.estimator_g = estimator_g
        self.estimator_m = estimator_m
        self.n_folds = n_folds
        self.seed = seed

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        tags.regressor_tags.poor_score = True
        tags.non_deterministic = True
        return tags

    def _ensure_dataframe(self, X, feature_names=None) -> pd.DataFrame:
        """
        Converts input data X to a pandas DataFrame.

        - If X is a DataFrame: return a copy and coerce column names to strings.
        - If X is array-like: reshape 1-D to (n_samples, 1), then:
            * if feature_names provided, uses them
            * else use string column names '0','1','2'....

        Parameters
        ----------
        X : array-like or pd.DataFrame
            Input features.
        feature_names : list of str, optional
            Names for the columns. If None, names are auto-generated.

        Returns
        -------
        pd.DataFrame
            DataFrame with named columns suitable for downstream processing.
        """

        if isinstance(X, pd.DataFrame):
            X_df = X.copy()
            X_df.columns = [str(c) for c in X_df.columns]
            return X_df

        # Convert array-like to ndarray and ensure 2D
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)

        n_cols = arr.shape[1]

        # If user provided explicit feature_names, use those (and verify length)
        if feature_names is not None:
            if len(feature_names) != n_cols:
                raise ValueError(
                    f"feature_names has length {len(feature_names)} but input has {n_cols} columns"
                )
            columns = [str(c) for c in feature_names]
        else:
            # Default: integer column names 0,1,2,... to match sklearn-style DataFrames
            columns = [str(i) for i in range(n_cols)]

        return pd.DataFrame(arr, columns=columns)

    def _read_roles(self) -> Tuple[str, List[str]]:
        """
        Read roles from DAG without mutating the original user-supplied DAG.
        """
        if not isinstance(self.dag, DAG):
            raise ValueError("causal_graph must be an instance of pgmpy's DAG class.")
        dag_copy = self.dag.copy()
        dag_copy.is_valid_causal_structure()

        exposure = dag_copy.get_role("exposure")
        if len(exposure) != 1:
            raise NotImplementedError(
                "This estimator supports exactly one exposure variable."
            )
        exposure_col = exposure[0]

        adj_list = dag_copy.get_role("adjustment")
        return exposure_col, adj_list

    def _prepare_feature_df(
        self, X, feature_names: Optional[Sequence[str]] = None
    ) -> pd.DataFrame:
        """
        Ensure input X is a DataFrame whose columns match the DAG roles required by this estimator.

        Behavior:
        - If X is a DataFrame with semantic column names (e.g., 'x0','x1','x2'), verify required columns exist and
            retrun a DataFrame with columns ordered as [exposure] + adjustments + pretreatment_vars (if present).
        - If X is an ndarray or DataFrame with generic column names (feature_0, feature_1, ... OR integer names),
            treat the first N columns as corresponding to the required DAG features and rename them to role names.
        """
        exposure_col, adj_cols = self._read_roles()

        required_features = [exposure_col] + list(adj_cols)

        # Convert to DataFrame
        X_df = self._ensure_dataframe(X, feature_names=feature_names)

        # Standard named-columns path: ensure required columns exist
        missing = set(required_features) - set(X_df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in input data: {sorted(missing)}. Required columns: {required_features}"
            )

        # return DataFrame with exact ordering of required features
        return X_df[required_features].copy()

    def fit(self, X, y, sample_weight: Optional[Any] = None):
        # Step 0: Validate inputs
        if not isinstance(self.n_folds, int):
            raise ValueError("n_folds must be an integer >= 1 ")
        if self.n_folds < 1:
            raise ValueError("n_folds must be an integer >= 1 ")

        X_arr, y_arr = validate_data(
            self, X, y, accept_sparse=False, ensure_2d=True, ensure_all_finite=True
        )

        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.ndim != 1:
                raise ValueError("sample_weight must be 1D of shape (n_samples,)")
            elif sample_weight.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weight must have shape (n_samples,)")

        # Step 1: Preprocess the input data.
        self.n_features_in_ = X_arr.shape[1]
        df = self._prepare_feature_df(X, feature_names=None)
        self.feature_columns_ = list(df.columns)
        df["outcome"] = np.asarray(y).ravel()
        exposure_col, adj_cols = self._read_roles()
        n_samples = df.shape[0]

        # Step 2: Prepare nuisance covariates excluding treatment
        if len(adj_cols) == 0:
            # Use an intercept only column to ensure estimators recieve a 2D array when adj_col is empty
            covariates_df = pd.DataFrame(
                {"_intercept": np.ones(n_samples)}, index=df.index
            )
        else:
            # use the adjustment columns as the features for nuisance models
            covariates_df = df[adj_cols].copy()

        target_vec = df["outcome"]
        exposure_vec = df[exposure_col]

        # Step 3: Fit nuisance models
        # If the user requests for single fold (n_folds == 1), perform a full sample nuisance fit
        if int(self.n_folds) == 1:
            ml_g = clone(self.estimator_g)
            ml_m = (
                clone(self.estimator_m)
                if self.estimator_m is not None
                else clone(self.estimator_g)
            )

            # Fit nuisance models on the entire covariate set
            ml_g.fit(covariates_df, target_vec, sample_weight=sample_weight)
            g_pred = ml_g.predict(covariates_df)
            # ensure a pandas Series aligned with covariates_df.index (and numeric)
            g_hat_ser = pd.Series(g_pred, index=covariates_df.index, dtype=float)

            ml_m.fit(covariates_df, exposure_vec, sample_weight=sample_weight)
            m_pred = ml_m.predict(covariates_df)
            m_hat_ser = pd.Series(m_pred, index=covariates_df.index, dtype=float)

            self.estimator_g_ = ml_g
            self.estimator_m_ = ml_m

            # always store Series for consistency
            self.g_hat_ = g_hat_ser
            self.m_hat_ = m_hat_ser
        else:
            # Cross-fitting branch
            splitter = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )

            g_hat = pd.Series(0.0, index=df.index, dtype=float)
            m_hat = pd.Series(0.0, index=df.index, dtype=float)

            for train_idx, test_idx in splitter.split(covariates_df, exposure_vec):
                ml_g = clone(self.estimator_g)
                ml_m = (
                    clone(self.estimator_m)
                    if self.estimator_m is not None
                    else clone(self.estimator_g)
                )

                ml_g.fit(covariates_df.iloc[train_idx], target_vec.iloc[train_idx])
                g_test_pred = ml_g.predict(covariates_df.iloc[test_idx])

                test_index = covariates_df.iloc[test_idx].index
                g_test_pred_ser = pd.Series(g_test_pred, index=test_index, dtype=float)
                g_hat.loc[g_test_pred_ser.index] = g_test_pred_ser

                # Same for m
                ml_m.fit(covariates_df.iloc[train_idx], exposure_vec.iloc[train_idx])
                pred_m = ml_m.predict(covariates_df.iloc[test_idx])
                m_test_pred_ser = pd.Series(pred_m, index=test_index, dtype=float)
                m_hat.loc[m_test_pred_ser.index] = m_test_pred_ser

            # After loop, store Series
            self.g_hat_ = g_hat
            self.m_hat_ = m_hat

        # Step 4: orthogonal estimate (OLS on residuals)
        y_res = target_vec - self.g_hat_
        t_res = exposure_vec - self.m_hat_

        # reshape treatment residuals to 2D (sklearn expects 2D X)
        X_t_for_ols = t_res.to_frame(name="t_res")

        # perform a simple OLS of y_res on t_res (and intercept)
        estimator = LinearRegression()
        estimator.fit(X_t_for_ols, y_res, sample_weight=sample_weight)
        self.ols_estimator_ = estimator

        theta = (
            float(estimator.coef_[0])
            if getattr(estimator, "coef_", None) is not None
            and len(estimator.coef_) > 0
            else 0.0
        )
        self.treatment_effect_ = theta
        # store coef_ as a Python list (or pandas.Series) if you want to avoid np; tests may expect numpy ndarray
        self.coef_ = [theta] + [0.0] * len(adj_cols)
        self.intercept_ = float(estimator.intercept_)

        # Step 5: Bookeeping and return
        self._design_columns = [exposure_col] + adj_cols
        self.n_folds_ = self.n_folds
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """
        Computes final prediction: (intercept + theta*exposure + g_pred)
        """
        # ensure estimator is fitted
        check_is_fitted(self, "n_features_in_")

        # Handle sklearn compatibility checks
        validate_data(self, X, reset=False, ensure_2d=True, ensure_all_finite=True)

        # Map to DAG role columns (this will rename generic features to role names)
        X_df = self._prepare_feature_df(X, feature_names=None)

        exposure_col = self.feature_columns_[0]  # stored earlier in fit
        adj_cols = self.feature_columns_[1:]

        if exposure_col not in X_df.columns:
            raise ValueError(
                f"Exposure '{exposure_col}' not found in input columns for predict()."
            )

        # compute g_pred using stored nuisance models
        if (
            hasattr(self, "estimator_g_")
            and self.estimator_g_ is not None
            and len(adj_cols) > 0
        ):
            X_adj = X_df[adj_cols].copy()
            raw_g = self.estimator_g_.predict(X_adj)
            # normalize to Series aligned with X_df.index
            if isinstance(raw_g, pd.Series):
                g_pred_ser = raw_g.reindex(X_df.index).astype(float)
            else:
                g_pred_ser = pd.Series(raw_g, index=X_df.index, dtype=float)
        else:
            g_pred_ser = pd.Series(
                getattr(self, "y_mean_", 0.0), index=X_df.index, dtype=float
            )

        # treatment values as Series
        t_ser = X_df[exposure_col].astype(float)

        # ensure treatment_effect_ is scalar
        theta = float(self.ols_estimator_.coef_[0])

        # Compute predictions
        preds_ser = pd.Series(self.intercept_, index=X_df.index, dtype=float)
        preds_ser = preds_ser.add(theta * t_ser, fill_value=0.0)
        preds_ser = preds_ser.add(g_pred_ser, fill_value=0.0)

        # return numpy array for sklearn compatibility
        return preds_ser.to_numpy()
