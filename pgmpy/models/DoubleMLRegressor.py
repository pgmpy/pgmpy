import copy
import warnings
from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
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
        Number of folds for cross-fitting (>=2).
    random_state : int or None
        Random seed for folding.
    allow_array_unnamed : bool
        If True, allow NumPy arrays without feature_names by automatically naming columns
        'x0','x1',... (unsafe for real causal work). Default False.
    """

    def __init__(
        self,
        dag: DAG,
        estimator_g: Any,
        estimator_m: Optional[Any] = None,
        n_folds: int = 5,
        seed: Optional[int] = None,
        allow_array_unnamed: bool = False,
    ):

        self.dag = dag
        self.estimator_g = estimator_g
        self.estimator_m = estimator_m
        self.n_folds = n_folds
        self.seed = seed
        self.allow_array_unnamed = allow_array_unnamed

    def set_params(self, **params):
        """Set only recognized params; ignore unknown ones (sklearn checks expect no exceptions)."""
        valid = set(self.get_params(deep=False).keys())
        for key, val in params.items():
            if key in valid:
                object.__setattr__(self, key, val)
            else:
                # Do not raise: sklearn's validation harness may try random set_params names.
                warnings.warn(
                    f"Ignoring unknown parameter in set_params: {key}", UserWarning
                )
        return self

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
        tags.regressor_tags.poor_score = True
        tags.non_deterministic = True
        return tags

    def _ensure_dataframe(self, X, feature_names=None) -> pd.DataFrame:
        """
        Convert input X to a pandas DataFrame
        - If X is a DataFrame: return a copy. If feature_names were provided, the copy's
          columns will be set to feature_names (order assumed correct).
        - If X is an array-like: convert to ndarray, reshape if 1-D, and set columns.
        - If feature_names is provided, use it; otherwise generate generic names
          feature_0, feature_1, ...ror instructing the user to pass a DataFrame or feature_names.
        """
        if isinstance(X, pd.DataFrame):
            return X.copy()
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=feature_names)

    def _read_roles(self) -> Tuple[str, List[str]]:
        """Read roles from DAG without mutating the original user-supplied DAG."""
        # Work on a deep copy so we never change the user's DAG object.
        dag_copy = copy.deepcopy(self.dag)

        if not (
            hasattr(dag_copy, "get_role")
            and hasattr(dag_copy, "is_valid_causal_structure")
        ):
            if isinstance(DAG, type) and not isinstance(self.dag, DAG):
                raise ValueError(
                    "dag must be an instance of pgmpy's DAG or implement get_role/is_valid_causal_structure."
                )
            if not hasattr(dag_copy, "get_role") or not hasattr(
                dag_copy, "is_valid_causal_structure"
            ):
                raise ValueError(
                    "dag must implement get_role(role) and is_valid_causal_structure()."
                )

        try:
            dag_copy.is_valid_causal_structure()
        except Exception as e:
            raise ValueError(f"DAG validation failed: {e}")

        exposure_list = dag_copy.get_role("exposure") or []
        if not exposure_list:
            raise ValueError(
                "DAG must define an 'exposure' role. Use dag.with_role('exposure', var)."
            )
        if len(exposure_list) != 1:
            raise NotImplementedError(
                "This estimator supports exactly one exposure variable."
            )
        exposure_col = exposure_list[0]

        adj_raw = dag_copy.get_role("adjustment") or []
        if isinstance(adj_raw, (list, tuple, set)):
            adj_list = list(adj_raw)
        elif adj_raw is None:
            adj_list = []
        else:
            adj_list = [adj_raw]

        adj_list = [c for c in adj_list if c is not None and c != exposure_col]
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

        cols = list(X_df.columns)
        is_generic_feature_style = all(
            str(c).startswith("feature_") for c in cols
        ) or all(isinstance(c, (int, np.integer)) for c in cols)

        if is_generic_feature_style:
            found = X_df.shape[1]
            required = len(required_features)
            if found < required:
                raise ValueError(
                    f"Input has {found} features, but the causal model "
                    f"requires {len(required_features)}: {required_features}"
                )
            # select the first N columns and rename them to the DAG role names
            out = X_df.iloc[:, : len(required_features)].copy()
            out.columns = required_features
            return out

        # Standard named-columns path: ensure required columns exist
        missing = set(required_features) - set(X_df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in input data: {sorted(missing)}. Required columns: {required_features}"
            )

        # return DataFrame with exact ordering of required features
        return X_df[required_features].copy()

    def fit(self, X, y, sample_weight: Optional[Any] = None):
        # validate input & set sklearn convention attributes
        X_arr, y_arr = validate_data(
            self, X, y, accept_sparse=False, ensure_2d=True, force_all_finite=True
        )
        self.n_features_in_ = X_arr.shape[1]

        try:
            n_folds_requested = int(self.n_folds)
        except Exception:
            raise ValueError(f"n_folds must be integer-like; got {self.n_folds!r}")

        n_folds = max(2, min(n_folds_requested, X_arr.shape[0]))

        # coerce sample_weight if provided (accept pd.Series)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)
            if sample_weight.ndim > 1:
                if sample_weight.shape == (X_arr.shape[0], 1):
                    sample_weight = sample_weight.ravel()
                else:
                    raise ValueError("sample_weight must be 1D of shape (n_samples,)")
            # Length mismatch: raise ValueError - sklearn tests expect this behaviour
            if sample_weight.shape[0] != X_arr.shape[0]:
                raise ValueError("sample_weight must have shape (n_samples,)")

        # Map inputs to DAG-role-named DataFrame
        dfX = self._prepare_feature_df(X, feature_names=None)
        self.feature_columns_ = list(dfX.columns)  # expose for predict

        df = dfX.copy()
        df["outcome"] = np.asarray(y_arr).ravel()

        exposure_col = self.feature_columns_[0]
        adj_cols = self.feature_columns_[1:]

        missing = [c for c in [exposure_col] + adj_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns required by DAG roles: {missing}. When using arrays, "
                f"pass a DataFrame with correct column names."
            )

        # prepare nuisance covariates excluding treatment
        if len(adj_cols) == 0:
            X_for_nuisance = np.empty((df.shape[0], 0))
        else:
            X_for_nuisance = df[adj_cols].to_numpy(dtype=float)

        y_vec = df["outcome"].to_numpy(dtype=float)
        t_vec = df[exposure_col].to_numpy(dtype=float)
        n_samples = df.shape[0]
        if n_samples < 2:
            raise ValueError(f"Not enough samples to fit. n_samples = {n_samples}")

        # do not coerce self.n_folds here; compute local bounded folds
        n_folds = max(2, min(int(self.n_folds), n_samples))

        unique_vals, counts = np.unique(t_vec, return_counts=True)
        use_stratify = unique_vals.size == 2 and np.min(counts) >= n_folds
        splitter = (
            StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
            if use_stratify
            else KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        )

        g_hat = np.zeros(n_samples, dtype=float)
        m_hat = np.zeros(n_samples, dtype=float)

        for train_idx, test_idx in splitter.split(
            X_for_nuisance, t_vec if use_stratify else None
        ):
            ml_g = clone(self.estimator_g)
            ml_m = (
                clone(self.estimator_m)
                if self.estimator_m is not None
                else clone(self.estimator_g)
            )

            X_train = (
                X_for_nuisance[train_idx]
                if X_for_nuisance.size
                else np.empty((train_idx.shape[0], 0))
            )
            X_test = (
                X_for_nuisance[test_idx]
                if X_for_nuisance.size
                else np.empty((test_idx.shape[0], 0))
            )
            y_train = y_vec[train_idx]
            t_train = t_vec[train_idx]

            if X_train.shape[1] == 0:
                g_test_pred = np.repeat(y_train.mean(), X_test.shape[0])
                m_test_pred = np.repeat(t_train.mean(), X_test.shape[0])
            else:
                ml_g.fit(X_train, y_train)
                g_test_pred = np.asarray(ml_g.predict(X_test)).ravel()

                ml_m.fit(X_train, t_train)
                if hasattr(ml_m, "predict_proba"):
                    try:
                        probs = ml_m.predict_proba(X_test)
                        if probs.ndim == 2 and probs.shape[1] >= 2:
                            m_test_pred = probs[:, 1]
                        else:
                            m_test_pred = probs.ravel()
                    except Exception:
                        m_test_pred = ml_m.predict(X_test)
                else:
                    m_test_pred = ml_m.predict(X_test)
                m_test_pred = np.asarray(m_test_pred).ravel()

            g_hat[test_idx] = g_test_pred
            m_hat[test_idx] = m_test_pred

        if np.any(np.isnan(g_hat)) or np.any(np.isnan(m_hat)):
            raise RuntimeError("NaN in out-of-fold nuisance predictions.")

        self.g_hat_ = g_hat
        self.m_hat_ = m_hat

        # final full-sample fits for prediction
        if X_for_nuisance.shape[1] == 0:
            self.estimator_g_ = None
            self.estimator_m_ = None
            self.y_mean_ = float(y_vec.mean())
            self.t_mean_ = float(t_vec.mean())
        else:
            full_g = clone(self.estimator_g)
            full_m = (
                clone(self.estimator_m)
                if self.estimator_m is not None
                else clone(self.estimator_g)
            )
            if X_for_nuisance.size == 0:
                X_for_nuisance = np.empty((n_samples, 0))
            else:
                X_for_nuisance_full = X_for_nuisance
            full_g.fit(X_for_nuisance_full, y_vec)
            full_m.fit(X_for_nuisance_full, t_vec)
            self.estimator_g_ = full_g
            self.estimator_m_ = full_m
            self.y_mean_ = float(y_vec.mean())
            self.t_mean_ = float(t_vec.mean())

        # orthogonal estimate (OLS on residuals)
        y_res = y_vec - self.g_hat_
        t_res = t_vec - self.m_hat_
        X_res = np.column_stack([np.ones(len(t_res)), t_res])
        theta_coef, *_ = np.linalg.lstsq(X_res, y_res, rcond=None)
        intercept = float(theta_coef[0])
        theta = float(theta_coef[1])

        self.treatment_effect_ = theta
        self.coef_ = np.concatenate(
            ([theta], np.zeros(len([exposure_col] + adj_cols) - 1, dtype=float))
        )
        self.intercept_ = intercept

        self._design_columns = [exposure_col] + adj_cols
        self.n_folds_ = n_folds
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "n_features_in_")

        X_arr = validate_data(
            self, X, reset=False, ensure_2d=True, force_all_finite=True
        )
        found = X_arr.shape[1]
        expected = getattr(self, "n_features_in_", None)
        if expected is not None and found != expected:
            raise ValueError(
                f"Found array with {found} feature(s) (shape[1]={found}) while "
                f"{self.__class__.__name__} is expecting {expected} features as input."
            )

        # Map to DAG role columns (this will rename generic features to role names)
        X_df = self._prepare_feature_df(X, feature_names=None)

        exposure_col = self.feature_columns_[0]  # stored earlier in fit
        adj_cols = self.feature_columns_[1:]

        if exposure_col not in X_df.columns:
            raise ValueError(
                f"Exposure '{exposure_col}' not found in input columns for predict()."
            )

        # compute g_pred using stored nuisance models or y_mean_ fallback
        if (
            hasattr(self, "estimator_g_")
            and self.estimator_g_ is not None
            and len(adj_cols) > 0
        ):
            X_adj = X_df[adj_cols].to_numpy(dtype=float)
            g_pred = np.asarray(self.estimator_g_.predict(X_adj)).ravel()
        else:
            g_pred = np.repeat(getattr(self, "y_mean_", 0.0), X_df.shape[0])

        t_vals = X_df[exposure_col].to_numpy(dtype=float).ravel()
        preds = self.intercept_ + self.treatment_effect_ * t_vals + g_pred
        return np.asarray(preds).ravel()
