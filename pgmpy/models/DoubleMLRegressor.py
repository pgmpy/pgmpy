import warnings
from types import SimpleNamespace
from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_X_y

from pgmpy.base.DAG import DAG


class _SklearnTags:
    """
    Wrapper for sklearn tags expected by estimator_checks.
    Exposes tag keys as attributes, provides input_tags, and implements
    dict-like access and required internal flags such as _skip_test.
    """

    def __init__(self, tags_dict: Mapping[str, Any]):
        self._tags = dict(tags_dict or {})

        # Expose tags as attributes for attribute-style access
        for k, v in self._tags.items():
            # don't overwrite internal attributes accidentally
            if not hasattr(self, k):
                setattr(self, k, v)

        # Provide input_tags namespace expected by newer sklearn internals
        self.input_tags = SimpleNamespace(
            two_d_array=True,
            # add other input-related flags if needed:
            # allow_nd=False, allow_sparse=False, dtype="float"
        )

        # Required by sklearn's harness in various versions
        self._skip_test = False
        # convenience alias
        self._supports = self._tags

    # dict-like helpers
    def get(self, key, default=None):
        return self._tags.get(key, default)

    def keys(self):
        return self._tags.keys()

    def items(self):
        return self._tags.items()

    def __getitem__(self, key):
        return self._tags[key]

    def __repr__(self):
        return f"_SklearnTags({self._tags!r}, input_tags={self.input_tags!r})"


class DoubleMLRegressor(BaseEstimator, RegressorMixin):
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
    feature_names : sequence of str or None
        If X passed to fit is a NumPy array, this **must** be provided and must match the
        number of columns in X. If X is a DataFrame and feature_names is provided, the DataFrame
        will be copied and its columns renamed to feature_names (so pass names in the same order
        as the array/columns).
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
        random_state: Optional[int] = None,
        feature_names: Optional[Sequence[str]] = None,
        allow_array_unnamed: bool = False,
    ):

        self.dag = dag
        self.estimator_g = estimator_g
        self.estimator_m = estimator_m
        self.n_folds = int(n_folds)
        self.random_state = random_state

        # implementing feature names for mapping of NumPy arrays
        if feature_names is None:
            self.feature_names = None
        else:
            # convert to list of strings and validate uniqueness
            self.feature_names = [str(fn) for fn in feature_names]
            if len(set(self.feature_names)) != len(self.feature_names):
                raise ValueError("feature_names must be unique.")
        self.allow_array_unnamed = bool(allow_array_unnamed)

    def _more_tags(self):
        return {"requires_y": True, "no_sparse_input": True}

    def get_tags(self):
        tags = {"requires_y": True, "no_sparse_input": True}
        try:
            more = self._more_tags()
            if isinstance(more, dict):
                tags.update(more)
        except Exception:
            pass
        return tags

    def __sklearn_tags__(self):
        """
        Return compatibility tags object for sklearn estimator checks.
        Provides a robust default tag set for regressors and uses the class-level
        wrapper so sklearn internals can access attributes like .input_tags,
        ._skip_test, and tag attributes (e.g., requires_fit).
        """
        # Start from get_tags() if present, else fall back to defaults
        try:
            base_tags = dict(self.get_tags())
        except Exception:
            base_tags = {}

        defaults = {
            "requires_fit": True,
            "requires_y": True,
            "no_sparse_input": True,
            "multioutput": False,
            "allow_nan": False,
            "requires_positive_y": False,
            "X_types": ["2darray"],
        }

        for k, v in defaults.items():
            base_tags.setdefault(k, v)

        # Return the wrapper object expected by sklearn's tests
        return _SklearnTags(base_tags)

    def _as_dataframe(self, X) -> pd.DataFrame:
        """
        Convert X to a pandas DataFrame with deterministic column names.

        Rules:
         - If X is a DataFrame: return a copy. If feature_names were provided, the copy's
           columns will be set to feature_names (order assumed correct).
         - If X is an ndarray:
             * if feature_names is provided: use them as columns (length checked).
             * elif allow_array_unnamed True: synthesize x0,x1,... and warn the user.
             * else: raise ValueError instructing the user to pass a DataFrame or feature_names.
        """
        if isinstance(X, pd.DataFrame):
            df = X.copy()
            if self.feature_names is not None:
                # enforce caller-provided mapping order (user must ensure correct order)
                if len(self.feature_names) != df.shape[1]:
                    raise ValueError(
                        "feature_names length does not match DataFrame columns. "
                        f"{len(self.feature_names)} != {df.shape[1]}"
                    )
                df.columns = list(self.feature_names)
            return df

        # numpy array path
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        n_cols = arr.shape[1]

        if self.feature_names is not None:
            if len(self.feature_names) != n_cols:
                raise ValueError(
                    "feature_names length does not match number of columns in X. "
                    f"{len(self.feature_names)} != {n_cols}"
                )
            cols = list(self.feature_names)
            return pd.DataFrame(arr, columns=cols)

        if self.allow_array_unnamed:
            warnings.warn(
                "Passing a NumPy array without feature_names: columns will be named x0, x1, ... "
                "This is unsafe for causal estimation; prefer passing a DataFrame or feature_names.",
                UserWarning,
            )
            cols = [f"x{i}" for i in range(n_cols)]
            return pd.DataFrame(arr, columns=cols)

        # reject ambiguous arrays
        raise ValueError(
            "Ambiguous input: X is a NumPy array but no feature_names were provided. "
            "Either pass X as a pandas DataFrame with column names matching the DAG roles, "
            "or construct the estimator with feature_names=[...]."
        )

    def _read_roles(self) -> Tuple[str, List[str]]:
        if not (
            hasattr(self.dag, "get_role")
            and hasattr(self.dag, "is_valid_causal_structure")
        ):
            # if DAG class is available check type; otherwise raise guidance
            if isinstance(DAG, type) and not isinstance(self.dag, DAG):
                raise ValueError(
                    "dag must be an instance of pgmpy's DAG or implement get_role/is_valid_causal_structure."
                )
            # else proceed but still require get_role/is_valid_causal_structure
            if not hasattr(self.dag, "get_role") or not hasattr(
                self.dag, "is_valid_causal_structure"
            ):
                raise ValueError(
                    "dag must implement get_role(role) and is_valid_causal_structure()."
                )

        # Validate the DAG; is_valid_causal_structure raises useful errors if invalid
        try:
            self.dag.is_valid_causal_structure()
        except Exception as e:
            raise ValueError(f"DAG validation failed: {e}")

        exposure_list = self.dag.get_role("exposure") or []
        if not exposure_list:
            raise ValueError(
                "DAG must define an 'exposure' role. Use dag.with_role('exposure', var)."
            )
        if len(exposure_list) != 1:
            raise NotImplementedError(
                "This estimator supports exactly one exposure variable."
            )
        exposure_col = exposure_list[0]

        # adjustments
        adj_raw = self.dag.get_role("adjustment") or []
        if isinstance(adj_raw, (list, tuple, set)):
            adj_list = list(adj_raw)
        elif adj_raw is None:
            adj_list = []
        else:
            adj_list = [adj_raw]

        adj_list = [c for c in adj_list if c is not None and c != exposure_col]
        return exposure_col, adj_list

    # -----Fit ---- Predict ---
    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):
        X_arr, y_arr = check_X_y(
            X, y, accept_sparse=False, ensure_2d=True, force_all_finite=True
        )

        # convert to DataFrame (and validate mapping)
        dfX = X if isinstance(X, pd.DataFrame) else self._as_dataframe(X_arr)
        feature_names = list(dfX.columns)

        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = np.array(feature_names, dtype=object)

        df = dfX.copy()
        df["outcome"] = np.asarray(y_arr).ravel()

        exposure_col, adj_cols = self._read_roles()

        # ensure DAG-required columns are present
        missing = [c for c in [exposure_col] + adj_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns required by DAG roles: {missing}. "
                "When using arrays, pass feature_names mapping or pass a DataFrame with correct column names."
            )

        adj_cols = [c for c in adj_cols if c in df.columns and c != exposure_col]
        design_cols = [exposure_col] + adj_cols

        # prepare nuisance covariates excluding treatment
        if len(adj_cols) == 0:
            X_for_nuisance = np.empty((df.shape[0], 0))
        else:
            X_for_nuisance = df[adj_cols].to_numpy(dtype=float)

        y_vec = df["outcome"].to_numpy(dtype=float)
        t_vec = df[exposure_col].to_numpy(dtype=float)
        n_samples = df.shape[0]
        if n_samples < 2:
            raise ValueError("Not enough samples to fit.")

        n_folds = max(2, min(self.n_folds, n_samples))
        unique_vals, counts = np.unique(t_vec, return_counts=True)
        use_stratify = unique_vals.size == 2 and np.min(counts) >= n_folds
        splitter = (
            StratifiedKFold(
                n_splits=n_folds, shuffle=True, random_state=self.random_state
            )
            if use_stratify
            else KFold(n_splits=n_folds, shuffle=True, random_state=self.random_state)
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
            full_g.fit(X_for_nuisance, y_vec)
            full_m.fit(X_for_nuisance, t_vec)
            self.estimator_g_ = full_g
            self.estimator_m_ = full_m
            self.y_mean_ = float(y_vec.mean())
            self.t_mean_ = float(t_vec.mean())

        # orthogonal estimate
        y_res = y_vec - self.g_hat_
        t_res = t_vec - self.m_hat_
        X_res = np.column_stack([np.ones(len(t_res)), t_res])
        theta_coef, *_ = np.linalg.lstsq(X_res, y_res, rcond=None)
        intercept = float(theta_coef[0])
        theta = float(theta_coef[1])

        self.treatment_effect_ = theta
        self.coef_ = np.concatenate(
            ([theta], np.zeros(len(design_cols) - 1, dtype=float))
        )
        self.intercept_ = intercept

        self._design_columns = design_cols.copy()
        self.n_folds_ = n_folds
        self.is_fitted_ = True
        return self

    def predict(self, X):
        check_is_fitted(self, "is_fitted_")
        X_df = self._as_dataframe(X)

        exposure_col = self._design_columns[0]
        adj_cols = self._design_columns[1:]

        if exposure_col not in X_df.columns:
            raise ValueError(
                f"Exposure '{exposure_col}' not found in input columns for predict()."
            )

        if (
            hasattr(self, "estimator_g_")
            and self.estimator_g_ is not None
            and len(adj_cols) > 0
        ):
            X_adj = X_df[adj_cols].to_numpy(dtype=float)
            g_pred = self.estimator_g_.predict(X_adj)
            g_pred = np.asarray(g_pred).ravel()
        else:
            g_pred = np.repeat(getattr(self, "y_mean_", 0.0), X_df.shape[0])

        t_vals = X_df[exposure_col].to_numpy(dtype=float).ravel()
        preds = self.intercept_ + self.treatment_effect_ * t_vals + g_pred
        return np.asarray(preds).ravel()
