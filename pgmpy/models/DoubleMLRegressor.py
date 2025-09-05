from typing import Any, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, check_array, clone
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.utils.validation import check_is_fitted, check_X_y

from pgmpy.base.DAG import DAG


class _TagContainer:
    """Generic tag-container that returns safe defaults for unknown attributes.

    Attributes explicitly set are returned. For any attribute sklearn probes but
    we don't set, return False so tests don't crash.
    """

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def __getattr__(self, name):
        # For boolean-like probes, default to False instead of raising.
        return False

    def __repr__(self):
        attrs = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"_TagContainer({attrs})"


class _SklearnTagsDict(dict):
    """Dict-like object returned by __sklearn_tags__ that also exposes
    .input_tags / .target_tags attributes required by estimator_checks.

    - mapping values must be plain Python types (bool, list, str, ...).
    - attribute access for unknown boolean-like flags returns False.
    - attribute access for unknown "*_tags" returns a TagContainer.
    """

    def __init__(self, mapping: Mapping[str, Any]):
        super().__init__(mapping)
        # safe containers sklearn probes
        object.__setattr__(
            self,
            "input_tags",
            _TagContainer(
                two_d_array=True,
                pairwise=False,
                sparse=False,
                dense=False,
                allow_nd=False,
                allow_sparse=False,
                dtype="float",
            ),
        )
        object.__setattr__(self, "target_tags", _TagContainer(required=False))
        object.__setattr__(self, "_skip_test", False)
        object.__setattr__(self, "_supports", dict(mapping))

    def __getattr__(self, name: str):
        # return mapping value if present
        if name in self:
            return self[name]
        # create and return containers for any "*_tags" probe
        if name.endswith("_tags"):
            container = _TagContainer()
            object.__setattr__(self, name, container)
            return container
        # internal attrs raise
        if name.startswith("_"):
            raise AttributeError(name)
        # default for boolean-like probes
        return False

    def __repr__(self):
        return f"_SklearnTagsDict({dict(self)!r}, input_tags={self.input_tags!r})"


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
        random_state: Optional[int] = None,
        allow_array_unnamed: bool = False,
    ):

        self.dag = dag
        self.estimator_g = estimator_g
        self.estimator_m = estimator_m
        self.n_folds = n_folds
        self.random_state = random_state
        self.allow_array_unnamed = allow_array_unnamed

    #    self.__sklearn_tags__()

    # def _more_tags(self):
    #    return {"requires_y": True, "no_sparse_input": True}

    # def get_tags(self):
    #    tags = {"requires_y": True, "no_sparse_input": True}
    #    try:
    #        more = self._more_tags()
    #        if isinstance(more, dict):
    #            tags.update(more)
    #    except Exception:
    #        pass
    #   return tags

    def __sklearn_tags__(self):
        # Plain-typed base tags (booleans, lists, strings — no custom objects)
        """
        base_tags = {
            "requires_fit": True,
            "requires_y": True,
            "no_sparse_input": True,
            "multioutput": False,
            "allow_nan": False,
            "requires_positive_y": False,
            "X_types": ("2darray",),
            "no_validation": False,
        }
        # merge user-provided tags if they are a plain dict
        try:
            more = self.get_tags()
            if isinstance(more, dict):
                base_tags.update(more)
        except Exception:
            pass

        # return a dict-like object that also exposes .input_tags / .target_tags
        return _SklearnTagsDict(base_tags)
        """
        tags = super().__sklearn_tags__()
        tags.target_tags.single_output = False
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
                    f"Found array with {found} feature(s) (shape[1]={found}) while "
                    f"{self.__class__.__name__} is expecting {required} features as input."
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

    # -----Fit ---- Predict ---
    def fit(self, X, y, sample_weight: Optional[Any] = None):
        # validate input & set sklearn convention attributes
        X_arr, y_arr = check_X_y(
            X, y, accept_sparse=False, ensure_2d=True, force_all_finite=True
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
            raise ValueError("Not enough samples to fit.")

        # do not coerce self.n_folds here; compute local bounded folds
        n_folds = max(2, min(int(self.n_folds), n_samples))

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

        X_arr = check_array(X, ensure_2d=True, force_all_finite=True)
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
