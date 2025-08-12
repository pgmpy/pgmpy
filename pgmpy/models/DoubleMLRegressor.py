"""flake8: noqa"""

from typing import Any, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, check_X_y


class DoubleMLRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        dag: Any,
        estimator_g: Any,
        estimator_m: Optional[Any] = None,
        adjustment_set: Optional[Sequence[str]] = None,
        treatment_col: Optional[str] = None,
        n_folds: int = 5,
        random_state: Optional[int] = None,
    ):
        self.dag = dag
        self.estimator_g = estimator_g
        self.estimator_m = estimator_m
        self.adjustment_set = adjustment_set
        self.treatment_col = treatment_col
        self.n_folds = n_folds
        self.random_state = random_state

    def _more_tags(self):
        return {"requires_y": True, "no_sparse_input": True}

    # ---------- small helpers ----------
    def _as_dataframe(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X.copy()
        arr = np.asarray(X)
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        cols = [f"x{i}" for i in range(arr.shape[1])]
        return pd.DataFrame(arr, columns=cols)

    def _resolve_treatment_and_adjustments(
        self, df: pd.DataFrame
    ) -> Tuple[str, List[str]]:
        # treatment priority: explicit -> dag.exposure -> first column
        if self.treatment_col is not None and self.treatment_col in df.columns:
            treat = self.treatment_col
        else:
            exp = getattr(self.dag, "exposure", None)
            if exp is None:
                treat = df.columns[0]
            else:
                exp_list = list(exp) if isinstance(exp, (list, tuple, set)) else [exp]
                if len(exp_list) != 1:
                    raise NotImplementedError(
                        "Only single exposure supported in " "skeleton."
                    )
                treat = exp_list[0] if exp_list[0] in df.columns else df.columns[0]

        # adjustments priority: explicit -> dag.get_adjustment_set() -> []
        if self.adjustment_set is not None:
            adj = [c for c in self.adjustment_set if c in df.columns and c != treat]
        else:
            adj = []
            if hasattr(self.dag, "get_adjustment_set"):
                try:
                    adj = [
                        c
                        for c in self.dag.get_adjustment_set()
                        if c in df.columns and c != treat
                    ]
                except Exception:
                    adj = []
        return treat, adj

    def fit(self, X, y, sample_weight: Optional[np.ndarray] = None):

        X_arr, y_arr = check_X_y(
            X, y, accept_sparse=False, ensure_2d=True, force_all_finite=True
        )

        X = X if isinstance(X, pd.DataFrame) else self._as_dataframe(X_arr)
        feature_names = list(X.columns)

        self.n_features_in_ = X_arr.shape[1]
        self.feature_names_in_ = np.array(feature_names, dtype=object)

        df = X.copy()
        df["outcome"] = np.asarray(y_arr).ravel()

        treat_col, adj_cols = self._resolve_treatment_and_adjustments(df)
        design_cols = [treat_col] + [c for c in adj_cols if c != treat_col]
        y_vec = df["outcome"].to_numpy(dtype=float)

        # clone nuisance learners and fit (full-sample, simple) ----
        ml_g = clone(self.estimator_g)
        ml_m = (
            clone(self.estimator_m)
            if self.estimator_m is not None
            else clone(self.estimator_g)
        )

        adj_cols_for_nuisance = [c for c in design_cols if c != treat_col]
        if len(adj_cols_for_nuisance) == 0:
            X_for_g = np.empty((len(y_vec), 0))
        else:
            X_for_g = df[adj_cols_for_nuisance].to_numpy(dtype=float)

        if X_for_g.shape[1] == 0:
            g_hat = np.repeat(y_vec.mean(), len(y_vec))
            self.estimator_g_ = None
        else:
            ml_g.fit(X_for_g, y_vec)
            self.estimator_g_ = ml_g
            g_hat = ml_g.predict(X_for_g)
            g_hat = np.asarray(g_hat).ravel()

        # fit m (treatment mechanism)
        t_vec = df[treat_col].to_numpy(dtype=float)
        if X_for_g.shape[1] == 0:
            m_hat = np.repeat(t_vec.mean(), len(t_vec))
            self.estimator_m_ = None
        else:
            ml_m.fit(X_for_g, t_vec)
            self.estimator_m_ = ml_m
            # if classifier, prefer predict_proba for propensity
            if hasattr(ml_m, "predict_proba"):
                try:
                    probs = ml_m.predict_proba(X_for_g)
                    # if binary, take second column
                    if probs.ndim == 2 and probs.shape[1] >= 2:
                        m_hat = probs[:, 1]
                    else:
                        m_hat = probs.ravel()
                except Exception:
                    m_hat = ml_m.predict(X_for_g)
            else:
                m_hat = ml_m.predict(X_for_g)
            m_hat = np.asarray(m_hat).ravel()

        self.g_hat_ = g_hat
        self.m_hat_ = m_hat

        # orthogonalization
        y_res = y_vec - self.g_hat_
        t_res = t_vec - self.m_hat_

        # regress y_res on t_res (with intercept)
        X_res = np.column_stack([np.ones(len(t_res)), t_res])
        theta_coef, *_ = np.linalg.lstsq(X_res, y_res, rcond=None)

        intercept = float(theta_coef[0])
        theta = float(theta_coef[1])  # single-treatment effect

        # store both a clear treatment_effect_ and a coef_ vector
        self.treatment_effect_ = theta
        # create coef_ vector aligned with design_cols
        # position 0 is treatment, rest correspond to adjustments
        self.coef_ = np.concatenate(
            ([theta], np.zeros(len(design_cols) - 1, dtype=float))
        )
        self.intercept_ = intercept

        self._design_columns = design_cols.copy()
        self.is_fitted_ = True
        return self

    def predict(self, X):
        """Return predictions from the fitted internal linear model."""
        check_is_fitted(self, "is_fitted_")
        X_df = self._as_dataframe(X)

        missing = [c for c in self._design_columns if c not in X_df.columns]
        if missing:
            n_needed = len(self._design_columns)
            if X_df.shape[1] < n_needed:
                raise ValueError(
                    f"Predict input has fewer columns than required: "
                    f"need {n_needed}, got {X_df.shape[1]}"
                )
            mapped_cols = list(X_df.columns[:n_needed])
        else:
            mapped_cols = self._design_columns

        X_design = X_df[mapped_cols].to_numpy(dtype=float)
        preds = self.intercept_ + X_design.dot(self.coef_)
        return np.asarray(preds).ravel()
