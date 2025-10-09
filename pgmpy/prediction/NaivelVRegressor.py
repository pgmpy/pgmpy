from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted, validate_data


class NaiveIVRegressor(RegressorMixin, BaseEstimator):
    """
    Naive Instrumental Variable (IV) regressor (single exposure, single instrument).
    Closed-form estimator for scalar X and scalar instrument Z:
        beta_hat = Cov(Z, Y) / Cov(Z, X)
    Intercept is estimated as: intercept = mean(Y) - beta_hat * mean(X)
    TO : DO

    Parameters
    ----------
    causal_graph : optional
        If provided, used to get roles 'exposure', 'outcome', and 'instrument' via
        causal_graph.get_role(role_name). If not provided, `instrument` must be set.
    stage1_estimator : optional, sklearn regressor
        Estimator for stage 1 regression of exposure on instrument(s) and pretreatment covariates.
        Must implement fit() and predict() methods. Default is None.
    stage2_estimator : optional, sklearn regressor
        Estimator for stage 2 regression of outcome on predicted exposure and pretreatment covariates.
    TO : DO

    Examples
    --------
    TO : DO

    References
    ---------
    TO : DO
    """

    def __init__(
        self,
        causal_graph,
        stage1_estimator: Optional[Any] = None,
        stage2_estimator: Optional[Any] = None,
    ):
        self.causal_graph = causal_graph
        self.stage1_estimator = stage1_estimator
        self.stage2_estimator = stage2_estimator

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

    def _prepare_feature_df(self, X) -> pd.DataFrame:
        """
        Accept a numpy/pandas dataframe and returns pandas df
        If numpy array is passed, it converts to pandas df
        """
        required_features = self.feature_columns_

        # Step 2: Convert input to DataFrame format
        if isinstance(X, pd.DataFrame):
            X_df = X

        else:
            # For numpy arrays, use range index as column names
            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                raise ValueError(
                    "Reshape your data: X must be 2D. If using a 1D array, reshape it to (n_samples, 1)."
                )
            X_df = pd.DataFrame(X_arr, columns=range(X_arr.shape[1]))

        # Step 3: Validation: column names must exactly match DAG variables
        missing_columns = set(required_features) - set(X_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in input data: {list(missing_columns)}. "
                f"DAG expects columns: {required_features}, but got: {list(X_df.columns)}"
            )

        return X_df[required_features]

    def fit(self, X, y, sample_weight: Optional[Any] = None):
        """
        This method performs two-stage least squares regression using the specified causal graph.
        It first fits the stage 1 estimator to predict the exposure variable from the instrument
        and pretreatment variables, then fits the stage 2 estimator to predict the outcome
        variable from the predicted exposure and pretreatment variables.
        """
        validate_data(accept_sparse=False, ensure_2d=True, dtype="numeric")

        X_df = self._prepare_feature_df(X)
        y_arr = np.asarray(y).ravel()

        exposure_vars = list(self.causal_graph.get_role("exposure"))
        outcome_vars = list(self.causal_graph.get_role("outcome"))
        instrument_vars = list(self.causal_graph.get_role("instrument"))
        pretreatment_vars = list(self.causal_graph.get_role("pretreatment"))

        if len(exposure_vars) != 1:
            raise ValueError(
                f"NaiveIVRegressor requires exactly one exposure; got {len(exposure_vars)}"
            )
        if len(outcome_vars) != 1:
            raise ValueError(
                f"NaiveIVRegresso requires exactly one outcome; got {len(outcome_vars)}"
            )
        if len(instrument_vars) < 1:
            raise ValueError(
                "NaiveIVRegresso requires at least one instrument variable in the causal graph."
            )

        self.exposure_var_ = exposure_vars[0]
        self.outcome_var_ = outcome_vars[0]
        self.instrument_var_ = instrument_vars
        self.pretreatment_var_ = pretreatment_vars

        required_cols = (
            [self.exposure_var_]
            + list(self.instrument_vars_)
            + list(self.pretreatment_vars_)
        )
        missing = [v for v in required_cols if v not in X_df.columns]
        if missing:
            raise ValueError(
                f"Missing required columns in X for NaiveIVRegressor: {missing}"
            )
        pretreatment_df = X_df[self.pretreatment_vars_]

        stage1_estimator = clone(self.stage1_estimator)
        stage2_estimator = clone(self.stage2_estimator)

        n = X_df.shape[0]
        x_arr = X_df[self.exposure_var_].to_numpy().ravel()
        z_df = X_df[self.instrument_vars_]

        stage1_X_df = pd.concat(
            [z_df.reset_index(drop=True), pretreatment_df.reset_index(drop=True)],
            axis=1,
        )

        # fit stage1: X ~ Z + W
        if sample_weight is None:
            stage1_estimator.fit(stage1_X_df, x_arr)
        else:
            stage1_estimator.fit(stage1_X_df, x_arr, sample_weight=sample_weight)

        x_hat = stage1_estimator.predict(stage1_X_df)

        stage2_X_df = pd.concat(
            [
                pd.Series(x_hat, name=self.exposure_var_).reset_index(drop=True),
                pretreatment_vars.reset_index(drop=True),
            ],
            axis=1,
        )

        # fit stage2: Y ~ X_hat
        if sample_weight is None:
            stage2_estimator.fit(stage2_X_df, y_arr)
        else:
            stage2_estimator.fit(stage2_X_df, y_arr, sample_weight=sample_weight)

        # store
        self.stage1_est_ = stage1_estimator
        self.stage2_est_ = stage2_estimator
        self.coef_ = np.asarray(stage2_estimator.coef_).ravel()
        self.intercept_ = float(stage2_estimator.intercept_)
        self.n_samples_ = n
        self.stage2_feature_names_in_ = list(stage2_X_df.columns)
        return self

    def predict(self, X):
        check_is_fitted(self, "stage1_est_")
        check_is_fitted(self, "stage2_est_")

        validate_data(
            self, X, accept_sparse=False, ensure_2d=True, dtype="numeric", reset=False
        )

        X_df = self._prepare_feature_df(X)

        # ensure needed columns present
        missing = [
            v
            for v in [self.exposure_var_] + self.instrument_vars_ + self.control_vars_
            if v not in X_df.columns
        ]
        if missing:
            raise ValueError(f"Missing required columns in X for prediction: {missing}")

        z_df = X_df[self.instrument_vars_].reset_index(drop=True)
        if len(self.control_vars_) == 0:
            pretreatment_vars = pd.DataFrame(
                {"_intercept": np.ones(X_df.shape[0])}, index=X_df.index
            ).reset_index(drop=True)
        else:
            pretreatment_vars = X_df[self.control_vars_].reset_index(drop=True)

        stage1_X_df = pd.concat([z_df, pretreatment_vars], axis=1)
        x_hat = self.stage1_est_.predict(stage1_X_df)

        stage2_X_df = pd.concat(
            [pd.Series(x_hat, name=self.exposure_var_), pretreatment_vars], axis=1
        )
        return self.stage2_est_.predict(stage2_X_df)
