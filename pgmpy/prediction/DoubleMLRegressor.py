from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted, validate_data

from pgmpy.global_vars import config


class DoubleMLRegressor(RegressorMixin, BaseEstimator):
    """
    Implements the Double Machine Learning (DoubleML) Regressor[1] with cross-fitting.

    This estimator implements the DoubleML algorithm with cross-fitting,
    supporting compatibility with scikit-learn's estimator API. It estimates
    the causal effect of a single treatment (exposure) variable on an outcome,
    adjusting for confounders specified in a user-supplied DAG.

    Given data (Y, T, X), where:
        Y : outcome variable
        T : treatment (exposure) variable
        X : adjustment (confounder + pretreatment) variables

    The DoubleML procedure estimates the treatment effect theta as follows:

    1. Nuisance Estimation:
        - Fit a model g(X) to predict Y from X (outcome nuisance model).
        - Fit a model m(X) to predict T from X (treatment nuisance model).

    2. Orthogonalization:
        - Compute residuals:
            y_res = Y - g_hat(X)
            t_res = T - m_hat(X)
        - These residuals remove variation explained by X, isolating the effect of T on Y.

    3. Final Estimation:
        - Fit a effect estimator on y_res on t_res:
            y_res = theta * t_res + E
        - The estimated coefficient theta is the causal effect of T on Y, adjusted for confounders X.

    For new data (T_new, X_new), the predicted outcome is:
        Y_pred = intercept + theta * T_new + g_hat(X_new)
    where g_hat(X_new) is the predicted outcome nuisance value for the new adjustment variables.

    Parameters
    ----------
    causal_graph : DAG, PDAG, ADMG, MAG, or PAG
        Causal graph with defined variable roles. The causal graph must have
        the following roles: `exposure`, `outcome`, and `adjustment`.
        Additionally, `pretreatment` can be specified.

    nuisance_estimators: an estimator or a tuple of estimators of size 2.
        If a single estimator is provided, it is used for both outcome and treatment nuisance models.
        If a tuple of two estimators is provided, the first is used for the treatment model
        and the second for the outcome model.

    n_folds : int, default=5
        Number of folds to use for cross-fitting. If 1, doesn't perform
        cross-fitting and computes in-sample residuals.

    seed : int or None
        Random seed for cross-fitting splits.

    Attributes
    ----------
    n_folds_ : int
        Number of folds used in cross-fitting.

    n_features_in_ : int
        Number of features seen during fit.

    n_samples_ : int
        Number of samples seen during fit.

    exposure_var_ : str
        Name of the exposure (treatment) variable.

    outcome_var_ : str
        Name of the outcome variable.

    adjustment_vars_ : list of str
        Names of adjustment (confounder) variables.

    pretreatment_vars_ : list of str
        Names of pretreatment variables.

    feature_columns_ : list of str
        Names of features used in the model.

    outcome_est_ : estimator-like or list of estimator-like
        Fitted outcome nuisance model(s).

    treatment_est_ : estimator-like or list of estimator-like
        Fitted treatment nuisance model(s).

    effect_estimator_ : estimator-like
        Fitted final effect estimator.

    Examples
    --------
    >>> # Example 1: With adjustments and cross-fitting
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.linear_model import LinearRegression
    >>> from pgmpy.base.DAG import DAG
    >>> from pgmpy.prediction import DoubleMLRegressor

    >>> # Simulate data from a linear Gaussian BN that we use to estimate the causal effect from.
    >>> lgbn = DAG.from_dagitty(
    ...     "dag { X -> T [beta=0.2] X -> Y [beta=0.3] T -> Y [beta=0.4] }"
    ... )
    >>> data = lgbn.simulate(n_samples=1000, seed=42)
    >>> X = data.loc[:, ["X", "T"]]
    >>> y = data["Y"]

    >>> # construct a DAG (roles must match DataFrame column names)
    >>> dag = DAG(
    ...     lgbn.edges(), roles={"exposure": "T", "adjustment": "X", "outcome": "Y"}
    ... )
    >>> dml = DoubleMLRegressor(
    ...     causal_graph=dag,
    ...     nuisance_estimators=LinearRegression(),
    ...     effect_estimator=LinearRegression(),
    ...     n_folds=3,
    ... )
    >>> _ = dml.fit(X, y)
    >>> dml.effect_estimator_.coef_.round(1)
    array([0.4])

    >>> preds = dml.predict(X.iloc[:5])
    >>> preds.shape
    (5,)

    References
    ----------
    .. [1] Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
           C., Newey, W., & Robins, J. (2018). Double/debiased machine learning for
           treatment and structural parameters. The Econometrics Journal, 21(1),
           C1-C68.

    """

    def __init__(
        self,
        causal_graph,
        nuisance_estimators,
        effect_estimator,
        n_folds: int = 5,
        seed: Optional[int] = None,
    ):

        self.causal_graph = causal_graph
        self.nuisance_estimators = nuisance_estimators
        self.effect_estimator = effect_estimator
        self.n_folds = n_folds
        self.seed = seed

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.regressor_tags.poor_score = True
        return tags

    def _prepare_feature_df(self, X) -> pd.DataFrame:
        """
        Convert input (either numpy array or dataframe) to a DataFrame and
        validate that column names exactly match DAG variables.

        If a numpy array is provided, it is converted to a DataFrame with
        range index column names (0, 1, ..., n_features-1).
        """
        # Step 1: Get required feature columns
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
        # Step 0: Validate inputs

        # Step 0.1: Check `nuisance_estimators`, `effect_estimator`, and assign variables.
        if isinstance(self.nuisance_estimators, tuple):
            if len(self.nuisance_estimators) != 2:
                raise ValueError(
                    "If nuisance_estimators is a tuple, it must have exactly two elements."
                )
            treatment_est = clone(self.nuisance_estimators[0])
            outcome_est = clone(self.nuisance_estimators[1])
        else:
            treatment_est = clone(self.nuisance_estimators)
            outcome_est = clone(self.nuisance_estimators)

        effect_est = clone(self.effect_estimator)

        # Step 0.2: Validate `n_folds`
        if (not isinstance(self.n_folds, int)) and (self.n_folds < 1):
            raise ValueError("n_folds must be an integer >= 1 ")
        self.n_folds_ = int(self.n_folds)

        # Step 0.3: Validate `X`, `y`, and `sample_weight`.
        validate_data(self, X, y, accept_sparse=False, ensure_2d=True, dtype="numeric")

        # Step 1: Initialize data structures and read roles from DAG.

        # Step 1.1: Get roles from the causal graph and assign to attributes.
        exposure_vars = list(self.causal_graph.get_role("exposure"))
        outcome_vars = list(self.causal_graph.get_role("outcome"))
        adjustment_vars = list(self.causal_graph.get_role("adjustment"))
        pretreatment_vars = list(self.causal_graph.get_role("pretreatment"))

        self.exposure_var_ = exposure_vars[0]
        self.outcome_var_ = outcome_vars[0]
        self.adjustment_vars_ = adjustment_vars
        self.pretreatment_vars_ = pretreatment_vars
        self.feature_columns_ = (
            [self.exposure_var_] + adjustment_vars + pretreatment_vars
        )

        # Step 1.2: Prepare feature dataframe and sample weights.
        df = self._prepare_feature_df(X)
        self.n_samples_ = df.shape[0]

        if sample_weight is None:
            sample_weight = pd.Series(
                1.0, index=range(df.shape[0]), dtype=config.get_dtype()
            )
        df = df.assign(outcome=np.asarray(y))
        exposure_vec = df[self.exposure_var_]

        # Step 2: Prepare covariate dataframe. If no adjustment or pretreatment variables, use intercept only.
        if len(self.adjustment_vars_ + self.pretreatment_vars_) == 0:
            covariates_df = pd.DataFrame(
                {"_intercept": np.ones(self.n_samples_)}, index=df.index
            )
        else:
            covariates_df = df[self.adjustment_vars_ + self.pretreatment_vars_]

        # Step 3: Fit nuisance models
        # Step 3.1: If n_folds = 1, fit nuisance models on full data and compute in-sample predictions.
        if int(self.n_folds) == 1:
            outcome_est.fit(covariates_df, df["outcome"], sample_weight=sample_weight)
            outcome_pred = outcome_est.predict(covariates_df)

            treatment_est.fit(covariates_df, exposure_vec, sample_weight=sample_weight)
            treatment_pred = treatment_est.predict(covariates_df)

            self.outcome_est_ = outcome_est
            self.treatment_est_ = treatment_est

        # Step 3.2: If n_folds > 1, perform cross-fitting and compute out-of-sample predictions.
        else:
            splitter = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )

            outcome_pred = pd.Series(0.0, index=df.index, dtype=config.get_dtype())
            treatment_pred = pd.Series(0.0, index=df.index, dtype=config.get_dtype())

            self.outcome_est_ = []
            self.treatment_est_ = []
            for train_idx, test_idx in splitter.split(covariates_df, exposure_vec):
                outcome_est_kfold = clone(outcome_est)
                outcome_est_kfold.fit(
                    covariates_df.iloc[train_idx],
                    df["outcome"].iloc[train_idx],
                    sample_weight=sample_weight.iloc[train_idx],
                )
                outcome_pred.iloc[test_idx] = outcome_est_kfold.predict(
                    covariates_df.iloc[test_idx]
                )
                self.outcome_est_.append(outcome_est_kfold)

                treatment_est_kfold = clone(treatment_est)
                treatment_est_kfold.fit(
                    covariates_df.iloc[train_idx],
                    exposure_vec.iloc[train_idx],
                    sample_weight=sample_weight.iloc[train_idx],
                )

                treatment_pred.iloc[test_idx] = treatment_est_kfold.predict(
                    covariates_df.iloc[test_idx]
                )
                self.treatment_est_.append(treatment_est_kfold)

        # Step 4: Compute the residuals.
        outcome_res = df["outcome"] - outcome_pred
        treatment_res = exposure_vec - treatment_pred

        # Step 5: Fit the final effect estimator on the residuals.
        effect_est.fit(
            treatment_res.to_frame(), outcome_res, sample_weight=sample_weight
        )
        self.effect_estimator_ = effect_est

        return self

    def predict(self, X):
        """
        Computes final prediction: (intercept + theta*exposure + g_pred)
        """
        # Step 0: Validate inputs and check if fitted
        check_is_fitted(self, "effect_estimator_")
        check_is_fitted(self, "outcome_est_")
        check_is_fitted(self, "treatment_est_")

        validate_data(
            self, X, accept_sparse=False, ensure_2d=True, dtype="numeric", reset=False
        )

        # Step 1: Prepare feature DataFrame
        X_df = self._prepare_feature_df(X)
        X_new_treatment = X_df[self.exposure_var_]
        if len(self.adjustment_vars_ + self.pretreatment_vars_) == 0:
            X_new_covariates = pd.DataFrame(
                {"_intercept": np.ones(X_df.shape[0])}, index=X_df.index
            )
        else:
            X_new_covariates = X_df[self.adjustment_vars_ + self.pretreatment_vars_]

        # Step 2: Compute and return predictions.
        # Step 2.1: If single fold, use the predictions from the single fitted
        #           nuisance model to compute the outcome prediction
        if self.n_folds_ == 1:
            res_x_new = X_new_treatment - self.treatment_est_.predict(X_new_covariates)
            outcome_pred = self.effect_estimator_.predict(
                res_x_new.to_frame()
            ) + self.outcome_est_.predict(X_new_covariates)

        # Step 2.2: If cross-fitted, average the predictions from each fold's
        #           nuisance model to compute the outcome prediction.
        else:
            treatment_preds = np.column_stack(
                [est.predict(X_new_covariates) for est in self.treatment_est_]
            )
            treatment_pred_mean = np.mean(treatment_preds, axis=1)

            outcome_preds = np.column_stack(
                [est.predict(X_new_covariates) for est in self.outcome_est_]
            )
            outcome_pred_mean = np.mean(outcome_preds, axis=1)

            res_x_new = (X_new_treatment - treatment_pred_mean).to_frame().values
            outcome_pred = self.effect_estimator_.predict(res_x_new) + outcome_pred_mean

        return outcome_pred
