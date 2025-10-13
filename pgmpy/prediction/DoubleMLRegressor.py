from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_is_fitted, validate_data


class DoubleMLRegressor(RegressorMixin, BaseEstimator):
    """
    Implements the Double Machine Learning Regressor[1] (DML2) with cross-fitting.

    This estimator implements the DoubleML algorithm with cross-fitting in a
    scikit-learn compatible estimator API. It uses user-specified causal graphs
    to extract exposure, outcome, and adjustment variables and uses that to
    fit/predict a Double ML regressor. The model is defined as follows:

    Given data D: (Y, T, X), where:
        Y : outcome variable
        T : treatment (exposure) variable
        X : adjustment (confounder + pretreatment) variables

    The DoubleML fitting procedure consists of three main steps:

    1. Sample splitting into `n_folds` folds for cross-fitting.
    2. Fitting two nuisance estimators on each fold:
        - Outcome Models (`outcome_est_`): Predict Y using X.
        - Treatment Models (`treatment_est_`): Predict T from X.

    2. Computing residuals using nuisance estimators on each fold.
        - Outcome residuals: `Y - outcome_est_.predict(X)`
        - Treatment residuals: `T - treatment_est_.predict(X)`

    3. Stack the residuals from the folds together and fit the effect estimator
    (`effect_est_`) to predict the outcome residuals from the treatment
    residuals.

    Using the fitted models, predictions on new data `(X_new, T_new)` are computed as:

        `res_T_new = T_new - treatment_est_.predict(X_new)`
        `Y_pred =  effect_est_(res_T_new) + outcome_est_.predict(X_new)`

    Parameters
    ----------
    causal_graph : DAG, PDAG, ADMG, MAG, or PAG
        Causal graph with defined variable roles. The causal graph must have
        the following roles: `exposure`, `outcome`, and `adjustment`.
        Additionally, `pretreatment` can be specified.

    nuisance_estimators: an estimator or a tuple of estimators of size 2 (default=LinearRegression)
        If a single estimator is provided, it is used for both outcome and
        treatment nuisance models.

        If a tuple of two estimators is provided, the first one is used for the
        treatment model and the second for the outcome model.

        If None, defaults to LinearRegression for both models.

    effect_estimator : estimator-like (default=LinearRegression)
        Estimator for the final effect estimation step. Must have a `fit` method
        and a `predict` method. If None, defaults to LinearRegression.

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

    effect_est_ : estimator-like
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
    >>> dml.fit(X, y)
    >>> dml.effect_est_
    LinearRegression()
    >>> dml.effect_est_.coef_.round(1)
    array([0.4])

    >>> preds = dml.predict(X.iloc[:5])
    >>> preds.shape
    (5,)

    >>> dml.n_folds_
    3
    >>> dml.n_samples_
    1000

    Notes
    -----
    While the implementations allows the effect estimator to be any sklearn
    compatible estimator, the theoretical guarantees for DoubleML hold when the
    effect estimator is a linear model (such as LinearRegression). Using
    non-linear effect estimators may lead to biased estimates.

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
        nuisance_estimators=None,
        effect_estimator=None,
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
        """
        Fit the DoubleML model using the provided data.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature data containing exposure and adjustment variables.
            If a numpy array is provided, it is converted to a dataframe with column names starting from 0.

        y : pandas.Series, pandas.DataFrame, or numpy.ndarray
            Outcome variable. If a DataFrame is provided, it must have a single column.

        sample_weight : array-like of shape (n_samples,), optional
            Sample weights to be used in fitting the nuisance and effect estimators.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # Step 0: Validate inputs

        # Step 0.1: Check `nuisance_estimators`, `effect_estimator`, and assign variables.
        if self.nuisance_estimators is None:
            treatment_est = LinearRegression()
            outcome_est = LinearRegression()

        elif isinstance(self.nuisance_estimators, tuple):
            if len(self.nuisance_estimators) != 2:
                raise ValueError(
                    "If nuisance_estimators is a tuple, it must have exactly two elements."
                )
            treatment_est = clone(self.nuisance_estimators[0])
            outcome_est = clone(self.nuisance_estimators[1])
        else:
            treatment_est = clone(self.nuisance_estimators)
            outcome_est = clone(self.nuisance_estimators)

        if self.effect_estimator is None:
            effect_est = LinearRegression()
        else:
            effect_est = clone(self.effect_estimator)

        # Step 0.2: Validate `n_folds`
        if (not isinstance(self.n_folds, int)) or (self.n_folds < 1):
            raise ValueError("n_folds must be an integer >= 1 ")
        self.n_folds_ = self.n_folds

        # Step 0.3: Validate `X` and `y`
        validate_data(self, X, y, accept_sparse=False, ensure_2d=True, dtype="numeric")

        # Step 0.4: Validate single exposure and outcome.
        exposure_vars = list(self.causal_graph.get_role("exposure"))
        outcome_vars = list(self.causal_graph.get_role("outcome"))

        if len(exposure_vars) != 1:
            raise ValueError(
                f"DoubleMLRegressor only supports a single exposure variable. Got: {len(exposure_vars)}"
            )

        if len(outcome_vars) != 1:
            raise ValueError(
                f"DoubleMLRegressor only supports a single outcome variable. Got: {len(outcome_vars)}"
            )

        # Step 0.5: Check if n_folds is greater than n_samples.
        if self.n_folds_ > np.asarray(X).shape[0]:
            raise ValueError(
                "The number of folds specified is greater than the number of samples."
            )

        # Step 1: Initialize data structures and read roles from DAG.

        # Step 1.1: Get roles from the causal graph and assign to attributes.
        self.exposure_var_ = exposure_vars[0]
        self.outcome_var_ = outcome_vars[0]
        self.adjustment_vars_ = list(self.causal_graph.get_role("adjustment"))
        self.pretreatment_vars_ = list(self.causal_graph.get_role("pretreatment"))
        self.feature_columns_ = (
            [self.exposure_var_] + self.adjustment_vars_ + self.pretreatment_vars_
        )

        # Step 1.2: Prepare feature dataframe and sample weights.
        df = self._prepare_feature_df(X)
        df.insert(0, self.outcome_var_, np.asarray(y))

        exposure_df = df[self.exposure_var_]
        outcome_df = df[self.outcome_var_]

        self.n_samples_ = df.shape[0]

        if sample_weight is None:
            sample_weight = np.ones(self.n_samples_)

        # Step 2: Prepare covariate dataframe. If no adjustment or pretreatment
        #         variables, use intercept only.
        if len(self.adjustment_vars_ + self.pretreatment_vars_) == 0:
            covariates_df = pd.DataFrame(
                {"_intercept": np.ones(self.n_samples_)}, index=df.index
            )
        else:
            covariates_df = df[self.adjustment_vars_ + self.pretreatment_vars_]

        # Step 3: Fit nuisance models
        # Step 3.1: If n_folds = 1, fit nuisance models on full data and
        #           compute in-sample predictions.
        self.outcome_est_ = []
        self.treatment_est_ = []
        if int(self.n_folds) == 1:
            outcome_est.fit(covariates_df, outcome_df, sample_weight=sample_weight)
            outcome_pred = outcome_est.predict(covariates_df)

            treatment_est.fit(covariates_df, exposure_df, sample_weight=sample_weight)
            treatment_pred = treatment_est.predict(covariates_df)

            self.outcome_est_.append(outcome_est)
            self.treatment_est_.append(treatment_est)

        # Step 3.2: If n_folds > 1, perform cross-fitting and compute out-of-sample predictions.
        else:
            splitter = KFold(
                n_splits=self.n_folds, shuffle=True, random_state=self.seed
            )

            outcome_pred = pd.Series(0.0, index=df.index)
            treatment_pred = pd.Series(0.0, index=df.index)

            for train_idx, test_idx in splitter.split(covariates_df, exposure_df):
                outcome_est_kfold = clone(outcome_est)
                outcome_est_kfold.fit(
                    covariates_df.iloc[train_idx],
                    outcome_df.iloc[train_idx],
                    sample_weight=sample_weight[train_idx],
                )
                outcome_pred.iloc[test_idx] = outcome_est_kfold.predict(
                    covariates_df.iloc[test_idx]
                )
                self.outcome_est_.append(outcome_est_kfold)

                treatment_est_kfold = clone(treatment_est)
                treatment_est_kfold.fit(
                    covariates_df.iloc[train_idx],
                    exposure_df.iloc[train_idx],
                    sample_weight=sample_weight[train_idx],
                )

                treatment_pred.iloc[test_idx] = treatment_est_kfold.predict(
                    covariates_df.iloc[test_idx]
                )
                self.treatment_est_.append(treatment_est_kfold)

        # Step 4: Compute the residuals.
        outcome_res = outcome_df - outcome_pred
        treatment_res = exposure_df - treatment_pred

        # Step 5: Fit the final effect estimator on the residuals.
        effect_est.fit(
            treatment_res.to_frame(), outcome_res, sample_weight=sample_weight
        )
        self.effect_est_ = effect_est

        return self

    def predict(self, X):
        """
        Makes conditional interventional (CATE) predictions using the fitted DoubleML model.

        Parameters
        ----------
        X : pandas.DataFrame
            Feature data containing data for exposure and adjustment variables for which to make predictions.

        Returns
        -------
        outcome_pred : numpy.ndarray
            Predicted outcome values.

        """
        # Step 0: Validate inputs and check if fitted
        check_is_fitted(self, "effect_est_")
        check_is_fitted(self, "outcome_est_")
        check_is_fitted(self, "treatment_est_")

        validate_data(
            self, X, accept_sparse=False, ensure_2d=True, dtype="numeric", reset=False
        )

        # Step 1: Prepare feature DataFrame
        X_df = self._prepare_feature_df(X)
        X_intervention = X_df.loc[:, [self.exposure_var_]]
        if len(self.adjustment_vars_ + self.pretreatment_vars_) == 0:
            X_new_covariates = pd.DataFrame(
                {"_intercept": np.ones(X_df.shape[0])}, index=X_df.index
            )
        else:
            X_new_covariates = X_df[self.adjustment_vars_ + self.pretreatment_vars_]

        # Step 2: Compute and return predictions. Average the predictions from
        #         each fold's nuisance model to compute the outcome prediction.
        outcome_preds = np.column_stack(
            [est.predict(X_new_covariates) for est in self.outcome_est_]
        )
        outcome_pred_mean = np.mean(outcome_preds, axis=1)

        outcome_pred = self.effect_est_.predict(X_intervention) + outcome_pred_mean

        return outcome_pred
