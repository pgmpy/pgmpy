from typing import Any, Optional

import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.utils.validation import check_is_fitted, validate_data

from pgmpy.prediction._base import _BaseCausalPrediction


class NaiveIVRegressor(_BaseCausalPrediction):
    """
    Implements Naive Instrumental Variable (IV) regressor (single exposure, multiple instruments).

    This estimator implements a simple two-stage least squares style procedure
    for the case of a single exposure and a single outcome with one or more
    instrumental variables. The first stage fits `exposure ~ instrument`
    using `stage1_estimator`. The second stage fits
    `outcome ~ predicted_exposure (+ pretreatment covariates)` using `stage2_estimator`.

    Parameters
    ----------
    causal_graph : DAG, PDAG, ADMG, MAG, or PAG
        Causal graph with defined variable roles

    stage1_estimator : optional, sklearn regressor (default = LinearRegression())
        Estimator for stage 1 regression of exposure on instrument(s)

    stage2_estimator : optional, sklearn regressor (default = LinearRegression())
        Estimator for stage 2 regression of outcome on predicted exposure and pretreatment covariates (if any).

    Attributes
    ----------
    exposure_var_ : str
        Name of the exposure variable (single).

    outcome_var_ : str
        Name of the outcome variable (single).

    instrument_vars_ : list of str
        Names of instrument variables extracted from the causal graph

    pretreatment_vars_ : list of str
        Names of pretreatment covariates extracted from the causal graph.

    feature_columns_fit_ : list of str
        Names of features used during 'fit'

    feature_columns_predict_ : list of str
        Names of features used during `predict`.

    stage1_est_ : estimator
        Fitted first-stage estimator.

    stage2_est_ : estimator
        Fitted second-stage estimator.

    coef_ : array-like
        Coefficients from the fitted `stage2_estimator` (if available).

    Examples
    --------
    >>> # Example 1: Basic usage with LinearRegression estimators
    >>> import pandas as pd
    >>> from pgmpy.base import DAG
    >>> from sklearn.linear_model import LinearRegression
    >>> from pgmpy.prediction import NaiveIVRegressor
    >>>
    >>> # Simulate data from a linear Gaussian Bayesian network
    >>> lgbn = DAG.from_dagitty(
    ...     "dag { Z1 -> X [beta=0.2] Z2 -> X [beta=0.2] X -> Y [beta=0.3] }"
    ... )
    >>> data = lgbn.simulate(1000, seed=42)  # returns a pandas DataFrame
    >>> df = data.loc[:, ["X", "Z1", "Z2"]]
    >>> df = (df - df.mean(axis=0)) / df.std(axis=0)
    >>> y = data["Y"]
    >>> G = DAG(
    ...     lgbn.edges(),
    ...     roles={"exposure": "X", "instrument": ("Z1", "Z2"), "outcome": "Y"},
    ... )
    >>>
    >>> model = NaiveIVRegressor(
    ...     causal_graph=G,
    ...     stage1_estimator=LinearRegression(),
    ...     stage2_estimator=LinearRegression(),
    ... )
    >>> # Fit the model and make predictions
    >>> _ = model.fit(df, y)
    >>> preds = model.predict(df)
    >>> preds.shape[0]
    1000

    >>> # Example 2: Usage with multiple instruments and pretreatment
    >>> import pandas as pd
    >>> from pgmpy.base import DAG
    >>> from sklearn.linear_model import LinearRegression
    >>> from pgmpy.prediction import NaiveIVRegressor
    >>>
    >>> # Simulate data from a linear Gaussian Bayesian Network
    >>> lgbn = DAG.from_dagitty(
    ...     "dag { U1 -> X [beta=0.3] U2 -> X [beta=0.2] U3 -> X [beta=0.1] "
    ...     "U4 -> X [beta=0.2] X -> Y [beta=0.6] P -> Y [beta=0.2] }"
    ... )
    >>> data = lgbn.simulate(300, seed=42)
    >>> df = data.loc[:, ["X", "U1", "U2", "U3", "P"]]
    >>>
    >>> dag = DAG(
    ...     ebunch=[
    ...         ("U1", "X"),
    ...         ("U2", "X"),
    ...         ("U3", "X"),
    ...         ("U4", "X"),
    ...         ("X", "Y"),
    ...         ("P", "Y"),
    ...     ],
    ...     roles={
    ...         "exposure": "X",
    ...         "instrument": ("U1", "U2", "U3"),
    ...         "outcome": "Y",
    ...         "pretreatment": ["P"],
    ...     },
    ... )
    >>> model = NaiveIVRegressor(
    ...     causal_graph=dag,
    ... )
    >>>
    >>> # Fit the model and make predictions
    >>> _ = model.fit(df, data["Y"])
    >>> preds = model.predict(df)
    >>> preds.shape[0]
    300

    >>> # Example 3: Usage with custom estimators and numpy array inputs
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.base import DAG
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from pgmpy.prediction import NaiveIVRegressor
    >>>
    >>> dag = DAG(
    ...     ebunch=[(1, 0), (0, 2)],
    ...     roles={"exposure": [0], "outcome": [2], "instrument": [1]},
    ... )
    >>> model = NaiveIVRegressor(
    ...     causal_graph=dag,
    ...     stage1_estimator=RandomForestRegressor(),
    ...     stage2_estimator=LinearRegression(),
    ... )
    >>>
    >>> # Simulate some random data
    >>> n_samples = 50
    >>> X_array = np.random.normal(0, 1, (n_samples, 2))
    >>> y_array = np.random.normal(0, 1, n_samples)
    >>>
    >>> # Fit the model and make predictions
    >>> _ = model.fit(X_array, y_array)
    >>> preds = model.predict(X_array)
    >>> preds.shape[0]
    50

    References
    ----------
    .. [1] “Instrumental Variables Estimation.”
           Wikipedia: https://en.wikipedia.org/wiki/Instrumental_variables_estimation
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

    def fit(self, X, y, sample_weight: Optional[Any] = None):
        """
        This method performs two-stage least squares regression using the specified causal graph.
        It first fits the stage 1 estimator to predict the exposure variable from the instrument,
        then fits the stage 2 estimator to predict the outcome variable from the predicted exposure
        and pretreatment variables.

        Parameters
        ----------
        X : pandas.DataFrame or numpy ndarray
            Feature data containing exposure, instrument, and pretreatment variables.

        y : pandas.Series, pandas.DataFrame, or numpy.ndarray
            Outcome variable.

        sample_weight : array-like, optional
            Sample weights for fitting the estimators.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        # Step 0: validate Inputs
        validate_data(
            self,
            X,
            y,
            accept_sparse=False,
            ensure_2d=True,
            ensure_min_features=2,
            dtype="numeric",
        )

        # Step 1: Initialize data structures and read roles from DAG.

        if self.stage1_estimator is None:
            self.stage1_estimator = LinearRegression()
        if self.stage2_estimator is None:
            self.stage2_estimator = LinearRegression()

        stage1_estimator = clone(self.stage1_estimator)
        stage2_estimator = clone(self.stage2_estimator)

        # Step 1.1: Get roles from the causal graph and assign to attributes.
        exposure_vars = self.causal_graph.get_role("exposure")
        outcome_vars = self.causal_graph.get_role("outcome")
        instrument_vars = self.causal_graph.get_role("instrument")

        # Step 1.2: Validate that exactly one exposure, one outcome and atleast one instrument are specified.
        if len(exposure_vars) != 1:
            raise ValueError(
                f"The current implementation only works for a single exposure; got {len(exposure_vars)}"
            )
        if len(outcome_vars) != 1:
            raise ValueError(
                f"The current implementation only works for a single outcome; got {len(outcome_vars)}"
            )
        if len(instrument_vars) < 1:
            raise ValueError("NaiveIVRegressor requires at least one instrument.")

        self.exposure_var_ = exposure_vars[0]
        self.outcome_var_ = outcome_vars[0]
        self.instrument_vars_ = instrument_vars
        self.pretreatment_vars_ = self.causal_graph.get_role("pretreatment")
        self.feature_columns_fit_ = (
            [self.exposure_var_] + self.instrument_vars_ + self.pretreatment_vars_
        )

        # Step 1.2: Prepare feature dataframes and sample weights
        df = self._prepare_feature_df(X, required_features=self.feature_columns_fit_)

        self.feature_columns_predict_ = [self.exposure_var_] + self.pretreatment_vars_

        exposure_df = df[self.exposure_var_]
        instrument_df = df[self.instrument_vars_]
        pretreatment_df = df[self.pretreatment_vars_]

        # Step 2: fit stage1: E ~ Z
        stage1_estimator.fit(instrument_df, exposure_df, sample_weight=sample_weight)
        t_hat = stage1_estimator.predict(instrument_df)

        # Step 2.1: fit stage2: Y ~ t_hat + X
        t_hat_2d = pd.DataFrame(t_hat.reshape(-1, 1), columns=[self.exposure_var_])
        covariates_df = pd.concat([t_hat_2d, pretreatment_df], axis=1)
        stage2_estimator.fit(covariates_df, y, sample_weight=sample_weight)

        # step 3: Store fitted estimators and coefficients
        self.stage1_est_ = stage1_estimator
        self.stage2_est_ = stage2_estimator
        self.coef_ = self.stage2_est_.coef_

        return self

    def predict(self, X):
        # Step 0: Validate Inputs and check if fit has been called
        check_is_fitted(self, "stage1_est_")
        check_is_fitted(self, "stage2_est_")

        validate_data(
            self, X, accept_sparse=False, ensure_2d=True, dtype="numeric", reset=False
        )

        # Step 1: Prepare feature DataFrame for prediction
        X_df = self._prepare_feature_df(
            X, required_features=self.feature_columns_predict_
        )

        exposure = X_df[self.exposure_var_]
        pre_treatment = X_df[self.pretreatment_vars_]

        # Step 2: Predict using stage2 estimator
        y_pred = self.stage2_est_.predict(pd.concat([exposure, pre_treatment], axis=1))
        return y_pred
