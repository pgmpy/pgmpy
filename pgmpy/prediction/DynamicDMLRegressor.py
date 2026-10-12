#!/usr/bin/env python3
"""
Dynamic Double/Debiased Machine Learning for Sequential Treatment Effects.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.utils.validation import check_array, check_is_fitted


class DynamicDMLRegressor(RegressorMixin, BaseEstimator):
    """
    Dynamic Double/Debiased Machine Learning for sequential treatment effects.

    Estimates causal effects of treatments at different time periods on a
    final outcome using recursive g-estimation with Neyman orthogonal moments.

    Model Specification
    -------------------
    For a panel dataset with m periods, the structural model is:

    :math:`Y_i = sum_{t=1}^{m} psi_t T_{it} + eta(X_i) + epsilon_i`

    where:

    - :math:`Y_i` is the final outcome for unit i
    - :math:`T_{it}` is the treatment at period t for unit i
    - :math:`psi_t` are the dynamic treatment effect parameters
    - :math:`eta(X_i)` is the baseline nuisance function of confounders
    - :math:`epsilon_i` is random noise

    The algorithm uses cross-fitted nuisance models to estimate:
    - Outcome model: :math:`q_t(X_t) = E[Y | X_t]`
    - Treatment model: :math:`p_{j,t}(X_t) = E[T_j | X_t]` for j ≥ t

    Assumptions
    -----------
    1. **Sequential Conditional Exogeneity**: No unobserved confounders after
       conditioning on observed covariates X_t at each time t.
    2. **Positivity**: Treatment probabilities are bounded away from 0 and 1
       (overlap assumption).
    3. **Correct specification**: Nuisance models adequately capture the
       conditional expectations.

    Parameters
    ----------
    causal_graph : DAG, PDAG, ADMG, MAG, or PAG, optional
        Causal graph with defined variable roles. If provided, the graph must have
        the following roles: `exposure`, `outcome`. Additionally, `adjustment` and
        `pretreatment` can be specified for confounders.

        Note: For DynamicDML, the treatment variable T must be passed separately
        to the fit() method and is not included in the X dataframe, unlike the
        tabular DoubleMLRegressor where T is extracted from X using roles.

    nuisance_estimators : estimator or tuple of estimators, default=None
        Machine learning models for nuisance estimation.

        - If a single estimator: used for both treatment and outcome models
        - If a tuple of (treatment_est, outcome_est): uses separate models
        - If None or 'auto': automatically selects based on data types:
            * Integer dtypes → RandomForestRegressor
            * Float dtypes → LinearRegression

    effect_estimator : estimator, default=None
        Estimator for final effect estimation from residualized treatment to
        residualized outcome. If None, defaults to LinearRegression(fit_intercept=False).

        Note: Theoretical guarantees hold when using linear models. Non-linear
        effect estimators may lead to biased estimates.

    n_folds : int, default=2
        Number of cross-fitting folds (minimum 2) for within-period cross-validation.
        Uses KFold with shuffle=True to create folds within each time period separately.

    n_periods : int or None, default=None
        Number of treatment periods. If None, inferred from unique values in groups.

    seed : int, RandomState instance or None, default=None
        Random seed for cross-fitting splits.

    Attributes
    ----------
    `n_folds_` : int
        Number of folds used in cross-fitting.

    `exposure_var_` : str or None
        Name of the exposure (treatment) variable extracted from causal_graph.

    `outcome_var_` : str or None
        Name of the outcome variable extracted from causal_graph.

    `adjustment_vars_` : list of str
        Names of adjustment (confounder) variables extracted from causal_graph.

    `pretreatment_vars_` : list of str
        Names of pretreatment variables extracted from causal_graph.

    `feature_columns_` : list
        Names/indices of features used in the model (exposure + adjustments + pretreatment).

    `coef_` : ndarray of shape (n_periods, n_treatments)
        Estimated dynamic treatment effect parameters {ψ₁, ..., ψₘ}.
        coef_[t] is the effect of treatment at period t on final outcome.
        Only available when effect_estimator has a coef_ attribute (e.g., LinearRegression).

    `outcome_est_` : list of estimators
        Fitted outcome nuisance models for each period. outcome_est_[t] predicts
        E[Y | X_t] for observations in period t.

    `treatment_est_` : list of lists of estimators
        Fitted treatment nuisance models. treatment_est_[t] is a list of models
        for periods j >= t, where treatment_est_[t][j-t] predicts E[T_j | X_t].

    `effect_est_` : estimator
        Fitted effect estimation model mapping residualized treatments to
        residualized outcomes.

    `residuals_y_` : ndarray of shape (n_samples, n_periods)
        Cached outcome residuals Y - q̂_t(X_t) for each period for diagnostics.

    `residuals_t_` : ndarray of shape (n_samples, n_periods, n_periods)
        Cached treatment residuals. residuals_t_[i, j, t] = T_j^i - p̂_{j,t}(X_t^i).

    `covariance_` : ndarray of shape (n_periods * n_treatments, n_periods * n_treatments)
        Asymptotic covariance matrix V = J^{-1}ΣJ^{-T} for inference.

    `n_features_in_` : int
        Number of features seen during fit (sklearn standard).

    `feature_names_in_` : ndarray of shape (n_features_in_,)
        Names of features if X is pandas DataFrame (sklearn standard).

    `n_samples_` : int
        Number of samples seen during fit.

    `groups_` : ndarray
        Stored groups array from fit().

    `discrete_treatment_` : bool
        Inferred flag indicating whether treatments are discrete (integer dtype).

    `discrete_outcome_` : bool
        Inferred flag indicating whether outcome is discrete (integer dtype).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.base import DAG
    >>> from pgmpy.prediction import DynamicDMLRegressor
    >>>
    >>> # Generate synthetic panel data: 100 units over 3 periods
    >>> np.random.seed(42)
    >>> n_units, n_periods = 100, 3
    >>> X_list, T_list, y_list, groups_list = [], [], [], []
    >>>
    >>> for t in range(n_periods):
    ...     X_t = np.random.randn(n_units, 5)  # Confounders
    ...     T_t = np.random.binomial(1, 0.5, (n_units, 1))  # Binary treatment
    ...     # Outcome with time-varying treatment effects
    ...     true_effect = 0.5 - 0.15 * t  # Decreasing: [0.5, 0.35, 0.2]
    ...     y_t = (
    ...         true_effect * T_t[:, 0]
    ...         + 0.2 * X_t[:, 0]
    ...         + np.random.randn(n_units) * 0.1
    ...     )
    ...     X_list.append(X_t)
    ...     T_list.append(T_t)
    ...     y_list.append(y_t)
    ...     groups_list.append(np.full(n_units, t))
    ...
    >>>
    >>> X = np.vstack(X_list)
    >>> T = np.vstack(T_list)
    >>> y = np.hstack(y_list)
    >>> groups = np.hstack(groups_list)
    >>>
    >>> # Fit model
    >>> model = DynamicDMLRegressor(n_periods=3, n_folds=2, seed=42)
    >>> model.fit(X, T, y, groups=groups)
    DynamicDMLRegressor(...)
    >>>
    >>> # Estimated coefficients (should be close to [0.5, 0.35, 0.2])
    >>> model.coef_.ravel().round(2)  # doctest: +SKIP
    array([0.48, 0.33, 0.19])
    >>>
    >>> # Predict potential outcomes under interventions
    >>> X_new = np.random.randn(10, 5)
    >>> y_treated = model.predict(X_new, T=1)  # E[Y | do(T=1), X]
    >>> y_control = model.predict(X_new, T=0)  # E[Y | do(T=0), X]
    >>>
    >>> # Compute treatment effects
    >>> effects = model.effect(X_new, T0=0, T1=1)
    >>> effects.shape
    (10,)

    References
    ----------
    Lewis, G., & Syrgkanis, V. (2021). Double/Debiased Machine Learning for
    Dynamic Treatment Effects via g-Estimation. NeurIPS.
    https://arxiv.org/abs/2002.07285

    Notes
    -----
    - Assumes sequential conditional exogeneity (no unobserved confounding)
    - Assumes positivity (overlap in treatment assignments)
    """

    def __init__(
        self,
        causal_graph=None,
        nuisance_estimators=None,
        effect_estimator=None,
        n_folds=2,
        n_periods=None,
        seed=None,
    ):
        self.causal_graph = causal_graph
        self.nuisance_estimators = nuisance_estimators
        self.effect_estimator = effect_estimator
        self.n_folds = n_folds
        self.n_periods = n_periods
        self.seed = seed

    def __sklearn_tags__(self):
        """Tags for sklearn compatibility."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        tags.regressor_tags.poor_score = True
        return tags

    def _prepare_feature_df(self, X) -> pd.DataFrame:
        """
        Convert input (either numpy array or dataframe) to a DataFrame and
        validate that column names exactly match required feature columns.

        If a numpy array is provided, it is converted to a DataFrame with
        range index column names (0, 1, ..., n_features-1).

        Parameters
        ----------
        X : array-like or DataFrame
            Input features.

        Returns
        -------
        X_df : DataFrame
            Feature DataFrame with validated columns matching feature_columns_.
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

        # Step 3: Validation: column names must exactly match required features
        missing_columns = set(required_features) - set(X_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in input data: {list(missing_columns)}. "
                f"DAG Expected columns: {required_features}, but got: {list(X_df.columns)}"
            )

        return X_df[required_features]

    def fit(self, X, T, y, groups):
        """
        Fit the DynamicDMLRegressor model using panel data.

        This method fits the ENTIRE panel dataset containing all time periods and units.
        The data should be stacked such that rows represent (unit, time) pairs, with the
        `groups` parameter identifying which time period each observation belongs to.

        Data Structure
        --------------
        For n_units=100, n_periods=3, the input should be:
        - X.shape = (300, n_features)  # 100 units x 3 periods
        - T.shape = (300, n_treatments)
        - y.shape = (300,)
        - groups = [0,0,...,0, 1,1,...,1, 2,2,...,2]  # 100 zeros, 100 ones, 100 twos

        The fit procedure:
        1. For each period t, fits nuisance models E[Y|X] and E[T_j|X] using
           cross-fitting within that period's data
        2. Computes residualized outcomes and treatments
        3. Uses backward recursive g-estimation to recover dynamic treatment effects

        **Important**: Unlike tabular DoubleMLRegressor where treatment T is extracted
        from X using causal graph roles, here T must be passed as a SEPARATE argument
        because we need the full treatment history across all periods. The causal_graph
        (if provided) is used only for selecting adjustment variables from X.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray of shape (n_samples, n_features)
            Feature data containing adjustment variables.

            - If DataFrame: Column names must match the causal graph variable names
              (when causal_graph provided), or can be any names (when causal_graph is None).
              If DataFrame contains extra columns (e.g., exposure), only the required
              adjustment+pretreatment columns will be extracted.

            - If array: Will be converted to DataFrame with column names 0, 1, 2, ...
              When causal_graph is provided, the DAG variable names should match these
              integer indices.

            **Treatment T is NEVER included in X** - it's passed separately as the T argument.

        T : array-like of shape (n_samples, n_treatments) or (n_samples,)
            Treatment variables for each observation. Passed separately from X.
            Will be reshaped to 2D if 1D. This is the variable we want to estimate
            causal effects for.

        y : array-like of shape (n_samples,)
            Final outcome for each observation. Typically the same value is replicated
            across all periods for the same unit (outcome measured at end of study).

        groups : array-like of shape (n_samples,)
            Period identifier (0 to m-1) for each observation. REQUIRED for panel
            structure. All observations with groups==t belong to time period t.

        Returns
        -------
        self : object
            Fitted estimator with coef_ and effect_est_ attributes.

        Notes
        -----
        The current implementation uses a max_lag=1 convention, meaning at period t
        we only model E[T_t | X_t] (contemporaneous), not E[T_j | X_t] for j > t.
        This is a simplification that avoids modeling the full treatment history.
        """
        # STEP 1: Validate inputs
        self._validate_inputs(X, T, y, groups)

        # STEP 2: Extract roles from causal graph (if provided)
        if self.causal_graph is not None:
            # Extract variable roles
            exposure_vars = list(self.causal_graph.get_role("exposure"))
            outcome_vars = list(self.causal_graph.get_role("outcome"))

            # Validate exactly one exposure and one outcome
            if len(exposure_vars) != 1:
                raise ValueError(
                    f"Exactly one exposure variable must be defined in causal_graph. "
                    f"Got {len(exposure_vars)}: {exposure_vars}"
                )
            if len(outcome_vars) != 1:
                raise ValueError(
                    f"Exactly one outcome variable must be defined in causal_graph. "
                    f"Got {len(outcome_vars)}: {outcome_vars}"
                )

            # Store extracted roles
            self.exposure_var_ = exposure_vars[0]
            self.outcome_var_ = outcome_vars[0]
            self.adjustment_vars_ = list(self.causal_graph.get_role("adjustment"))
            self.pretreatment_vars_ = list(self.causal_graph.get_role("pretreatment"))

            # Feature columns = adjustment + pretreatment (NOT exposure, as T passed separately)
            self.feature_columns_ = self.adjustment_vars_ + self.pretreatment_vars_
        else:
            # No causal graph: use ALL columns in X as features (adjustment variables)
            # Treatment T is passed separately, NOT extracted from X
            self.exposure_var_ = None
            self.outcome_var_ = None
            self.adjustment_vars_ = []
            self.pretreatment_vars_ = []

            # Determine feature columns from X
            if isinstance(X, pd.DataFrame):
                self.feature_columns_ = list(X.columns)
            else:
                # For arrays, use range index (will be set after conversion in _prepare_feature_df)
                X_arr = np.asarray(X)
                self.feature_columns_ = list(range(X_arr.shape[1]))

        # STEP 3: Prepare feature DataFrame
        X_df = self._prepare_feature_df(X)

        # Convert T, y, groups to arrays and DataFrames
        T = np.asarray(T)
        y = np.asarray(y)
        groups = np.asarray(groups)

        # Reshape T to 2D if 1D
        if T.ndim == 1:
            T = T.reshape(-1, 1)

        # Convert to DataFrames for consistent indexing
        T_df = pd.DataFrame(
            T, index=X_df.index, columns=[f"T{i}" for i in range(T.shape[1])]
        )
        y_df = pd.Series(y, index=X_df.index, name="y")
        groups_series = pd.Series(groups, index=X_df.index, name="groups")

        self.groups_ = groups

        # STEP 4: Infer n_periods if not provided
        if self.n_periods is None:
            self.n_periods_ = len(np.unique(groups))
        else:
            self.n_periods_ = self.n_periods

        # Set n_folds_
        self.n_folds_ = self.n_folds

        # STEP 5: Infer discrete_treatment and discrete_outcome from dtypes
        self.discrete_treatment_ = self._infer_discrete(T)
        self.discrete_outcome_ = self._infer_discrete(y)

        # STEP 6: Parse nuisance_estimators and select models
        if self.nuisance_estimators is None:
            # Auto-select based on dtypes
            treatment_est = self._select_model(None, self.discrete_treatment_)
            outcome_est = self._select_model(None, self.discrete_outcome_)
        elif isinstance(self.nuisance_estimators, tuple):
            if len(self.nuisance_estimators) != 2:
                raise ValueError(
                    "If nuisance_estimators is a tuple, it must have exactly two elements: (treatment_est, outcome_est)"
                )
            treatment_est = clone(self.nuisance_estimators[0])
            outcome_est = clone(self.nuisance_estimators[1])
        else:
            # Single estimator for both
            treatment_est = clone(self.nuisance_estimators)
            outcome_est = clone(self.nuisance_estimators)

        # STEP 7: Set effect estimator
        if self.effect_estimator is None:
            effect_est = LinearRegression(fit_intercept=False)
        else:
            effect_est = clone(self.effect_estimator)

        # STEP 8: STAGE 1 - Cross-fitted nuisance estimation
        self._fit_nuisances(X_df, T_df, y_df, groups_series, treatment_est, outcome_est)

        # STEP 9: STAGE 2 - Backward recursive parameter estimation
        self._fit_parameters(effect_est)

        # STEP 10: Compute asymptotic covariance
        self._compute_covariance()

        # STEP 11: Set sklearn-required attributes
        self.n_features_in_ = X_df.shape[1]  # Number of features (columns)
        self.n_samples_ = X_df.shape[0]  # Number of samples (rows)
        self.feature_names_in_ = np.array(X_df.columns)

        # STEP 12: Return self
        return self

    def predict(self, X, T):
        """
        Predict conditional treatment effects: E[Y | do(T=T1), X] - E[Y | do(T=T0), X].

        ALGORITHM:
        1. For baseline E[Y | do(T=T0), X]:
           - Compute residualized treatment: T0_res = T0 - E[T|X] (using fitted models)
           - Effect from treatment: sum over periods t of coef_[t] @ T0_res
           - Add baseline: effect + E[Y|X]

        2. Similarly for E[Y | do(T=T1), X]

        3. Return difference: E[Y | do(T=T1), X] - E[Y | do(T=T0), X]

        Note: This computes CONDITIONAL treatment effects (CATE) for specific X values,
        not the average treatment effect (ATE) across the population.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate data for prediction. Should match the adjustment variables used in fit().

        T : array-like of shape (n_samples, n_treatments) or scalar
            Treatment regime to evaluate. Can be:
            - Scalar: same treatment for all samples and all periods
            - 1D array of shape (n_treatments,): same treatment pattern for all samples
            - 2D array of shape (n_samples, n_treatments): sample-specific treatments

        Returns
        -------
        outcomes : ndarray of shape (n_samples,)
            Predicted potential outcomes E[Y|do(T),X] under the intervention.

        """
        check_is_fitted(self, ["coef_", "n_periods_", "outcome_est_"])
        X = check_array(X)

        n_treatments = self.coef_.shape[1]

        # Handle T with broadcasting
        if np.isscalar(T):
            T = np.full((X.shape[0], n_treatments), T)
        else:
            T = np.asarray(T)
            if T.ndim == 1:
                if T.shape[0] == n_treatments:
                    # Single treatment pattern for all samples
                    T = np.tile(T, (X.shape[0], 1))
                else:
                    # Single treatment per sample (broadcast to all periods)
                    T = T.reshape(-1, 1)
                    if n_treatments > 1:
                        T = np.tile(T, (1, n_treatments))

        # Step 1: Compute baseline outcome eta(X) using outcome models
        # Average predictions across periods (each period's model estimates E[Y|X_t])
        eta_X = np.zeros(X.shape[0])
        for t, outcome_model in enumerate(self.outcome_est_):
            eta_X += outcome_model.predict(X)
        eta_X /= self.n_periods_  # Average across periods

        # Step 2: Compute treatment effects sum_t psi_t * T_t
        treatment_effect = np.zeros(X.shape[0])
        for t in range(self.n_periods_):
            if n_treatments == 1:
                treatment_effect += T[:, 0] * self.coef_[t, 0]
            else:
                treatment_effect += T @ self.coef_[t]

        # Step 3: Return counterfactual outcome
        return eta_X + treatment_effect

    def effect(self, X, T0=0, T1=1):
        """
        Compute treatment effects: E[Y|do(T1),X] - E[Y|do(T0),X].

        This is a convenience method that computes the difference in potential
        outcomes under two interventional treatment regimes.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Covariate data.

        T0 : scalar or array-like, default=0
            Baseline treatment regime (typically control: T=0).

        T1 : scalar or array-like, default=1
            Alternative treatment regime (typically treated: T=1).

        Returns
        -------
        effects : ndarray of shape (n_samples,)
            Treatment effects for each sample.

        Notes
        -----
        With LINEAR effect models (default), the treatment effect is constant across
        all samples because:

        E[Y|do(T1),X] - E[Y|do(T0),X] = [sum_t psi_t*T1_t + eta(X)] - [sum_t psi_t*T0_t + eta(X)]
                                       = sum_t psi_t*(T1_t - T0_t)

        The X-dependent baseline eta(X) cancels out. For heterogeneous treatment effects
        that vary with X, use non-linear effect estimators or include interaction terms.
        """
        y_t1 = self.predict(X, T=T1)
        y_t0 = self.predict(X, T=T0)
        return y_t1 - y_t0

    def effect_interval(self, X, T0=0, T1=1, alpha=0.05):
        """
        Compute (1-α) confidence intervals for conditional treatment effects.

        ALGORITHM:
        Uses asymptotic normality: effect ± z_α/2 * sqrt(variance)
        Variance comes from self.covariance_ (sandwich estimator).

        **Important**: This method assumes the effect_estimator is a linear model
        with a coef_ attribute. For non-linear effect estimators, consider using
        bootstrap-based inference instead.

        Parameters
        ----------
        X : array-like
            Adjustment variables.
        T0, T1 : scalar or array-like
            Treatment contrasts.
        alpha : float, default=0.05
            Significance level for (1-alpha) confidence interval.

        Returns
        -------
        lb : ndarray
            Lower bounds.
        ub : ndarray
            Upper bounds.

        -- self notes --
        -----
        This implementation uses the analytical covariance matrix computed during fit().
        For more flexible inference that works with any effect estimator, a separate
        BootstrapInference class that wraps this estimator could be implemented.
        """
        check_is_fitted(self, ["coef_", "covariance_"])

        effects = self.effect(X, T0=T0, T1=T1)

        # Compute standard errors from covariance diagonal
        variances = np.diag(self.covariance_)
        # Average variance across periods for simplicity
        avg_variance = np.mean(variances)
        std_error = np.sqrt(avg_variance)

        z = norm.ppf(1 - alpha / 2)

        lb = effects - z * std_error
        ub = effects + z * std_error

        return lb, ub

    def score(self, X, T, y, groups):
        """
        Return negative MSE of final stage residuals (sklearn convention).

        1. Recompute predictions using fitted model
        2. Compute residuals after applying all treatment effects
        3. Return -MSE (higher is better)

        Parameters
        ----------
        X, T, y, groups : array-like
            Test data with same format as fit().

        Returns
        -------
        score : float
            Negative mean squared error.
        """
        check_is_fitted(self, ["coef_"])

        # Convert inputs
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X)
        T = np.asarray(T)
        y = np.asarray(y)

        if T.ndim == 1:
            T = T.reshape(-1, 1)

        # Compute predictions: E[Y | do(T), X]
        y_pred = self.predict(X, T=T)

        # Compute MSE
        residuals = y - y_pred
        mse = np.mean(residuals**2)

        return -mse

    # =====================================================================
    # INTERNAL METHODS
    # =====================================================================

    def _validate_inputs(self, X, T, y, groups):
        """
        Validate input data format and consistency.

        CHECKS:
        1. X, T, y, groups have compatible first dimensions
        2. groups has values in range [0, m-1]
        3. No NaN/inf values in any input
        4. T and y are 1D or 2D

        Parameters
        ----------
        X, T, y, groups : array-like
            Input data to validate.

        Raises
        ------
        ValueError
            If validation fails with descriptive error message.
        """
        # Convert to arrays for checking
        if hasattr(X, "values"):
            X = X.values
        X = np.asarray(X)
        T = np.asarray(T)
        y = np.asarray(y)
        groups = np.asarray(groups)

        # Check first dimension compatibility
        if not (X.shape[0] == T.shape[0] == y.shape[0] == groups.shape[0]):
            raise ValueError(
                f"Incompatible shapes: X {X.shape}, T {T.shape}, "
                f"y {y.shape}, groups {groups.shape}"
            )

        # Check groups range
        unique_groups = np.unique(groups)
        if self.n_periods is not None:
            expected_range = set(range(self.n_periods))
            actual_range = set(unique_groups)
            if actual_range != expected_range:
                raise ValueError(
                    f"groups must have values in [0, {self.n_periods - 1}], "
                    f"got {unique_groups}"
                )

        # Check for NaN/inf
        if np.any(np.isnan(X)) or np.any(np.isinf(X)):
            raise ValueError("X contains NaN or inf values")
        if np.any(np.isnan(T)) or np.any(np.isinf(T)):
            raise ValueError("T contains NaN or inf values")
        if np.any(np.isnan(y)) or np.any(np.isinf(y)):
            raise ValueError("y contains NaN or inf values")
        if np.any(np.isnan(groups)) or np.any(np.isinf(groups)):
            raise ValueError("groups contains NaN or inf values")

        # Check dimensions
        if y.ndim not in [1]:
            raise ValueError(f"y must be 1D, got shape {y.shape}")
        if T.ndim not in [1, 2]:
            raise ValueError(f"T must be 1D or 2D, got shape {T.shape}")

    def _infer_discrete(self, data):
        """
        Infer whether data is discrete or continuous based on dtype.

        Parameters
        ----------
        data : array-like
            Data to check.

        Returns
        -------
        is_discrete : bool
            True if data appears discrete, False otherwise.

        Notes
        -----
        Uses heuristic: integer dtype → discrete, float dtype → continuous.
        For ambiguous cases (e.g., float with few unique values), defaults to continuous.
        """
        data = np.asarray(data)

        # Check if dtype is integer subtype
        if np.issubdtype(data.dtype, np.integer):
            return True

        # Otherwise assume continuous (float types)
        return False

    def _select_model(self, model, is_discrete):
        """
        Auto-select model when model is None.

        RULES:
        - Continuous (float dtype) -> LinearRegression()
        - Discrete (integer dtype) -> RandomForestRegressor()

        Parameters
        ----------
        model : estimator or None
            Model specification. If None, auto-selects based on is_discrete.
        is_discrete : bool
            Whether outcome/treatment is discrete (inferred from dtype).

        Returns
        -------
        selected_model : estimator
            Selected or cloned estimator ready to use.
        """
        if model is None:
            if is_discrete:
                return RandomForestRegressor(n_estimators=100, random_state=self.seed)
            else:
                return LinearRegression()
        else:
            return clone(model)

    def _fit_nuisances(
        self, X_df, T_df, y_series, groups_series, treatment_est, outcome_est
    ):
        """
        STAGE 1: Cross-fitted nuisance estimation.

        ALGORITHM (max_lag=1 convention):
        For each period t = 1, ..., m:
            1. Extract data for period t (isolate by groups == t)
            2. Create KFold cross-validator for within-period cross-fitting
            3. For each fold (train_idx, test_idx) within period t:
                a. Fit outcome model: outcome_est.fit(X[train_idx], y[train_idx])
                   Models E[Y | X_t] using contemporaneous confounders
                b. Predict on test fold: y_pred[test_idx] = outcome_est.predict(X[test_idx])
                c. Fit treatment model: treatment_est.fit(X[train_idx], T[train_idx])
                   Models E[T_t | X_t] using contemporaneous confounders (max_lag=1)
                d. Predict: t_pred[test_idx] = treatment_est.predict(X[test_idx])
            4. Compute residuals:
                - residuals_y_[period_t_mask] = y - E[Y|X_t]
                - residuals_t_[period_t_mask] = T - E[T_t|X_t]
            5. Store fitted models

        **max_lag=1 Convention**: At each period t, we only model E[T_t | X_t]
        (contemporaneous relationship), not E[T_j | X_t] for future periods j > t.
        This simplifies the implementation and avoids modeling the full treatment
        history dynamics. The full generalization would model:
        - Period 0: E[T_0|X_0], E[T_1|X_0], E[T_2|X_0], ...
        - Period 1: E[T_1|X_1], E[T_2|X_1], ...
        - Period 2: E[T_2|X_2], ...

        But with max_lag=1, we only model the diagonal:
        - Period 0: E[T_0|X_0]
        - Period 1: E[T_1|X_1]
        - Period 2: E[T_2|X_2]

        Parameters
        ----------
        X_df : DataFrame of shape (n_samples, n_features)
            Adjustment variables only (NOT including treatment).
        T_df : DataFrame of shape (n_samples, n_treatments)
            Treatment variables.
        y_series : Series of shape (n_samples,)
            Outcome variable.
        groups_series : Series of shape (n_samples,)
            Period identifiers.
        treatment_est : estimator
            Base treatment nuisance estimator to clone for each fold.
        outcome_est : estimator
            Base outcome nuisance estimator to clone for each fold.

        Sets
        ----
        self.residuals_y_ : ndarray of shape (n_samples, n_periods)
        self.residuals_t_ : ndarray of shape (n_samples, n_treatments, n_periods)
        self.outcome_est_ : list of fitted outcome models
        self.treatment_est_ : list of lists of fitted treatment models
        """
        n_samples = X_df.shape[0]
        n_treatments = T_df.shape[1]

        # Initialize storage
        self.residuals_y_ = np.zeros((n_samples, self.n_periods_))
        self.residuals_t_ = np.zeros((n_samples, n_treatments, self.n_periods_))
        self.outcome_est_ = []
        self.treatment_est_ = []

        # Setup cross-validator
        if self.n_folds < 2:
            raise ValueError("n_folds must be at least 2 for cross-fitting")
        kfold = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.seed)

        # Loop over periods
        for t in range(self.n_periods_):
            # Get mask for current period
            period_mask = groups_series == t
            X_t = X_df[period_mask]
            y_t = y_series[period_mask]
            T_t = T_df[period_mask]

            # Initialize predictions with matching index
            y_pred = pd.Series(0.0, index=X_t.index)
            t_pred = pd.DataFrame(0.0, index=X_t.index, columns=T_df.columns)

            # Clone fresh models for this period
            period_treatment_models = []

            # Cross-fit within this period's data
            fold_idx = 0
            for train_idx, test_idx in kfold.split(X_t):
                # Fit outcome model using iloc for positional indexing
                outcome_model_fold = clone(outcome_est)
                outcome_model_fold.fit(X_t.iloc[train_idx], y_t.iloc[train_idx])
                y_pred.iloc[test_idx] = outcome_model_fold.predict(X_t.iloc[test_idx])

                # Fit treatment models for each treatment dimension
                for j, col in enumerate(T_df.columns):
                    treatment_model_fold = clone(treatment_est)
                    treatment_model_fold.fit(
                        X_t.iloc[train_idx], T_t.iloc[train_idx, j]
                    )
                    t_pred.iloc[test_idx, j] = treatment_model_fold.predict(
                        X_t.iloc[test_idx]
                    )

                    if fold_idx == 0:
                        period_treatment_models.append(treatment_model_fold)

                fold_idx += 1

            # Compute residuals for this period (convert to numpy for storage)
            self.residuals_y_[period_mask, t] = (y_t - y_pred).values
            for j in range(n_treatments):
                self.residuals_t_[period_mask, j, t] = (
                    T_t.iloc[:, j] - t_pred.iloc[:, j]
                ).values

            # Store models (use last fold's models as representative)
            self.outcome_est_.append(outcome_model_fold)
            self.treatment_est_.append(period_treatment_models)

    def _fit_parameters(self, effect_est):
        """
        STAGE 2: Backward recursive parameter estimation (peeling).

        ALGORITHM:
        For t = m, m-1, ..., 1:
            1. Compute calibrated outcome:
                y_calibrated = residuals_y_[period_t_mask]
                For j = t+1 to m:
                    y_calibrated -= coef_[j] @ residuals_t_[period_t_mask, j, t]
            2. Get treatment residuals at t:
                T_res_t = residuals_t_[period_t_mask, t, t]
            3. Fit effect estimator:
                effect_est.fit(T_res_t, y_calibrated)
                If effect_est has coef_ attribute, extract coef_[t]

        Parameters
        ----------
        effect_est : estimator
            Effect estimation model (e.g., LinearRegression(fit_intercept=False)).

        Sets
        ----
        self.effect_est_ : estimator
            Fitted effect estimation model (last period's fit).
        self.coef_ : ndarray of shape (n_periods, n_treatments), optional
            Estimated coefficients if effect_est has coef_ attribute.
        """

        # Initialize coef_ storage (will be populated if effect_est has coef_)
        coef_list = []

        # Loop backwards from m to 1
        for t in reversed(range(self.n_periods_)):
            # Get period mask
            period_mask = self.groups_ == t

            # Compute calibrated outcome (peel off future effects)
            y_calibrated = self.residuals_y_[period_mask, t].copy()

            for j in range(t + 1, self.n_periods_):
                # Subtract future treatment effects
                # For models with coef_, use stored coefficients
                if len(coef_list) > 0:
                    future_coef_idx = self.n_periods_ - 1 - j
                    future_effects = np.sum(
                        coef_list[future_coef_idx]
                        * self.residuals_t_[period_mask, :, j],
                        axis=1,
                    )
                    y_calibrated -= future_effects

            # Get treatment residuals at period t
            T_res_t = self.residuals_t_[period_mask, :, t]

            # Fit effect model
            period_effect_est = clone(effect_est)
            period_effect_est.fit(T_res_t, y_calibrated)

            # Store coefficient if available
            if hasattr(period_effect_est, "coef_"):
                coef_list.append(period_effect_est.coef_)

        # Store the effect estimator (use last fitted one)
        self.effect_est_ = period_effect_est

        # If coefficients were extracted, store them in forward order
        if len(coef_list) > 0:
            self.coef_ = np.array(list(reversed(coef_list)))

    def _compute_covariance(self):
        """
        Compute asymptotic variance V = J^{-1} Sigma J^{-T}.

        ALGORITHM:
        1. Initialize block matrices J and Sigma
        2. For each period pair (t, j):
            a. Compute Jacobian block:
                J[t, j] = (1/n) * residuals_t_[:, j, t].T @ residuals_t_[:, t, t]
            b. Compute final residuals:
                epsilon_t = y_calibrated[t] - coef_[t] @ residuals_t_[:, t, t]
            c. Compute variance block:
                Sigma[t, j] = (1/n) * sum(epsilon_t[i] * epsilon_j[i] *
                                           residuals_t_[i, t, t] @ residuals_t_[i, j, j].T)
        3. Invert: V = J^{-1} @ Sigma @ J^{-T}

        Sets
        ----
        self.covariance_ : ndarray of shape (n_periods * n_treatments, n_periods * n_treatments)
        """
        n_treatments = self.coef_.shape[1]
        total_dim = self.n_periods_ * n_treatments

        # Initialize matrices
        J = np.zeros((total_dim, total_dim))
        Sigma = np.zeros((total_dim, total_dim))

        # Compute final residuals per period
        final_residuals = {}
        for t in range(self.n_periods_):
            period_mask = self.groups_ == t
            y_calib = self.residuals_y_[period_mask, t].copy()

            for j in range(t + 1, self.n_periods_):
                future_effects = np.sum(
                    self.coef_[j] * self.residuals_t_[period_mask, :, j], axis=1
                )
                y_calib -= future_effects

            # Compute residuals: epsilon_t = y_calib - coef_[t] @ T_res_t
            T_res_t = self.residuals_t_[period_mask, :, t]
            epsilon_t = y_calib - np.sum(self.coef_[t] * T_res_t, axis=1)
            final_residuals[t] = epsilon_t

        # Fill block matrices
        for t in range(self.n_periods_):
            period_mask_t = self.groups_ == t
            n_t = np.sum(period_mask_t)

            for j in range(self.n_periods_):
                period_mask_j = self.groups_ == j

                # Indices for blocks
                t_start = t * n_treatments
                t_end = (t + 1) * n_treatments
                j_start = j * n_treatments
                j_end = (j + 1) * n_treatments

                # Jacobian block (approximate with period t data)
                T_res_t = self.residuals_t_[period_mask_t, :, t]
                if t == j:
                    J[t_start:t_end, j_start:j_end] = (1 / n_t) * (T_res_t.T @ T_res_t)

                # Variance block
                if period_mask_t.shape[0] == period_mask_j.shape[0]:
                    epsilon_t = final_residuals[t]
                    epsilon_j = final_residuals[j]
                    T_res_j = self.residuals_t_[period_mask_j, :, j]

                    for i in range(n_t):
                        Sigma[t_start:t_end, j_start:j_end] += (
                            epsilon_t[i]
                            * epsilon_j[i]
                            * np.outer(T_res_t[i], T_res_j[i])
                        ) / n_t

        # Invert and store (with regularization for stability)
        try:
            J_inv = np.linalg.inv(J + np.eye(total_dim) * 1e-6)
            self.covariance_ = J_inv @ Sigma @ J_inv.T
        except np.linalg.LinAlgError:
            warnings.warn(
                "Covariance matrix computation failed, using identity matrix. "
                "This may indicate insufficient data or numerical instability."
            )
            self.covariance_ = np.eye(total_dim)
