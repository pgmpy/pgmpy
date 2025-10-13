#!/usr/bin/env python3
"""
Dynamic Double/Debiased Machine Learning for Sequential Treatment Effects.
"""

import warnings

import numpy as np
from scipy.stats import norm
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold
from sklearn.utils.validation import check_array, check_is_fitted


class DynamicDMLRegressor(RegressorMixin, BaseEstimator):
    """
    Dynamic Double/Debiased Machine Learning for sequential treatment effects.
    Estimates causal effects of treatments at different time periods on a
    final outcome using recursive g-estimation with Neyman orthogonal moments.

    Parameters
    ----------
    causal_graph : DAG, PDAG, ADMG, MAG, or PAG, optional
        Causal graph with defined variable roles. If provided, the graph must have
        the following roles: `exposure`, `outcome`. Additionally, `adjustment` and
        `pretreatment` can be specified for confounders.
        If None, all features in X are treated as confounders, and T/y column names
        are used directly.

    model_y : estimator or 'auto', default='auto'
        Machine learning model for outcome E[Y | X]. Must implement fit() and predict().
        If 'auto': LinearRegression for continuous outcomes, RandomForestRegressor for discrete.

    model_t : estimator or 'auto', default='auto'
        Machine learning model for treatment E[T | X]. Must implement fit() and predict().
        If 'auto': LinearRegression for continuous treatments, RandomForestClassifier for discrete.

    cv : int, default=2
        Number of cross-fitting folds (minimum 2). Internally uses GroupKFold to respect
        panel structure when groups parameter is provided.

    n_periods : int or None, default=None
        Number of treatment periods. If None, inferred from unique values in groups.

    discrete_treatment : bool, default=False
        Whether treatments are discrete (affects automatic model selection when model_t='auto').

    discrete_outcome : bool, default=False
        Whether outcome is discrete (affects automatic model selection when model_y='auto').

    random_state : int, RandomState instance or None, default=None
        Controls randomness of cross-fitting splits.

    Attributes
    ----------
    exposure_var_ : str or None
        Name of the exposure (treatment) variable extracted from causal_graph.

    outcome_var_ : str or None
        Name of the outcome variable extracted from causal_graph.

    adjustment_vars_ : list of str
        Names of adjustment (confounder) variables extracted from causal_graph.

    pretreatment_vars_ : list of str
        Names of pretreatment variables extracted from causal_graph.

    feature_columns_ : list
        Names/indices of features used in the model (exposure + adjustments + pretreatment).

    coef_ : ndarray of shape (n_periods, n_treatments)
        Estimated dynamic treatment effect parameters {ψ₁, ..., ψₘ}.
        coef_[t] is the effect of treatment at period t on final outcome.

    models_y_ : list of length n_periods
        Fitted outcome models for each period.

    models_t_ : list of length n_periods
        Nested list structure. models_t_[t] is a list of fitted treatment models
        for periods j >= t. models_t_[t][j-t] predicts T_j given history up to t.

    residuals_y_ : ndarray of shape (n_samples, n_periods)
        Cached outcome residuals Y - q̂_t(X_t) for each period for diagnostics.

    residuals_t_ : ndarray of shape (n_samples, n_periods, n_periods)
        Cached treatment residuals. residuals_t_[i, j, t] = T_j^i - p̂_{j,t}(X_t^i).

    covariance_ : ndarray of shape (n_periods * n_treatments, n_periods * n_treatments)
        Asymptotic covariance matrix V = J^{-1}ΣJ^{-T} for inference.

    n_features_in_ : int
        Number of features seen during fit (sklearn standard).

    feature_names_in_ : ndarray of shape (n_features_in_,)
        Names of features if X is pandas DataFrame (sklearn standard).

    n_samples_ : int
        Number of samples seen during fit.

    groups_ : ndarray
        Stored groups array from fit().

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.base import DAG
    >>> from pgmpy.prediction import DynamicDMLRegressor
    >>>


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
        model_y="auto",
        model_t="auto",
        cv=2,
        n_periods=None,
        discrete_treatment=False,
        discrete_outcome=False,
        random_state=None,
    ):
        self.causal_graph = causal_graph
        self.model_y = model_y
        self.model_t = model_t
        self.cv = cv
        self.n_periods = n_periods
        self.discrete_treatment = discrete_treatment
        self.discrete_outcome = discrete_outcome
        self.random_state = random_state

    def __sklearn_tags__(self):
        """Tags for sklearn compatibility."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = False
        tags.regressor_tags.poor_score = True
        return tags

    def _prepare_feature_df(self, X):
        """
        Convert input (either numpy array or dataframe) to a DataFrame and
        validate that column names match DAG variables when causal_graph is provided.

        Parameters
        ----------
        X : array-like or DataFrame
            Input features.

        Returns
        -------
        X_df : DataFrame
            Feature DataFrame with validated columns.
        """
        # If no causal graph return as-is (will be validated elsewhere)
        if self.causal_graph is None:
            try:
                import pandas as pd

                if isinstance(X, pd.DataFrame):
                    return X
            except ImportError:
                pass

            X_arr = np.asarray(X)
            if X_arr.ndim == 1:
                raise ValueError(
                    "Reshape your data: X must be 2D. If using a 1D array, reshape it to (n_samples, 1)."
                )
            return X_arr

        # Get required feature columns from causal graph
        required_features = self.feature_columns_

        # Convert input to DataFrame format
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

        # Validation: column names must match DAG variables
        missing_columns = set(required_features) - set(X_df.columns)
        if missing_columns:
            raise ValueError(
                f"Missing required columns in input data: {list(missing_columns)}. "
                f"DAG expects columns: {required_features}, but got: {list(X_df.columns)}"
            )

        return X_df[required_features]

    def fit(self, X, T, y, groups):
        """
        Fit the DynamicDMLRegressor model using panel data.

        If causal_graph is provided:
        1. Extract exposure, outcome, and adjustment variables from graph roles
        2. Validate that exactly one exposure and one outcome are defined
        3. Use graph structure to guide feature selection

        Algorithm:
        1. Validate inputs
        2. Extract roles from causal_graph (if provided)
        3. Convert to numpy arrays
        4. Infer n_periods from groups if not provided
        5. Select models if 'auto'
        6. STAGE 1: Cross-fit nuisances (_fit_nuisances)
        7. STAGE 2: Backward recursive parameter estimation (_fit_parameters)
        8. Compute asymptotic covariance (_compute_covariance)
        9. Set sklearn-required attributes
        10. Return self

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Time-varying confounders/states. Each row is observation at time t for unit i.
            If causal_graph is provided, X must contain columns for exposure and adjustment variables.

        T : array-like of shape (n_samples, n_treatments) or (n_samples,)
            Treatments for each observation. Will be reshaped to 2D if 1D.
            If causal_graph is provided, this should match the exposure variable from the graph.

        y : array-like of shape (n_samples,)
            Final outcome for each observation. Typically replicated across periods
            for the same unit (final outcome measured at end).

        groups : array-like of shape (n_samples,)
            Period identifier (0 to m-1) for each observation.
            REQUIRED for panel structure.

        Returns
        -------
        self : object
            Fitted estimator with coef_ attribute.
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
            self.feature_columns_ = (
                [self.exposure_var_] + self.adjustment_vars_ + self.pretreatment_vars_
            )
        else:
            # No causal graph: use all features as confounders
            self.exposure_var_ = None
            self.outcome_var_ = None
            self.adjustment_vars_ = []
            self.pretreatment_vars_ = []
            self.feature_columns_ = None

        # STEP 3: Convert to numpy arrays
        if hasattr(X, "columns"):
            self.feature_names_in_ = np.array(X.columns)
            X = X.values
        else:
            X = np.asarray(X)

        T = np.asarray(T)
        y = np.asarray(y)
        groups = np.asarray(groups)

        # Reshape T to 2D if 1D
        if T.ndim == 1:
            T = T.reshape(-1, 1)

        self.groups_ = groups

        # STEP 4: Infer n_periods if not provided
        if self.n_periods is None:
            self.n_periods_ = len(np.unique(groups))
        else:
            self.n_periods_ = self.n_periods

        # STEP 5: Select models if 'auto'
        self.model_y_ = self._select_model(self.model_y, self.discrete_outcome)
        self.model_t_ = self._select_model(self.model_t, self.discrete_treatment)

        # STEP 6: STAGE 1 - Cross-fitted nuisance estimation
        self._fit_nuisances(X, T, y, groups)

        # STEP 7: STAGE 2 - Backward recursive parameter estimation
        self._fit_parameters()

        # STEP 8: Compute asymptotic covariance
        self._compute_covariance()

        # STEP 9: Set sklearn-required attributes
        self.n_features_in_ = X.shape[1]  # Number of features (columns)
        self.n_samples_ = X.shape[0]  # Number of samples (rows)

        # STEP 10: Return self
        return self

    def predict(self, X, T0=None, T1=None):
        """
        Predict treatment effects.

        ALGORITHM:
        Compute E[Y | do(T=T1)] - E[Y | do(T=T0)] using estimated coef_.
        Effect = sum over periods t of coef_[t] @ (T1_t - T0_t)

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Features for prediction.

        T0 : array-like of shape (n_samples, n_treatments) or scalar, optional
            Baseline treatment level. If None, uses zeros.

        T1 : array-like of shape (n_samples, n_treatments) or scalar, optional
            Alternative treatment level. If None, uses ones.

        Returns
        -------
        effects : ndarray of shape (n_samples,)
            Estimated treatment effect for each sample.

        Notes
        -----
        Based on counterfactual decomposition (Lemma 2.1 of paper).
        """
        check_is_fitted(self, ["coef_", "n_periods_"])
        X = check_array(X)

        n_treatments = self.coef_.shape[1]

        # Handle T0, T1
        if T0 is None:
            T0 = np.zeros((X.shape[0], n_treatments))
        elif np.isscalar(T0):
            T0 = np.full((X.shape[0], n_treatments), T0)
        else:
            T0 = np.asarray(T0)
            if T0.ndim == 1:
                T0 = T0.reshape(-1, 1)

        if T1 is None:
            T1 = np.ones((X.shape[0], n_treatments))
        elif np.isscalar(T1):
            T1 = np.full((X.shape[0], n_treatments), T1)
        else:
            T1 = np.asarray(T1)
            if T1.ndim == 1:
                T1 = T1.reshape(-1, 1)

        # Compute effect
        effects = np.zeros(X.shape[0])
        for t in range(self.n_periods_):
            effects += (T1 - T0) @ self.coef_[t]

        return effects

    def effect(self, X, T0=0, T1=1):
        """
        Convenience method for predict() with default contrasts.

        Parameters
        ----------
        X : array-like
            Features.
        T0, T1 : scalar or array-like
            Treatment contrasts.

        Returns
        -------
        effects : ndarray
            Treatment effects.
        """
        return self.predict(X, T0=T0, T1=T1)

    def effect_interval(self, X, T0=0, T1=1, alpha=0.05):
        """
        Compute (1-α) confidence intervals for treatment effects.

        ALGORITHM:
        Use asymptotic normality: effect ± z_α/2 * sqrt(variance)
        Variance comes from self.covariance_

        Parameters
        ----------
        X : array-like
            Features.
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

        # Compute predictions
        y_pred = self.predict(X, T0=0, T1=T)

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

    def _select_model(self, model, is_discrete):
        """
        Auto-select model when model='auto'.

        RULES:
        - Continuous + 'auto' -> LinearRegression()
        - Discrete + 'auto' -> RandomForestClassifier() or RandomForestRegressor()

        Parameters
        ----------
        model : estimator or 'auto'
            Model specification.
        is_discrete : bool
            Whether outcome/treatment is discrete.

        Returns
        -------
        selected_model : estimator
            Cloned estimator ready to use.
        """
        if model == "auto":
            if is_discrete:
                return RandomForestRegressor(
                    n_estimators=100, random_state=self.random_state
                )
            else:
                return LinearRegression()
        else:
            return clone(model)

    def _fit_nuisances(self, X, T, y, groups):
        """
        STAGE 1: Cross-fitted nuisance estimation.

        ALGORITHM:
        For each period t = 1, ..., m:
            1. Create GroupKFold cross-validator with groups
            2. Initialize storage for predictions
            3. For each fold (train_idx, test_idx):
                a. Fit outcome model on train fold: model_y.fit(X[train_idx], y[train_idx])
                b. Predict on test fold: y_pred[test_idx] = model_y.predict(X[test_idx])
                c. For each future period j >= t:
                    - Fit treatment model: model_t.fit(X[train_idx], T[train_idx, j])
                    - Predict: t_pred[test_idx, j] = model_t.predict(X[test_idx])
            4. Compute residuals:
                - residuals_y_[period_t_mask] = y[period_t_mask] - y_pred[period_t_mask]
                - residuals_t_[period_t_mask, j, t] = T[period_t_mask, j] - t_pred[period_t_mask, j]
            5. Store fitted models in self.models_y_ and self.models_t_

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
        T : ndarray of shape (n_samples, n_treatments)
        y : ndarray of shape (n_samples,)
        groups : ndarray of shape (n_samples,)

        Sets
        ----
        self.residuals_y_ : ndarray of shape (n_samples, n_periods)
        self.residuals_t_ : ndarray of shape (n_samples, n_periods, n_periods)
        self.models_y_ : list of fitted models
        self.models_t_ : list of lists of fitted models
        """
        n_samples = X.shape[0]
        n_treatments = T.shape[1]

        # Initialize storage
        self.residuals_y_ = np.zeros((n_samples, self.n_periods_))
        self.residuals_t_ = np.zeros((n_samples, n_treatments, self.n_periods_))
        self.models_y_ = []
        self.models_t_ = []

        # Setup cross-validator
        if self.cv < 2:
            raise ValueError("cv must be at least 2 for cross-fitting")
        kfold = GroupKFold(n_splits=self.cv)

        # Loop over periods
        for t in range(self.n_periods_):
            # Get mask for current period
            period_mask = groups == t
            X_t = X[period_mask]
            y_t = y[period_mask]
            T_t = T[period_mask]

            y_pred = np.zeros(np.sum(period_mask))
            t_pred = np.zeros((np.sum(period_mask), n_treatments))

            # Clone fresh models for this period
            period_models_t = []

            # Cross-fit
            fold_idx = 0
            for train_idx, test_idx in kfold.split(X_t, y_t, groups[period_mask]):
                # Fit outcome model
                model_y_fold = clone(self.model_y_)
                model_y_fold.fit(X_t[train_idx], y_t[train_idx])
                y_pred[test_idx] = model_y_fold.predict(X_t[test_idx])

                # Fit treatment models for j >= t
                for j in range(n_treatments):
                    model_t_fold = clone(self.model_t_)
                    model_t_fold.fit(X_t[train_idx], T_t[train_idx, j])
                    t_pred[test_idx, j] = model_t_fold.predict(X_t[test_idx])

                    if fold_idx == 0:
                        period_models_t.append(model_t_fold)

                fold_idx += 1

            # Compute residuals for this period
            self.residuals_y_[period_mask, t] = y_t - y_pred
            for j in range(n_treatments):
                self.residuals_t_[period_mask, j, t] = T_t[:, j] - t_pred[:, j]

            # Store models (use last fold's models as representative)
            self.models_y_.append(model_y_fold)
            self.models_t_.append(period_models_t)

    def _fit_parameters(self):
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
            3. Fit effect estimator (OLS without intercept):
                effect_model = LinearRegression(fit_intercept=False)
                effect_model.fit(T_res_t.reshape(-1, 1), y_calibrated)
                coef_[t] = effect_model.coef_

        Sets
        ----
        self.coef_ : ndarray of shape (n_periods, n_treatments)
        """
        n_treatments = self.residuals_t_.shape[1]
        self.coef_ = np.zeros((self.n_periods_, n_treatments))

        # Loop backwards from m to 1
        for t in reversed(range(self.n_periods_)):
            # Get period mask
            period_mask = self.groups_ == t

            # Compute calibrated outcome (peel off future effects)
            y_calibrated = self.residuals_y_[period_mask, t].copy()

            for j in range(t + 1, self.n_periods_):
                # Subtract future treatment effects
                # coef_[j] is shape (n_treatments,)
                # residuals_t_[period_mask, :, j] is shape (n_samples_t, n_treatments)
                future_effects = np.sum(
                    self.coef_[j] * self.residuals_t_[period_mask, :, j], axis=1
                )
                y_calibrated -= future_effects

            # Get treatment residuals at period t
            T_res_t = self.residuals_t_[period_mask, :, t]

            # Fit effect model (OLS without intercept)
            effect_model = LinearRegression(fit_intercept=False)
            effect_model.fit(T_res_t, y_calibrated)
            self.coef_[t] = effect_model.coef_

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
