from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

add_constant = _safe_import("statsmodels.api.add_constant")
GLS = _safe_import("statsmodels.api.GLS")
OLS = _safe_import("statsmodels.api.OLS")
WLS = _safe_import("statsmodels.api.WLS")


class LinearEstimator:
    """
    A linear estimator supporting OLS, GLS, and WLS regression models.
    """

    def __init__(self, graph, estimator_type="linear", **kwargs):
        """
        Initialize the linear estimator.

        Parameters
        ----------
        graph : object
            The causal graph used by the estimator.
        estimator_type : str, default="linear"
            The regression model to use: "linear", "OLS", "GLS", or "WLS".
        **kwargs : dict
            Additional arguments reserved for the estimator.
        """
        _check_soft_dependencies("statsmodels>=0.14.5", obj=self)
        self._supported_models = {"linear": OLS, "OLS": OLS, "GLS": GLS, "WLS": WLS}
        if estimator_type not in self._supported_models.keys():
            raise NotImplementedError(
                "We currently only support OLS, GLS, and WLS. Please specify which you would like to use."
            )
        else:
            self.estimator = self._supported_models[estimator_type]

    def _model(self, X, Y, Z, data, **kwargs):
        """
        Construct the regression model.

        Parameters
        ----------
        X : str
            Treatment variable.
        Y : str
            Outcome variable.
        Z : list
            Adjustment variables.
        data : pandas.DataFrame
            Dataset containing the variables.
        **kwargs : dict
            Additional arguments passed to the regression model.

        Returns
        -------
        statsmodels regression model
            The constructed regression model.
        """
        exog = add_constant(data[[X] + list(Z)])
        endog = data[Y]
        return self.estimator(endog=endog, exog=exog, **kwargs)

    def fit(self, X, Y, Z, data, **kwargs):
        """
        Fit the regression model and calculate the average treatment effect.

        Parameters
        ----------
        X : str
            Treatment variable.
        Y : str
            Outcome variable.
        Z : list
            Adjustment variables.
        data : pandas.DataFrame
            Dataset containing the variables.
        **kwargs : dict
            Additional arguments passed to the regression model.

        Returns
        -------
        LinearEstimator
            The fitted estimator.
        """
        self.estimator = self._model(X, Y, Z, data, **kwargs).fit()
        self.ate = self.estimator.params[X]
        return self

    def _get_ate(self):
        """Return the estimated average treatment effect."""
        return self.ate

    def summary(self):
        """Return the regression model summary."""
        return self.estimator.summary()
