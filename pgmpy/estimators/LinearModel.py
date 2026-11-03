from skbase.utils.dependencies import _check_soft_dependencies, _safe_import

add_constant = _safe_import("statsmodels.api.add_constant")
GLS = _safe_import("statsmodels.api.GLS")
OLS = _safe_import("statsmodels.api.OLS")
WLS = _safe_import("statsmodels.api.WLS")


class LinearEstimator:
    """
    A simple linear model built on statmodels.
    """

    def __init__(self, graph, estimator_type="linear", **kwargs):

        _check_soft_dependencies("statsmodels", obj=self)
        self._supported_models = {"linear": OLS, "OLS": OLS, "GLS": GLS, "WLS": WLS}
        if estimator_type not in self._supported_models.keys():
            raise NotImplementedError(
                "We currently only support OLS, GLS, and WLS. Please specify which you would like to use."
            )
        else:
            self.estimator = self._supported_models[estimator_type]

    def _model(self, X, Y, Z, data, **kwargs):
        exog = add_constant(data[[X] + list(Z)])
        endog = data[Y]
        return self.estimator(endog=endog, exog=exog, **kwargs)

    def fit(self, X, Y, Z, data, **kwargs):
        self.estimator = self._model(X, Y, Z, data, **kwargs).fit()
        self.ate = self.estimator.params[X]
        return self

    def _get_ate(self):
        return self.ate

    def summary(self):
        return self.estimator.summary()
