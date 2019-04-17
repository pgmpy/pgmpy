import statsmodels.api as sm
from statsmodels.api import OLS


class LinearEstimator(object):
    """
    A simple linear model built on statmodels.
    """
    def __init__(self, graph, **kwargs):
        pass

    def _model(self, X, Y, Z, data, **kwargs):
        exog = sm.add_constant(data[[X] + list(Z)])
        endog = data[Y]
        return OLS(endog=endog, exog=exog, **kwargs)

    def fit(self, X, Y, Z, data, **kwargs):
        self.estimator = self._model(X, Y, Z, data, **kwargs).fit()
        self.ate = self.estimator.params[X]
        return self

    def _get_ate(self):
        return self.ate

    def summary(self):
        return self.estimator.summary()
