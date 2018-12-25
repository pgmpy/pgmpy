import pandas as pd

from pgmpy.models import SEM
from pgmpy.data import Data


class SEMEstimator(object):
    """
    Base class of SEM estimators. All the estimators inherit this class.
    """
    def __init__(self, model):
        if not isinstance(model, SEM):
            raise ValueError("model should be an instance of SEM class. Got type: {t}".format(t=type(model)))

    def get_ols_fn(self):
        pass

    def get_uls_fn(self):
        pass

    def get_gls_fn(self):
        pass

    def fit(self, data, method):
        """
        Estimate the parameters of the model from the data.

        Parameters
        ----------
        data: pandas DataFrame or pgmpy.data.Data instance
            The data from which to estimate the parameters of the model.

        method: str ("ols"|"uls"|"gls"|"2sls")
            The fitting function to use.
            OLS: Ordinary Least Squares/Maximum Likelihood
            ULS: Unweighted Least Squares
            GLS: Generalized Least Squares
            2sls: 2-SLS estimator

        Returns
        -------
            pgmpy.model.SEM instance: Instance of the model with estimated parameters
        """
        if not isinstance(data, [pd.DataFrame, Data]):
            raise ValueError("data must be a pandas DataFrame. Got type: {t}".format(t=type(data)))

        if method == 'ols':
            minimization_fun = self.get_ols_fn()

        elif method == 'uls':
            minimization_fun = self.get_uls_fn()

        elif method == 'gls':
            minimization_fun = self.get_gls_fn()

        elif method == '2sls':
            raise NotImplementedError("2-SLS is not implemented yet")
