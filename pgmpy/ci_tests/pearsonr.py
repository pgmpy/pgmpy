import numpy as np
import pandas as pd
from scipy import stats

from ._base import _BaseCITest


class Pearsonr(_BaseCITest):
    """
    Compute Pearson correlation coefficient and p-value for testing non-correlation.

    Should be used only on continuous data. In case when :math:`Z \\neq \\emptyset` uses
    linear regression and computes pearson coefficient on residuals.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    Attributes
    ----------
    statistic_ : float
        Pearson's correlation coefficient (or partial correlation when Z is non-empty),
        ranging from -1 to 1. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. [2] https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    """

    _tags = {
        "name": "pearsonr",
        "data_types": ("continuous",),
        "default_for": "continuous",
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame):
        self.data = data
        super().__init__()

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute Pearson correlation coefficient and p-value.

        Sets ``self.statistic_`` (Pearson's r) and ``self.p_value_``.
        """
        data = self.data

        # Step 1: If Z is empty compute a non-conditional test.
        if len(Z) == 0:
            coef, p_value = stats.pearsonr(data.loc[:, X], data.loc[:, Y])

        # Step 2: If Z is non-empty, use linear regression to compute residuals and test independence on it.
        else:
            X_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, X], rcond=None)[0]
            Y_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, Y], rcond=None)[0]

            residual_X = data.loc[:, X] - data.loc[:, Z].dot(X_coef)
            residual_Y = data.loc[:, Y] - data.loc[:, Z].dot(Y_coef)
            coef, p_value = stats.pearsonr(residual_X, residual_Y)

        self.statistic_ = coef
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
