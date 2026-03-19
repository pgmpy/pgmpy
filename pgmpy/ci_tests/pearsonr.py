from typing import Tuple

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

    Returns
    -------
    result : bool or tuple
        If boolean=True, returns True if p-value >= significance_level, else False.
        If boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value).

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

    def _compute_statistic(
        self,
        X: str,
        Y: str,
        Z: list,
        **kwargs,
    ) -> Tuple[float, float]:
        """
        Compute Pearson correlation coefficient and p-value.

        Parameters
        ----------
        X : str
            The first variable for testing the independence condition X ⊥⊥ Y | Z.
        Y : str
            The second variable for testing the independence condition X ⊥⊥ Y | Z.
        Z : list
            A list of conditional variables for testing the condition X ⊥⊥ Y | Z.
        Returns
        -------
        tuple
            A tuple of (Pearson's correlation Coefficient, p-value).
        """
        # Step 1: If Z is empty compute a non-conditional test.
        data = self.data
        if len(Z) == 0:
            coef, p_value = stats.pearsonr(data.loc[:, X], data.loc[:, Y])

        # Step 2: If Z is non-empty, use linear regression to compute residuals and test independence on it.
        else:
            X_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, X], rcond=None)[0]
            Y_coef = np.linalg.lstsq(data.loc[:, Z], data.loc[:, Y], rcond=None)[0]

            residual_X = data.loc[:, X] - data.loc[:, Z].dot(X_coef)
            residual_Y = data.loc[:, Y] - data.loc[:, Z].dot(Y_coef)
            coef, p_value = stats.pearsonr(residual_X, residual_Y)

        return coef, p_value
