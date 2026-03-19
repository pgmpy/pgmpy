from typing import Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ._base import _BaseCITest


class GCM(_BaseCITest):
    """
    The Generalized Covariance Measure(GCM) test for CI.

    It performs linear regressions on the conditioning variable and then tests
    for a vanishing covariance between the resulting residuals. Details of the
    method can be found in [1].

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset in which to test the independence condition.


    Returns
    -------
    CI Test results: tuple or bool
        If boolean=True, returns True if p-value >= significance_level, else False. If
        boolean=False, returns a tuple of (Pearson's correlation Coefficient, p-value)

    References
    ----------
    .. [1] Rajen D. Shah, and Jonas Peters. "The Hardness of Conditional Independence Testing and the Generalised
        Covariance Measure".
    """

    _tags = {
        "name": "gcm",
        "data_types": ("continuous",),
        "default_for": None,
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
        Compute GCM statistic and p-value.

        Parameters
        ----------
        X : str
            The first variable for testing the independence condition X ⊥⊥ Y | Z.
        Y : str
            The second variable for testing the independence condition X ⊥⊥ Y | Z.
        Z : list
            A list of conditional variables for testing the condition X ⊥⊥ Y | Z.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple
            A tuple of (t-statistic, p-value).
        """
        # Step 1.1: Add another column with constant values to handle intercepts.
        data = self.data
        Z_aug = Z + ["intercept"]
        data_aug = data.assign(intercept=np.ones(data.shape[0]))

        # Step 2: Compute the linear regression and the residuals
        X_coef = np.linalg.lstsq(
            data_aug.loc[:, Z_aug], data_aug.loc[:, X], rcond=None
        )[0]
        Y_coef = np.linalg.lstsq(
            data_aug.loc[:, Z_aug], data_aug.loc[:, Y], rcond=None
        )[0]
        res_x = data_aug.loc[:, X] - data_aug.loc[:, Z_aug].dot(X_coef)
        res_y = data_aug.loc[:, Y] - data_aug.loc[:, Z_aug].dot(Y_coef)

        # Step 3: Compute the Generalised Covariance Measure.
        n = res_x.shape[0]
        t_stat = (1 / np.sqrt(n)) * np.dot(res_x, res_y) / np.std(res_x * res_y)

        # Step 4: Compute p-value using standard normal distribution.
        p_value = 2 * (1 - stats.norm.cdf(np.abs(t_stat)))

        return t_stat, p_value
