import numpy as np
import pandas as pd
from scipy import stats

from ._base import _BaseCITest


class GCM(_BaseCITest):
    r"""
    Generalized Covariance Measure (GCM) test for conditional independence.

    Regress :math:`X` and :math:`Y` on :math:`[1, Z]` using least squares, let :math:`r_X` and :math:`r_Y` denote the
    resulting residuals, and define :math:`U_i = r_{X, i} r_{Y, i}`. The resulting test statistic is

    .. math::
        T = \frac{1}{\sqrt{n}} \frac{\sum_{i=1}^n U_i}{\operatorname{std}(U_1, \ldots, U_n)},

    where :math:`n` is the sample size. Under the null hypothesis :math:`X \perp Y \mid Z`, this statistic is
    asymptotically standard normal. Details of the method can be found in [1].

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    Attributes
    ----------
    statistic_ : float
        The GCM test statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.

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

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute GCM statistic and p-value.

        Sets ``self.statistic_`` (t-statistic) and ``self.p_value_``.
        """
        # Step 1.1: Add another column with constant values to handle intercepts.
        data = self.data
        Z_aug = list(Z) + ["intercept"]
        data_aug = data.assign(intercept=np.ones(data.shape[0]))

        # Step 2: Compute the linear regression and the residuals
        X_coef = np.linalg.lstsq(data_aug.loc[:, Z_aug], data_aug.loc[:, X], rcond=None)[0]
        Y_coef = np.linalg.lstsq(data_aug.loc[:, Z_aug], data_aug.loc[:, Y], rcond=None)[0]
        res_x = data_aug.loc[:, X] - data_aug.loc[:, Z_aug].dot(X_coef)
        res_y = data_aug.loc[:, Y] - data_aug.loc[:, Z_aug].dot(Y_coef)

        # Step 3: Compute the Generalised Covariance Measure.
        n = res_x.shape[0]
        t_stat = (1 / np.sqrt(n)) * np.dot(res_x, res_y) / np.std(res_x * res_y)

        # Step 4: Compute p-value using standard normal distribution.
        p_value = 2 * stats.norm.sf(np.abs(t_stat))

        self.statistic_ = t_stat
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
