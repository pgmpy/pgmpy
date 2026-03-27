import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.linear_model import LinearRegression

from ._base import _BaseCITest


class GCM(_BaseCITest):
    r"""
    Generalized Covariance Measure (GCM) [1] test for conditional independence.

    Fit an estimator on :math:`X` and :math:`Y` on :math:`[1, Z]`, let :math:`r_X` and :math:`r_Y` denote the
    resulting residuals, and define :math:`U_i = r_{X, i} r_{Y, i}`. The resulting test statistic is

    .. math::
        T = \frac{1}{\sqrt{n}} \frac{\sum_{i=1}^n U_i}{\operatorname{std}(U_1, \ldots, U_n)},

    where :math:`n` is the sample size. Under the null hypothesis :math:`X \perp Y \mid Z`, this statistic is
    asymptotically standard normal.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.
    estimator: optional (default=None)
        Any regressor with fit and predict methods to compute residuals. If None, LinearRegression() is used
        as default.

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

    def __init__(self, data: pd.DataFrame, estimator=None):
        self.data = data

        if estimator is None:
            self.estimator = LinearRegression()
        else:
            # Check if estimator is sklearn compatible.
            required_methods = ["fit", "predict", "get_params", "set_params"]
            if not all(hasattr(estimator, method) for method in required_methods):
                raise ValueError(
                    "`estimator` must be a scikit-learn compatible.",
                    "It must have fit, predict methods and be clonable.",
                )
            self.estimator = estimator

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
        # Step 1: Append intercept column to ensure Z is never empty
        Z_data = np.column_stack([self.data.loc[:, list(Z)].values, np.ones(self.data.shape[0])])

        # Step 2: Compute residuals using the provided estimator
        est_x = clone(self.estimator)
        est_y = clone(self.estimator)
        est_x.fit(Z_data, self.data.loc[:, X])
        est_y.fit(Z_data, self.data.loc[:, Y])
        res_x = self.data.loc[:, X] - est_x.predict(Z_data)
        res_y = self.data.loc[:, Y] - est_y.predict(Z_data)

        # Step 3: Compute the Generalised Covariance Measure.
        n = res_x.shape[0]
        t_stat = (1 / np.sqrt(n)) * np.dot(res_x, res_y) / np.std(res_x * res_y)

        # Step 4: Compute p-value using standard normal distribution.
        p_value = 2 * stats.norm.sf(np.abs(t_stat))

        self.statistic_ = t_stat
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
