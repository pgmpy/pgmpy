import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression

from pgmpy.utils import preprocess_data

from ._base import _BaseCITest, _CITestResult, _ResidualMixin


class GCM(_ResidualMixin, _BaseCITest):
    r"""
    Generalized Covariance Measure (GCM) [1] test for conditional independence.

    Fit an estimator on :math:`X` and :math:`Y` on :math:`[1, Z]`, let :math:`r_X` and :math:`r_Y` denote the
    resulting residuals, and define :math:`U_i = r_{X, i} r_{Y, i}`. The resulting test statistic is

    .. math::
        T = \frac{1}{\sqrt{n}} \frac{\sum_{i=1}^n U_i}{\operatorname{std}(U_1, \ldots, U_n)},

    where :math:`n` is the sample size. Under the null hypothesis :math:`X \perp Y \mid Z`, this statistic is
    asymptotically standard normal.

    The effect size is correlation coefficient between the residuals: :math:`\textit{cor}(r_X, r_Y)`.

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
    effect_size_ : float
        Cohen's d. Set after calling the test.
    estimator_x_ : sklearn-compatible estimator
        The fitted estimator used for predicting X.
    estimator_y_ : sklearn-compatible estimator
        The fitted estimator used for predicting Y.

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

    def __init__(self, data: pd.DataFrame, estimator=None, use_cache: bool = True):
        self.data, self.dtypes = preprocess_data(data)
        self.estimator = LinearRegression() if estimator is None else estimator

        super().__init__(use_cache=use_cache)

    def _compute_result(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute GCM statistic and p-value.

        Returns the t-statistic and p-value.
        """
        # Step 1: Compute residuals of X and Y given Z.
        res_x, self.estimator_x_ = self.get_residuals(X, Z)
        res_y, self.estimator_y_ = self.get_residuals(Y, Z)
        res_x = np.asarray(res_x)
        res_y = np.asarray(res_y)

        # Step 2: Compute the Generalised Covariance Measure.
        n = res_x.shape[0]
        t_stat = (1 / np.sqrt(n)) * np.dot(res_x, res_y) / np.std(res_x * res_y)

        # Step 3: Compute p-value using standard normal distribution.
        p_value = 2 * stats.norm.sf(np.abs(t_stat))

        # Step 4: Compute effect size as correlation coefficient between residuals.
        effect_size = np.absolute(np.corrcoef(res_x, res_y, rowvar=False)[0, 1])

        return _CITestResult(statistic=t_stat, p_value=p_value, effect_size=effect_size)
