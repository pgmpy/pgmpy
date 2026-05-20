import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import chi2
from sklearn.base import clone

from pgmpy.ci_tests._base import _BaseCITest


class ProjectedDistanceCovariance(_BaseCITest):
    r"""
    Projected Distance Covariance (P-DCov) [1] test for conditional independence.

    Regress :math:`X` and :math:`Y` on :math:`Z` (using linear regression) and obtain
    the residuals :math:`r_X` and :math:`r_Y`. Then compute the distance covariance
    between these residuals. The resulting test statistic is:

    .. math::
        T = n \cdot \widehat{V}^2(r_X, r_Y) / S_2(r_X, r_Y),

    where :math:`\widehat{V}^2` denotes the empirical distance covariance between
    :math:`r_X` and :math:`r_Y`, and :math:`n` is the sample size.

    Under the null hypothesis :math:`X \perp Y \mid Z`, the residuals are independent,
    and the test statistic is expected to be small. Larger values indicate dependence.

    The p-value is computed using the asymptotic chi-square(1)
    approximation motivated by Theorem 3 of Fan et al. (2020).

    This implementation using a regression estimator and asymptotic inference.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    estimator : sklearn-like regressor, default=LinearRegression()
        Regression estimator implementing fit and predict
        used to regress X and Y on Z before computing
        distance covariance between residuals.

    Attributes
    ----------
    statistic_ : float
        The P-DCov test statistic. Set after calling the test.
    p_value_ : float
        The asymptotic p-value based on the chi-square(1) approximation.

    Examples
    --------
    >>> from pgmpy.models import LinearGaussianBayesianNetwork
    >>> from pgmpy.factors.continuous import LinearGaussianCPD
    >>> from pgmpy.ci_tests import ProjectedDistanceCovariance
    >>> from sklearn.linear_model import LinearRegression
    >>> # Z -> X, Z -> Y  (Z is the common cause)
    >>> model = LinearGaussianBayesianNetwork([("Z", "X"), ("Z", "Y")])
    >>> model.add_cpds(
    ...     LinearGaussianCPD("Z", [0], 1),
    ...     LinearGaussianCPD("X", [0, 1], 1, ["Z"]),
    ...     LinearGaussianCPD("Y", [0, 1], 1, ["Z"]),
    ... )
    >>> data = model.simulate(n_samples=200, seed=42)
    >>> test = ProjectedDistanceCovariance(data=data, estimator=LinearRegression())
    >>> stat, pval = test.run_test("X", "Y", ["Z"])
    >>> round(stat,3)
    np.float64(0.43)
    >>> round(pval,3)
    np.float64(0.512)

    References
    ----------
    .. [1] Fan, J., Feng, Y., and Xia, L. "A Projection-based Conditional Dependence
           Measure with Applications to High-dimensional Undirected Graphical Models".

    """

    _tags = {
        "name": "projected_distance_covariance",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, estimator=None):
        self.data = data
        self.estimator = estimator
        super().__init__()

    def run_test(self, X: str, Y: str, Z: list):

        data = self.data

        Z_aug = list(Z) + ["intercept"]
        data_aug = data.assign(intercept=np.ones(data.shape[0]))

        Xv = data_aug[X].values
        Yv = data_aug[Y].values
        Zv = data_aug[Z_aug].values
        model_x = clone(self.estimator)
        model_y = clone(self.estimator)

        model_x.fit(Zv, Xv)
        model_y.fit(Zv, Yv)

        residual_x = (Xv - model_x.predict(Zv)).reshape(-1, 1)
        residual_y = (Yv - model_y.predict(Zv)).reshape(-1, 1)

        n = residual_x.shape[0]

        a = squareform(pdist(residual_x))
        b = squareform(pdist(residual_y))

        S1 = (a * b).sum() / (n * n)
        S2 = (a.sum() / (n * n)) * (b.sum() / (n * n))
        S3 = (a.sum(axis=1) * b.sum(axis=1)).sum() / (n**3)
        V2 = S1 + S2 - 2.0 * S3
        statistic = n * V2 / S2

        p_value = 1 - chi2.cdf(statistic, df=1)

        self.statistic_ = statistic
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
