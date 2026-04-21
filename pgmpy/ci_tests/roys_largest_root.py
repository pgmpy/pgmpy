import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA

from pgmpy.utils import preprocess_data

from ._base import _BaseCITest, _ResidualMixin


class RoysLargestRoot(_ResidualMixin, _BaseCITest):
    r"""
    Roy's largest root CI test for mixed data [1].

    This test first residualizes :math:`X` and :math:`Y` with respect to :math:`[1, Z]`
    (see :class:`~pgmpy.ci_tests.PillaiTrace` for residualization details). Roy's largest
    root is the largest squared canonical correlation between :math:`R_X \in
    \mathbb{R}^{n \times p}` and :math:`R_Y \in \mathbb{R}^{n \times q}`:

    .. math::
        \text{RLR} = \hat{\rho}_1^2 = \max_k \hat{\rho}_k^2.

    The p-value is computed using the following F upper bound [1]:

    .. math::
        F_{\max} = \frac{\text{RLR} / v}{(1 - \text{RLR}) / n},

    with numerator degrees of freedom :math:`df_1 = v` and denominator degrees of freedom
    :math:`df_2 = n`, where :math:`v = \max(p, q)` and :math:`n = N - v - 1`.

    .. warning::
        This F-approximation yields an *upper bound* on the true significance level
        (i.e., the resulting p-value is optimistically small). When :math:`s = \min(p,q) = 1`
        or when the test is univariate (:math:`p = q = 1`), the approximation is exact [1].
        For :math:`s > 1`, it is advisable to corroborate a significant result with one of
        the other multivariate tests.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    estimator : estimator instance, optional
        Any sklearn-compatible estimator with ``fit``, ``predict``, and ``predict_proba``
        (if testing discrete variables) methods. If ``None`` (default), uses
        ``RandomForestClassifier`` for categorical targets and ``RandomForestRegressor``
        for continuous targets.

    Attributes
    ----------
    statistic_ : float
        Roy's largest root :math:`\hat{\rho}_1^2`. Set after calling the test.
    p_value_ : float
        Upper-bound p-value for the test, computed via the F approximation. Set after
        calling the test.

    References
    ----------
    .. [1] Muller, K. E. and Peterson B. L. (1984) Practical Methods for computing power in
           testing the multivariate general linear hypothesis. Computational Statistics &
           Data Analysis.
    """

    _tags = {
        "name": "roys_largest_root",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, estimator=None):
        self.data, self.dtypes = preprocess_data(data)
        self.estimator = estimator
        super().__init__()

    def run_test(self, X: str, Y: str, Z: list):
        """
        Compute Roy's largest root statistic and its upper-bound p-value.

        Sets ``self.statistic_`` (:math:`\\hat{\\rho}_1^2`) and ``self.p_value_``.

        Parameters
        ----------
        X : str
            The first variable for testing X _|_ Y | Z.
        Y : str
            The second variable for testing X _|_ Y | Z.
        Z : list
            Conditioning variables.

        Returns
        -------
        statistic : float
            Roy's largest root (largest squared canonical correlation).
        p_value : float
            Upper-bound p-value via F-approximation.
        """
        # Step 1: Compute residuals of X and Y given Z.
        res_x = self.get_residuals(X, Z)
        res_y = self.get_residuals(Y, Z)

        if isinstance(res_x, pd.Series):
            res_x = res_x.to_frame()
        if isinstance(res_y, pd.Series):
            res_y = res_y.to_frame()

        # Step 2: Compute the largest squared canonical correlation via CCA.
        p, q = res_x.shape[1], res_y.shape[1]
        s = min(p, q)
        cca = CCA(scale=False, n_components=s)
        res_x_c, res_y_c = cca.fit_transform(res_x, res_y)

        cancor2 = np.array([np.corrcoef(res_x_c[:, i], res_y_c[:, i])[0, 1] ** 2 for i in range(s)])

        RLR = float(np.max(cancor2))

        # Step 3: F upper bound (eq. 28 in [1]).
        # v = max(p, q) and n = N - v - 1 give a formula that is symmetric in p and q
        # and reduces to the exact F-test for the univariate case (p = q = 1).
        v = max(p, q)
        n = self.data.shape[0] - v - 1

        df1 = float(v)
        df2 = float(n)

        # Clip to avoid division by zero if RLR is numerically 1.
        RLR_clipped = min(RLR, 1.0 - 1e-10)
        F_stat = (RLR_clipped / v) / ((1.0 - RLR_clipped) / n)
        p_value = float(1.0 - stats.f.cdf(F_stat, df1, df2))

        self.statistic_ = RLR
        self.p_value_ = p_value
        return self.statistic_, self.p_value_
