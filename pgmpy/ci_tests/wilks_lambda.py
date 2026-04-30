import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA

from pgmpy.utils import preprocess_data

from ._base import _BaseCITest, _CITestResult, _ResidualMixin


class WilksLambda(_ResidualMixin, _BaseCITest):
    r"""
    Wilks' Lambda CI test for mixed data [1].

    This test first residualizes :math:`X` and :math:`Y` with respect to :math:`[1, Z]`
    (see :class:`~pgmpy.ci_tests.PillaiTrace` for residualization details). Wilks' Lambda
    is defined as the product of the residual fractions of the squared canonical correlations:

    .. math::
        W = \prod_{k=1}^{s} (1 - \hat{\rho}_k^2),

    where :math:`\hat{\rho}_k` are the canonical correlations between :math:`R_X \in
    \mathbb{R}^{n \times p}` and :math:`R_Y \in \mathbb{R}^{n \times q}`, and
    :math:`s = \min(p, q)`. :math:`W` lies in :math:`(0, 1]`, with values near 1
    indicating independence and values near 0 indicating dependence.

    The p-value is computed using Rao's F-approximation [1, 2]:

    .. math::
        F_W = \frac{(1 - W^{1/g}) / (pq)}{
        W^{1/g} / \left[g\!\left(n - \tfrac{q-p+1}{2}\right) - \tfrac{pq-2}{2}\right]},

    with numerator degrees of freedom :math:`df_1 = pq` and denominator degrees of
    freedom :math:`df_2 = g\!\left(n - \tfrac{q-p+1}{2}\right) - \tfrac{pq-2}{2}`, where
    :math:`n = N - p - 1` is the error degrees of freedom (treating :math:`R_X` as the
    predictor matrix in the general linear hypothesis framework), and

    .. math::
        g = \begin{cases}
            1 & \text{if } p^2 + q^2 \leq 5, \\
            \left[\dfrac{p^2 q^2 - 4}{p^2 + q^2 - 5}\right]^{1/2} & \text{otherwise.}
        \end{cases}

    When :math:`s = 1` or :math:`s = 2` the approximation is exact [1].

    The effect size is partial eta-squared: :math:`\eta^2 = 1 - W^{1/s}`.

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
        Wilks' Lambda :math:`W`. Set after calling the test.
    p_value_ : float
        The p-value for the test, computed via Rao's F-approximation. Set after calling
        the test.
    effect_size_ : float
        Partial eta-squared. Set after calling the test.
    estimator_x_ : sklearn-compatible estimator
        The fitted estimator used for predicting X.
    estimator_y_ : sklearn-compatible estimator
        The fitted estimator used for predicting Y.

    References
    ----------
    .. [1] Muller, K. E. and Peterson B. L. (1984) Practical Methods for computing power in
           testing the multivariate general linear hypothesis. Computational Statistics &
           Data Analysis.
    .. [2] Rao, C. R. (1973). Linear Statistical Inference and Its Applications (2nd ed.).
           John Wiley and Sons, New York.
    """

    _tags = {
        "name": "wilks_lambda",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, estimator=None, use_cache: bool = True):
        self.data, self.dtypes = preprocess_data(data)
        self.estimator = estimator
        super().__init__(use_cache=use_cache)

    def _compute_result(self, X: str, Y: str, Z: list):
        """
        Compute Wilks' Lambda statistic and p-value.

        Returns Wilks' Lambda and its p-value.

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
            Wilks' Lambda.
        p_value : float
            The p-value.
        """
        # Step 1: Compute residuals of X and Y given Z.
        res_x, self.estimator_x_ = self.get_residuals(X, Z)
        res_y, self.estimator_y_ = self.get_residuals(Y, Z)

        if isinstance(res_x, pd.Series):
            res_x = res_x.to_frame()
        if isinstance(res_y, pd.Series):
            res_y = res_y.to_frame()

        # Step 2: Compute Wilks' Lambda via CCA.
        p, q = res_x.shape[1], res_y.shape[1]
        s = min(p, q)
        cca = CCA(scale=False, n_components=s)
        res_x_c, res_y_c = cca.fit_transform(res_x, res_y)

        cancor2 = np.array([np.corrcoef(res_x_c[:, i], res_y_c[:, i])[0, 1] ** 2 for i in range(s)])

        W = np.prod(1.0 - cancor2)

        # Step 3: Rao's F-approximation.
        # Using a=p (predictor dims), b=q (response dims), n=N-p-1 (error df).
        # The resulting df2 is symmetric in p and q.
        a, b = p, q
        n = self.data.shape[0] - p - 1

        ab = a * b
        if a**2 + b**2 <= 5:
            g = 1.0
        else:
            g = np.sqrt((a**2 * b**2 - 4.0) / (a**2 + b**2 - 5.0))

        df1 = ab
        df2 = g * (n - (b - a + 1) / 2.0) - (ab - 2) / 2.0

        W_1g = W ** (1.0 / g)
        F_stat = ((1.0 - W_1g) * df2) / (W_1g * df1)
        p_value = 1.0 - stats.f.cdf(F_stat, df1, df2)

        return _CITestResult(statistic=W, p_value=p_value, effect_size=1.0 - W ** (1.0 / s))
