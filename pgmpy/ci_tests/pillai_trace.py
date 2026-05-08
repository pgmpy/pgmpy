import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA

from pgmpy.utils import preprocess_data

from ._base import _BaseCITest, _CITestResult, _ResidualMixin


class PillaiTrace(_ResidualMixin, _BaseCITest):
    r"""
    Pillai's trace test for conditional independence with mixed data [1].

    This test first residualizes :math:`X` and :math:`Y` with respect to :math:`[1, Z]` using an estimator (a
    RandomForest estimator by default, but any sklearn-compatible estimator can be provided). For a continuous target
    :math:`T`, the residual is

    .. math::
        r_T = T - \hat{T}(Z).

    For a categorical target :math:`T` with :math:`K` categories, let :math:`D_T \in \{0, 1\}^{n \times K}` be the
    dummy-encoded matrix of :math:`T`, and let :math:`\hat{D}_T(Z)` denote the predicted class probabilities from the
    classifier. The residual matrix [2] is defined as (last column dropped to avoid colinearity):

    .. math::
        R_T = \operatorname{drop\_last}\left(D_T - \hat{D}_T(Z)\right),

    Let :math:`R_X \in \mathbb{R}^{n \times p}` and :math:`R_Y \in \mathbb{R}^{n \times q}` be the residuals of
    :math:`X` and :math:`Y`, and :math:`\rho = \rho_1, \ldots, \rho_s` be the canonical correlations between them. The
    Pillai's trace statistic is:

    .. math::
        V = \sum_{i=1}^{s} \rho_i^2.

    The p-value is computed using :math:`F`-approximation as (:math:`p` and :math:`q` are the number of columns in
    residual matrices):

    .. math::
        F = \frac{V / (pq)}{(s - V) / \left[s (n - 1 + s - p - q)\right]}
          = \frac{V}{pq} \cdot \frac{s (n - 1 + s - p - q)}{s - V},

    with numerator degrees of freedom :math:`df_1 = pq` and denominator degrees of freedom
    :math:`df_2 = s (n - 1 + s - p - q)`, where :math:`n` is the sample size.

    The effect size is partial eta-squared: :math:`\eta^2 = V / s`.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    estimator : estimator instance, optional
        Any sklearn-compatible estimator with ``fit``, ``predict``, and ``predict_proba``
        (if testing discrete variables) methods. If ``None`` (default), uses
        ``RandomForestClassifier`` for categorical targets and ``RandomForestRegressor``
        for continuous targets. Conditioning variables are one-hot encoded before fitting.

    Attributes
    ----------
    statistic_ : float
        Pillai's trace statistic :math:`V`. Set after calling the test.
    p_value_ : float
        The p-value for the test, computed via F-approximation. Set after calling the test.
    effect_size_ : float
        Partial eta-squared. Set after calling the test.
    estimator_x_ : sklearn-compatible estimator
        The fitted estimator used for predicting X.
    estimator_y_ : sklearn-compatible estimator
        The fitted estimator used for predicting Y.

    References
    ----------
    .. [1] Ankan, Ankur, and Johannes Textor. "A simple unified approach to testing high-dimensional conditional
           independences for categorical and ordinal data." Proceedings of the AAAI Conference on Artificial
           Intelligence.
    .. [2] Li, C.; and Shepherd, B. E. 2010. Test of Association Between Two Ordinal Variables While Adjusting for
           Covariates. Journal of the American Statistical Association.
    .. [3] Muller, K. E. and Peterson B. L. (1984) Practical Methods for computing power in testing the multivariate
           general linear hypothesis. Computational Statistics & Data Analysis.
    """

    _tags = {
        "name": "pillai",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": "mixed",
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, estimator=None, use_cache: bool = True):
        self.data, self.dtypes = preprocess_data(data)
        self.estimator = estimator
        super().__init__(use_cache=use_cache)

    def _compute_result(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute Pillai's trace statistic and p-value.

        Returns Pillai's trace statistic and p-value.

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
            The Pillai's trace statistic.
        p_value : float
            The p-value.
        """
        # Steps 1: Compute residuals of X and Y given Z.
        res_x, self.estimator_x_ = self.get_residuals(X, Z)
        res_y, self.estimator_y_ = self.get_residuals(Y, Z)

        if isinstance(res_x, pd.Series):
            res_x = res_x.to_frame()
        if isinstance(res_y, pd.Series):
            res_y = res_y.to_frame()

        # Step 2: Compute Pillai's trace test statistic via CCA.
        n_components = min(res_x.shape[1], res_y.shape[1])
        cca = CCA(scale=False, n_components=n_components)
        res_x_c, res_y_c = cca.fit_transform(res_x, res_y)

        cancor = []
        for i in range(n_components):
            cancor.append(np.corrcoef(res_x_c[:, [i]].T, res_y_c[:, [i]].T)[0, 1])

        coef = (np.array(cancor) ** 2).sum()

        # Step 3: Compute p-value using F-approximation.
        s = min(res_x.shape[1], res_y.shape[1])
        df1 = res_x.shape[1] * res_y.shape[1]
        df2 = s * (self.data.shape[0] - 1 + s - res_x.shape[1] - res_y.shape[1])
        f_stat = (coef / df1) * (df2 / (s - coef))
        p_value = 1 - stats.f.cdf(f_stat, df1, df2)

        return _CITestResult(statistic=coef, p_value=p_value, effect_size=coef / s)
