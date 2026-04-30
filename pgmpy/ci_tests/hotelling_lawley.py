import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import CCA

from pgmpy.utils import preprocess_data

from ._base import _BaseCITest, _CITestResult, _ResidualMixin


class HotellingLawley(_ResidualMixin, _BaseCITest):
    r"""
    Hotelling-Lawley trace CI test for mixed data [1].

    This test first residualizes :math:`X` and :math:`Y` with respect to :math:`[1, Z]` The Hotelling-Lawley trace
    statistic is the sum of eigenvalue ratios of the canonical correlations between
    :math:`R_X \in \mathbb{R}^{n \times p}` and :math:`R_Y \in \mathbb{R}^{n \times q}`:

    .. math::
        \text{HLT} = \sum_{k=1}^{s} \frac{\hat{\rho}_k^2}{1 - \hat{\rho}_k^2}

    where :math:`\hat{\rho}_k` are the :math:`s = \min(p, q)` canonical correlations.
    HLT is an analog of the ANOVA F statistic for the multivariate setting.

    The p-value is computed using Pillai's F-approximation [1]:

    .. math::
        F_{\text{HLT}} = \frac{A_{\text{HLT}} / (pq)}{(1 - A_{\text{HLT}}) / \left[s(n - q - 1) + 2\right]},

    where :math:`A_{\text{HLT}} = \frac{\text{HLT}/s}{1 + \text{HLT}/s} \in [0,1]`,
    :math:`n = N - p - 1` is the error degrees of freedom, and :math:`s = \min(p, q)`.
    Numerator degrees of freedom are :math:`df_1 = pq` and denominator degrees of freedom
    are :math:`df_2 = s(n - q - 1) + 2 = s(N - p - q - 2) + 2`.

    The effect size is partial eta-squared: :math:`\eta^2 = \text{HLT} / (\text{HLT} + s)`.

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
        Hotelling-Lawley trace :math:`\text{HLT}`. Set after calling the test.
    p_value_ : float
        The p-value for the test, computed via Pillai's F-approximation. Set after calling
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
    """

    _tags = {
        "name": "hotelling_lawley",
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
        Compute the Hotelling-Lawley trace statistic and p-value.

        Returns the Hotelling-Lawley trace and p-value.

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
            The Hotelling-Lawley trace.
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

        # Step 2: Compute Hotelling-Lawley trace via CCA.
        p, q = res_x.shape[1], res_y.shape[1]
        s = min(p, q)
        cca = CCA(scale=False, n_components=s)
        res_x_c, res_y_c = cca.fit_transform(res_x, res_y)

        cancor2 = np.array([np.corrcoef(res_x_c[:, i], res_y_c[:, i])[0, 1] ** 2 for i in range(s)])
        # Clip to avoid division by zero when a canonical correlation is exactly 1.
        cancor2 = np.clip(cancor2, 0.0, 1.0 - 1e-10)

        HLT = np.sum(cancor2 / (1.0 - cancor2))

        # Step 3: Pillai's F-approximation.
        # Using a=p (predictor dims), b=q (response dims), n=N-p-1 (error df).
        # df2 = s*(n-b-1)+2 = s*(N-p-q-2)+2, which is symmetric in p and q.
        n = self.data.shape[0] - p - 1

        df1 = p * q
        df2 = s * (n - q - 1) + 2

        A_HLT = (HLT / s) / (1.0 + HLT / s)  # in [0, 1]
        F_stat = (A_HLT / df1) / ((1.0 - A_HLT) / df2)
        p_value = 1.0 - stats.f.cdf(F_stat, df1, df2)

        return _CITestResult(statistic=HLT, p_value=p_value, effect_size=HLT / (HLT + s))
