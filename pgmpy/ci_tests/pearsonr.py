import numpy as np
import pandas as pd
from scipy import special

from pgmpy.utils import regression_residual_covariance

from ._base import BaseCITest, _CITestResult


class Pearsonr(BaseCITest):
    r"""
    Partial Correlation test for conditional independence.

    If :math:`Z = \emptyset`, compute Pearson's correlation coefficient :math:`r_{XY}` and its two-sided p-value.

    If :math:`Z \neq \emptyset`, regress :math:`X` and :math:`Y` on :math:`[1, Z]` using least squares, compute the
    residuals :math:`r_X` and :math:`r_Y`, and define the partial correlation as the Pearson correlation between those
    residuals. The resulting test statistic is

    .. math::
        t = \rho_{XY \mid Z} \sqrt{\frac{n - |Z| - 2}{1 - \rho_{XY \mid Z}^2}},

    where :math:`n` is the sample size and :math:`|Z|` is the number of conditioning variables. Under the null
    hypothesis :math:`X \perp Y \mid Z`, this statistic is Student's t distribution with :math:`n - |Z| - 2` degrees of
    freedom.

    The effect size is the absolute partial correlation :math:`|\rho_{XY \mid Z}|`.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition. Its numeric columns must not contain missing
        values.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import Pearsonr
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(data=rng.standard_normal(size=(1000, 3)), columns=["X", "Y", "Z"])
    >>> test = Pearsonr(data=data)
    >>> test(X="X", Y="Y", Z=["Z"], significance_level=0.05)
    True
    >>> round(test.statistic_, 2)
    np.float64(0.01)
    >>> round(test.p_value_, 2)
    np.float64(0.87)
    >>> test.dof_
    997

    Attributes
    ----------
    statistic_ : float
        Pearson's correlation coefficient (or partial correlation when Z is non-empty),
        ranging from -1 to 1. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.
    effect_size_ : float
        Absolute partial correlation. Set after calling the test.

    References
    ----------
    - :footcite:t:`peerj_blue_driver`
    - :footcite:t:`wikipedia_partial_correlation`
    """

    _tags = {
        "name": "pearsonr",
        "data_types": ["continuous"],
        "default_for": "continuous",
        "requires_data": True,
        "capability:conditioning": True,
        "assumption:linearity": True,
        "assumption:additive_noise": True,
        "assumption:gaussian_noise": True,
    }

    def __init__(self, data: pd.DataFrame, use_cache: bool = True):
        self.data = data
        self._cov = np.atleast_2d(np.cov(data.to_numpy(dtype=float), rowvar=False, ddof=0))
        self._col_index = {col: index for index, col in enumerate(data.columns)}
        super().__init__(use_cache=use_cache)

    def _partial_correlation(self, X: str, Y: str, Z: list) -> tuple[float, int]:
        """Partial correlation of `X` and `Y` given `Z`, and the degrees of freedom of the test."""
        residual_cov = regression_residual_covariance(
            self._cov, [self._col_index[X], self._col_index[Y]], [self._col_index[var] for var in Z]
        )
        dof = self.data.shape[0] - len(Z) - 2
        return residual_cov[0, 1] / np.sqrt(residual_cov[0, 0] * residual_cov[1, 1]), dof

    def _compute_result(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute Pearson correlation coefficient and p-value.

        Returns Pearson's r, p-value, and optional degrees of freedom metadata. Without residual degrees of freedom
        (``n_samples <= len(Z) + 2``) the data carry no evidence against independence and the p-value is 1, as in
        :func:`scipy.stats.pearsonr` for two samples.
        """
        coef, dof = self._partial_correlation(X, Y, Z)
        attributes = {"dof_": dof} if len(Z) > 0 else {}
        if dof <= 0:
            return _CITestResult(statistic=coef, p_value=1.0, effect_size=abs(coef), attributes=attributes)

        t_statistic = coef * np.sqrt(dof / (1 - coef**2))
        p_value = 2 * special.stdtr(dof, -abs(t_statistic))

        return _CITestResult(statistic=coef, p_value=p_value, effect_size=abs(coef), attributes=attributes)
