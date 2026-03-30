import numpy as np
import pandas as pd
from scipy import stats

from ._base import _BaseCITest


class Pearsonr(_BaseCITest):
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

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import Pearsonr
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(data=rng.standard_normal(size=(1000, 3)), columns=["X", "Y", "Z"])
    >>> test = Pearsonr(data=data)
    >>> test(X="X", Y="Y", Z=["Z"], significance_level=0.05)
    np.True_
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

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Pearson_correlation_coefficient
    .. [2] https://en.wikipedia.org/wiki/Partial_correlation#Using_linear_regression
    """

    _tags = {
        "name": "pearsonr",
        "data_types": ("continuous",),
        "default_for": "continuous",
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
        Compute Pearson correlation coefficient and p-value.

        Sets ``self.statistic_`` (Pearson's r) and ``self.p_value_``.
        """
        data = self.data
        n_samples = data.shape[0]

        # Step 1: If Z is empty compute a non-conditional test.
        if len(Z) == 0:
            coef, p_value = stats.pearsonr(data.loc[:, X], data.loc[:, Y])

        # Step 2: If Z is non-empty, use linear regression to compute residuals and test independence on it.
        else:
            design_matrix = np.column_stack([np.ones(n_samples), data.loc[:, Z].to_numpy()])
            X_coef = np.linalg.lstsq(design_matrix, data.loc[:, X], rcond=None)[0]
            Y_coef = np.linalg.lstsq(design_matrix, data.loc[:, Y], rcond=None)[0]

            residual_X = data.loc[:, X] - design_matrix @ X_coef
            residual_Y = data.loc[:, Y] - design_matrix @ Y_coef

            coef = np.corrcoef(residual_X, residual_Y)[0, 1]
            self.dof_ = n_samples - len(Z) - 2
            t_statistic = coef * np.sqrt(self.dof_ / (1 - coef**2))
            p_value = 2 * stats.t.sf(np.abs(t_statistic), df=self.dof_)

        self.statistic_ = coef
        self.p_value_ = p_value

        return self.statistic_, self.p_value_
