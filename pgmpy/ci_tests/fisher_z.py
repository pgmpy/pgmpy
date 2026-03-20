import numpy as np
import pandas as pd
from scipy import stats

from ._base import _BaseCITest
from .pearsonr import Pearsonr


class FisherZ(_BaseCITest):
    r"""
    Fisher's Z test for conditional independence on continuous data.

    This test first computes the Pearson or partial correlation coefficient :math:`\rho_{XY \mid Z}` using
    :class:`Pearsonr`. It then applies the Fisher transformation and computes the test statistic as:

    .. math::
        Z = \sqrt{n - |Z| - 3} \cdot \operatorname{arctanh}(\rho_{XY \mid Z}),

    where :math:`n` is the sample size and :math:`|Z|` is the number of conditioning variables. Under the null
    hypothesis :math:`X \perp Y \mid Z`, :math:`Z` is approximately standard normal.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    Attributes
    ----------
    statistic_ : float
        The Fisher Z test statistic. Set after calling the test.
    p_value_ : float
        The two-sided p-value for the test. Set after calling the test.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.ci_tests import FisherZ
    >>> rng = np.random.default_rng(seed=42)
    >>> data = pd.DataFrame(rng.standard_normal((1000, 3)), columns=["X", "Y", "Z"])
    >>> test = FisherZ(data=data)
    >>> test("X", "Y", ["Z"], significance_level=0.05)
    np.True_
    >>> round(test.statistic_, 2)
    np.float64(0.17)
    >>> isinstance(test.p_value_, float)
    np.float64(0.87)
    """

    _tags = {
        "name": "fisher_z",
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
        Compute the Fisher Z statistic and p-value.

        Sets ``self.statistic_``, ``self.transformed_statistic_``, and ``self.p_value_``.
        """
        partial_corr, _ = Pearsonr(data=self.data).run_test(X=X, Y=Y, Z=Z)

        rho = np.clip(partial_corr, -0.999999, 0.999999)
        self.statistic_ = np.sqrt(self.data.shape[0] - len(Z) - 3) * np.arctanh(rho)
        self.p_value_ = 2 * stats.norm.sf(np.abs(self.statistic_))

        return self.statistic_, self.p_value_
