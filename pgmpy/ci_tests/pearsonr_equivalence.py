import numpy as np
import pandas as pd
from scipy import stats

from ._base import _CITestResult
from .pearsonr import Pearsonr


class PearsonrEquivalence(Pearsonr):
    r"""
    Pearson equivalence test [1] for conditional independence on continuous data.

    This test first computes the partial correlation coefficient :math:`\hat{\rho}_{XY \mid Z}` using :class:`Pearsonr`.
    Let :math:`\delta` denote ``delta_threshold``. The Fisher transform is computed as:

    .. math::
        z_\rho = \operatorname{arctanh}(\hat{\rho}_{XY \mid Z}),
        \qquad
        z_\delta = \operatorname{arctanh}(\delta),

    and defines

    .. math::
        c = \sqrt{n - |Z| - 3},

    where :math:`n` is the sample size and :math:`|Z|` is the number of conditioning variables.

    The test then performs a TOST (two one-sided tests) procedure for the equivalence hypothesis

    .. math::
        H_0: \rho_{XY \mid Z} \leq -\delta \;\; \text{or} \;\; \rho_{XY \mid Z} \geq \delta
        \qquad \text{vs.} \qquad
        H_1: -\delta < \rho_{XY \mid Z} < \delta.

    The two one-sided test statistics are:

    .. math::
        T_{\mathrm{lower}} = c (z_\rho + z_\delta),
        \qquad
        T_{\mathrm{upper}} = c (z_\rho - z_\delta),

    with corresponding p-values:

    .. math::
        p_{\mathrm{lower}} = 1 - \Phi(T_{\mathrm{lower}}),
        \qquad
        p_{\mathrm{upper}} = \Phi(T_{\mathrm{upper}}),

    where :math:`\Phi` is the standard normal CDF. The reported p-value is:

    .. math::
        p = \max(p_{\mathrm{lower}}, p_{\mathrm{upper}}).

    The effect size is the absolute partial correlation :math:`|\hat{\rho}_{XY \mid Z}|`.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset in which to test the independence condition.

    delta_threshold : float
        The equivalence bound (threshold for practical independence).

    Attributes
    ----------
    statistic_ : float
        Fisher z-transformed correlation coefficient :math:`z_\rho`. Set after calling the test.
    p_value_ : float
        The p-value from the TOST procedure. Independence is concluded when
        ``p_value_ < significance_level`` (opposite of standard CI tests). Set after calling the test.
    effect_size_ : float
        Absolute partial correlation. Set after calling the test.

    References
    ----------
    .. [1] Malinsky, Daniel. "A cautious approach to constraint-based causal model selection." arXiv preprint
            arXiv:2404.18232 (2024).
    """

    _tags = {
        "name": "pearsonr_equivalence",
        "data_types": ("continuous",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, delta_threshold: float = 0.1, use_cache: bool = True):
        self.delta_threshold = delta_threshold
        super().__init__(data=data, use_cache=use_cache)

    def is_independent(
        self,
        X: str,
        Y: str,
        Z: list | tuple = (),
        significance_level: float = 0.05,
    ) -> bool:
        """
        Perform the equivalence CI test.

        Note: Independence is concluded when p_value_ < significance_level
        (rejecting the null of dependence), which is the OPPOSITE of standard CI tests.

        Returns
        -------
        bool
            True if X ⊥⊥ Y | Z (p_value_ < significance_level), else False.
        """
        self.run_test(X=X, Y=Y, Z=list(Z))

        return self.p_value_ < significance_level

    def _compute_result(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute Pearson equivalence statistic and p-value.

        Returns the Fisher z-transformed partial correlation statistic and p-value.
        """
        # Step 2: Compute Partial Pearson Correlation via parent and clip to avoid infinities
        pearsonr_result = super()._compute_result(X, Y, Z)
        rho = np.clip(pearsonr_result.statistic, -0.999999, 0.999999)

        # Step 3: Fisher Z-Transformation
        coeff = np.arctanh(rho)
        z_delta = np.arctanh(self.delta_threshold)

        n = self.data.shape[0]
        s = len(Z)  # Number of conditioning variables

        std_error_factor = np.sqrt(n - s - 3)

        # Step 4: TOST (Two One-Sided Tests)
        # Step 4.1: H0: rho <= -delta  vs  H1: rho > -delta
        z_score_lower = std_error_factor * (coeff + z_delta)
        p_value_lower = 1 - stats.norm.cdf(z_score_lower)

        # Step 4.2: H0: rho >= delta   vs  H1: rho < delta
        z_score_upper = std_error_factor * (coeff - z_delta)
        p_value_upper = stats.norm.cdf(z_score_upper)

        return _CITestResult(
            statistic=coeff,
            p_value=max(p_value_lower, p_value_upper),
            effect_size=abs(pearsonr_result.statistic),
            attributes=dict(pearsonr_result.attributes),
        )
