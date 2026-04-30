import numpy as np
import pandas as pd
from scipy import special, stats

from ._base import _BaseCITest, _CITestResult

_LAMBDA_ALIASES = {
    "pearson": 1.0,
    "log-likelihood": 0.0,
    "freeman-tuckey": -0.5,
    "mod-log-likelihood": -1.0,
    "neyman": -2.0,
    "cressie-read": 2.0 / 3.0,
}


class PowerDivergence(_BaseCITest):
    r"""
    Cressie-Read power divergence test for conditional independence on discrete data [1].

    This test evaluates the null hypothesis :math:`X \perp Y \mid Z` using contingency tables. For a contingency table
    with observed counts :math:`O_{ij}` and expected counts :math:`E_{ij}` under independence, the Cressie-Read power
    divergence statistic is:

    .. math::
        T_\lambda = \frac{2}{\lambda(\lambda + 1)}
        \sum_{i, j} O_{ij} \left[\left(\frac{O_{ij}}{E_{ij}}\right)^\lambda - 1\right],

    for :math:`\lambda \notin \{-1, 0\}`. Different values of :math:`\lambda` recover common special cases such as the
    Pearson chi-square test and the log-likelihood ratio test.

    If :math:`Z = \emptyset`, all observations form a single stratum. Otherwise the data are partitioned by each
    observed configuration :math:`z` of :math:`Z`. For every stratum the observed counts :math:`O_{ij}^{(z)}` and
    expected counts under independence :math:`E_{ij}^{(z)} = R_i^{(z)} C_j^{(z)} / n^{(z)}` are computed, where
    :math:`R_i^{(z)}` and :math:`C_j^{(z)}` are the row and column marginals of the stratum and :math:`n^{(z)}` is
    the stratum size. The overall statistic and degrees of freedom are aggregated over strata:

    .. math::
        T = \sum_{z} T_\lambda^{(z)},
        \qquad
        \nu = \sum_{z} \nu^{(z)}.

    Under the null hypothesis, :math:`T` is treated with the usual chi-square asymptotic approximation, so the
    p-value is computed as:

    .. math::
        p = 1 - F_{\chi^2_\nu}(T),

    where :math:`F_{\chi^2_\nu}` is the CDF of the chi-square distribution with :math:`\nu` degrees of freedom.

    Two corrections are applied during this aggregation:

    **1. Adjusted (sparse) degrees of freedom.** A row or column that never occurs in a stratum contributes zero to both
    :math:`\nu^{(z)}` and :math:`T_\lambda^{(z)}` (its expected counts are zero, so its per-cell terms vanish). A
    stratum that collapses to a single active row or column has :math:`\nu^{(z)} = 0` and is effectively skipped.

    **2. Yates' continuity correction on 2x2 strata.** Whenever a stratum's active contingency table is 2x2
    (equivalently, :math:`\nu^{(z)} = 1`), Yates' continuity correction is applied to the observed counts before
    the per-cell power-divergence terms are evaluated:

    The effect size is Cramér's V:

    .. math::
        V = \sqrt{\frac{T}{n \cdot (k - 1)}},

    where :math:`k = \min(|X|, |Y|)` is the smaller number of categories and :math:`n` is the sample size.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset on which to test the independence condition.

    lambda_ : float or string
        The :math:`\lambda` parameter for the power divergence statistic. Some values of
        ``lambda_`` recover well-known special cases:

            * "pearson"             1          "Chi-squared test"
            * "log-likelihood"      0          "G-test or log-likelihood"
            * "freeman-tuckey"     -1/2        "Freeman-Tuckey Statistic"
            * "mod-log-likelihood"  -1         "Modified Log-likelihood"
            * "neyman"              -2         "Neyman's statistic"
            * "cressie-read"        2/3        "The value recommended in the paper[1]"

    Attributes
    ----------
    statistic_ : float
        The power divergence test statistic :math:`T`. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.
    dof_ : int
        Degrees of freedom :math:`\nu` for the test. Set after calling the test.
    effect_size_ : float
        Cramér's V. Set after calling the test.

    References
    ----------
    .. [1] Cressie, Noel, and Timothy RC Read. "Multinomial goodness‐of‐fit tests." Journal of the Royal Statistical
         Society: Series B (Methodological) 46.3 (1984): 440-464.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     data=np.random.randint(low=0, high=2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> test = PowerDivergence(data=data)
    >>> test(X="A", Y="C", Z=[], significance_level=0.05)
    np.True_
    >>> round(test.statistic_, 2)
    np.float64(0.03)
    >>> round(test.p_value_, 2)
    np.float64(0.86)
    >>> test.dof_
    1
    >>> test(X="A", Y="B", Z=["D"], significance_level=0.05)
    np.True_
    >>> test(X="A", Y="B", Z=["D", "E"], significance_level=0.05)
    np.False_
    """

    _tags = {
        "name": "power_divergence",
        "data_types": ("discrete",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, lambda_: str | float = "cressie-read", use_cache: bool = True):
        self.data = data
        self.lambda_ = lambda_
        self._codes = {}
        self._cardinalities = {}
        for col in data.columns:
            codes, uniques = pd.factorize(data[col], sort=False, use_na_sentinel=True)
            self._codes[col] = np.ascontiguousarray(codes, dtype=np.int64)
            self._cardinalities[col] = len(uniques)
        super().__init__(use_cache=use_cache)

    def _compute_result(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute the power-divergence statistic, p-value, and degrees of freedom.

        Each row is encoded as a flat index into an ``(n_strata, kx, ky)``
        contingency built in a single ``np.bincount``. Marginals, expected
        counts, dof, Yates' correction (on 2x2 strata), and the per-cell
        statistic are all computed via numpy broadcasting over that 3D table,
        then summed to a single chi and one ``chi2.sf`` call.
        """
        x_codes = self._codes[X]
        y_codes = self._codes[Y]
        kx = self._cardinalities[X]
        ky = self._cardinalities[Y]

        # Step 1: Build flat index into an (n_strata, kx, ky) contingency table.
        # For |Z|=0 all rows go into a single stratum. Otherwise use the full
        # product space of Z cardinalities directly (avoids costly np.unique).
        if len(Z) == 0:
            z_idx = np.zeros(len(x_codes), dtype=np.int64)
            n_strata = 1
        else:
            z_idx = np.zeros(len(x_codes), dtype=np.int64)
            n_strata = 1
            for col in Z:
                z_idx = z_idx * self._cardinalities[col] + self._codes[col]
                n_strata *= self._cardinalities[col]

        # Step 2: Build the (n_strata, kx, ky) contingency table in one bincount.
        flat = (z_idx * kx + x_codes) * ky + y_codes
        observed = np.bincount(flat, minlength=n_strata * kx * ky).reshape(n_strata, kx, ky).astype(np.float64)

        # Step 3: Per-stratum marginals and expected counts.
        row_sums = observed.sum(axis=2, keepdims=True)
        col_sums = observed.sum(axis=1, keepdims=True)
        n_per = observed.sum(axis=(1, 2), keepdims=True)
        with np.errstate(invalid="ignore"):
            expected = row_sums * col_sums / n_per
        safe = expected > 0

        # Step 4: Per-stratum dof = (active_rows - 1) * (active_cols - 1).
        rows_active = (row_sums.squeeze(axis=2) > 0).sum(axis=1)
        cols_active = (col_sums.squeeze(axis=1) > 0).sum(axis=1)
        dof_per = (rows_active - 1).clip(min=0) * (cols_active - 1).clip(min=0)
        dof = int(dof_per.sum())

        # Step 5: Yates' continuity correction on 2x2 strata, matching scipy.stats.chi2_contingency.
        correction_mask = (dof_per == 1)[:, None, None]
        if correction_mask.any():
            diff = expected - observed
            adjustment = np.minimum(0.5, np.abs(diff)) * np.sign(diff)
            observed = np.where(correction_mask, observed + adjustment, observed)

        # Step 6: Power-divergence statistic and p-value. dof=0 (every stratum
        # degenerate) yields p_value=NaN, treated as "not independent" downstream.
        terms = self._power_divergence_terms(observed, expected, safe)
        chi = terms.sum()
        p_value = stats.chi2.sf(chi, df=dof)

        n = len(self.data)
        k = min(self._cardinalities[X], self._cardinalities[Y])
        effect_size = np.sqrt(chi / (n * max(k - 1, 1)))

        return _CITestResult(statistic=chi, p_value=p_value, effect_size=effect_size, attributes={"dof_": dof})

    def _power_divergence_terms(self, observed, expected, safe):
        """Per-cell power-divergence contribution for the (n_strata, kx, ky) table."""
        lam = self.lambda_
        if isinstance(lam, str):
            lam = _LAMBDA_ALIASES[lam]

        # Pearson: (O - E)^2 / E.
        if lam == 1.0:
            terms = np.zeros_like(expected)
            np.divide((observed - expected) ** 2, expected, out=terms, where=safe)
            return terms
        # Log-likelihood: 2 * O * log(O / E).
        if lam == 0.0:
            return 2.0 * (special.xlogy(observed, observed) - special.xlogy(observed, expected))
        # Modified log-likelihood: 2 * E * log(E / O).
        if lam == -1.0:
            return 2.0 * (special.xlogy(expected, expected) - special.xlogy(expected, observed))

        # General Cressie-Read on E > 0 cells; zero elsewhere.
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio = observed / expected
            full = observed * (np.power(ratio, lam) - 1.0) / (0.5 * lam * (lam + 1.0))
        return np.where(safe, full, 0.0)
