import numpy as np
import pandas as pd
from scipy import stats

from pgmpy import logger

from ._base import _BaseCITest


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

    If :math:`Z = \emptyset`, the implementation constructs the contingency table of :math:`X` and :math:`Y` from the
    full dataset and computes :math:`T_\lambda` with :func:`scipy.stats.chi2_contingency`.

    If :math:`Z \neq \emptyset`, the data are partitioned by each observed configuration :math:`z` of :math:`Z`. For
    each stratum, a contingency table for :math:`X` and :math:`Y` is constructed and its power divergence statistic
    :math:`T_\lambda^{(z)}` and degrees of freedom :math:`\nu^{(z)}` are computed. The overall statistic used in the
    code is:

    .. math::
        T = \sum_{z} T_\lambda^{(z)},
        \qquad
        \nu = \sum_{z} \nu^{(z)},

    where the sum runs over strata whose contingency tables do not contain an all-zero row or all-zero column. Strata
    with such degenerate tables are skipped.

    Under the null hypothesis, :math:`T` is treated with the usual chi-square asymptotic approximation, so the
    p-value is computed as:

    .. math::
        p = 1 - F_{\chi^2_\nu}(T),

    where :math:`F_{\chi^2_\nu}` is the CDF of the chi-square distribution with :math:`\nu` degrees of freedom.

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
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> test = PowerDivergence(data)
    >>> test("A", "C", [], significance_level=0.05)
    True
    >>> test("A", "B", ["D"], significance_level=0.05)
    True
    >>> test("A", "B", ["D", "E"], significance_level=0.05)
    False
    """

    _tags = {
        "name": "power_divergence",
        "data_types": ("discrete",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, lambda_: str | float = "cressie-read"):
        self.data = data
        self.lambda_ = lambda_
        super().__init__()

    def run_test(
        self,
        X: str,
        Y: str,
        Z: list,
    ):
        """
        Compute power divergence statistic, p-value, and degrees of freedom.

        Sets ``self.statistic_`` (chi-squared), ``self.p_value_``, and ``self.dof_``.
        """
        data = self.data

        # Step 1: Do a simple contingency test if there are no conditional variables.
        if len(Z) == 0:
            chi, p_value, dof, expected = stats.chi2_contingency(
                data.groupby([X, Y], observed=False).size().unstack(Y, fill_value=0),
                lambda_=self.lambda_,
            )

        # Step 3: If there are conditionals variables, iterate over unique states
        else:
            chi = 0
            dof = 0
            for z_state, df in data.groupby(list(Z), observed=True):
                # Compute the contingency table
                unique_x, x_inv = np.unique(df[X], return_inverse=True)
                unique_y, y_inv = np.unique(df[Y], return_inverse=True)
                contingency = np.bincount(
                    x_inv * len(unique_y) + y_inv,
                    minlength=len(unique_x) * len(unique_y),
                ).reshape(len(unique_x), len(unique_y))

                # If all values of a column in the contingency table are zeros, skip the test.
                if any(contingency.sum(axis=0) == 0) or any(contingency.sum(axis=1) == 0):
                    if isinstance(z_state, str):
                        logger.info(f"Skipping the test {X} _|_ {Y} | {Z[0]}={z_state}. Not enough samples")
                    else:
                        z_str = ", ".join([f"{var}={state}" for var, state in zip(Z, z_state)])
                        logger.info(f"Skipping the test {X} _|_ {Y} | {z_str}. Not enough samples")
                else:
                    c, _, d, _ = stats.chi2_contingency(contingency, lambda_=self.lambda_)
                    chi += c
                    dof += d
            p_value = 1 - stats.chi2.cdf(chi, df=dof)

        self.statistic_ = chi
        self.p_value_ = p_value
        self.dof_ = dof

        return self.statistic_, self.p_value_
