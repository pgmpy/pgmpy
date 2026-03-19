import numpy as np
import pandas as pd
from scipy import stats

from pgmpy.global_vars import logger

from ._base import _BaseCITest


class PowerDivergence(_BaseCITest):
    """
    Computes the Cressie-Read power divergence statistic [1]. The null hypothesis for the test is X is independent of Y
    given Z. A lot of the frequency comparison based statistics (eg. chi-square, G-test etc) belong to power divergence
    family, and are special cases of this test.

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    lambda_: float or string
        The lambda parameter for the power_divergence statistic. Some values of
        lambda_ results in other well known tests:

            * "pearson"             1          "Chi-squared test"
            * "log-likelihood"      0          "G-test or log-likelihood"
            * "freeman-tuckey"     -1/2        "Freeman-Tuckey Statistic"
            * "mod-log-likelihood"  -1         "Modified Log-likelihood"
            * "neyman"              -2         "Neyman's statistic"
            * "cressie-read"        2/3        "The value recommended in the paper[1]"

    Attributes
    ----------
    statistic_ : float
        The chi-squared test statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.
    dof_ : int
        Degrees of freedom for the test. Set after calling the test.

    References
    ----------
    .. [1] Cressie, Noel, and Timothy RC Read. "Multinomial goodness‐of‐fit tests."
      Journal of the Royal Statistical Society: Series B (Methodological) 46.3 (1984): 440-464.

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
