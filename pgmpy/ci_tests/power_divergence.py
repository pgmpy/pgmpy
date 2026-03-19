from typing import Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from pgmpy.global_vars import logger

from ._base import _BaseCITest


class PowerDivergence(_BaseCITest):
    """
    Computes the Cressie-Read power divergence statistic [1]. The null hypothesis
    for the test is X is independent of Y given Z. A lot of the frequency comparision
    based statistics (eg. chi-square, G-test etc) belong to power divergence family,
    and are special cases of this test.

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

    Returns
    -------
    result : bool or tuple
        If boolean=False, returns (chi, p_value, dof).
        If boolean=True, returns True if p_value > significance_level.

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
    >>> test("A", "C", [], boolean=True, significance_level=0.05)
    True
    >>> test("A", "B", ["D"], boolean=True, significance_level=0.05)
    True
    >>> test("A", "B", ["D", "E"], boolean=True, significance_level=0.05)
    False
    """

    _tags = {
        "name": "power_divergence",
        "data_types": ("discrete",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame, lambda_: Union[str, float] = "cressie-read"):
        self.data = data
        self.lambda_ = lambda_
        super().__init__()

    def _compute_statistic(
        self,
        X: str,
        Y: str,
        Z: list,
        **kwargs,
    ) -> Tuple[float, float, int]:
        """
        Compute power divergence statistic, p-value, and degrees of freedom.

        Parameters
        ----------
        X : str
            The first variable for testing the independence condition X ⊥⊥ Y | Z.
        Y : str
            The second variable for testing the independence condition X ⊥⊥ Y | Z.
        Z : list
            A list of conditional variables for testing the condition X ⊥⊥ Y | Z.
        **kwargs
            Additional arguments.

        Returns
        -------
        tuple
            A tuple of (chi, p_value, dof).
        """
        # Step 1: Check if the arguments are valid and type conversions.
        data = self.data
        if (X in Z) or (Y in Z):
            raise ValueError(
                f"The variables X or Y can't be in Z. Found {X if X in Z else Y} in Z."
            )

        # Step 2: Do a simple contingency test if there are no conditional variables.
        if len(Z) == 0:
            chi, p_value, dof, expected = stats.chi2_contingency(
                data.groupby([X, Y], observed=False).size().unstack(Y, fill_value=0),
                lambda_=self.lambda_,
            )

        # Step 3: If there are conditionals variables, iterate over unique states
        else:
            chi = 0
            dof = 0
            for z_state, df in data.groupby(Z, observed=True):
                # Compute the contingency table
                unique_x, x_inv = np.unique(df[X], return_inverse=True)
                unique_y, y_inv = np.unique(df[Y], return_inverse=True)
                contingency = np.bincount(
                    x_inv * len(unique_y) + y_inv,
                    minlength=len(unique_x) * len(unique_y),
                ).reshape(len(unique_x), len(unique_y))

                # If all values of a column in the contingency table are zeros, skip the test.
                if any(contingency.sum(axis=0) == 0) or any(
                    contingency.sum(axis=1) == 0
                ):
                    if isinstance(z_state, str):
                        logger.info(
                            f"Skipping the test {X} _|_ {Y} | {Z[0]}={z_state}. Not enough samples"
                        )
                    else:
                        z_str = ", ".join(
                            [f"{var}={state}" for var, state in zip(Z, z_state)]
                        )
                        logger.info(
                            f"Skipping the test {X} _|_ {Y} | {z_str}. Not enough samples"
                        )
                else:
                    c, _, d, _ = stats.chi2_contingency(
                        contingency, lambda_=self.lambda_
                    )
                    chi += c
                    dof += d
            p_value = 1 - stats.chi2.cdf(chi, df=dof)

        return chi, p_value, dof
