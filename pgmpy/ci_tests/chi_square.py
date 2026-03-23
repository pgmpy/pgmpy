import pandas as pd

from .power_divergence import PowerDivergence


class ChiSquare(PowerDivergence):
    """
    Pearson chi-square test for conditional independence on discrete data.

    This class is a thin specialization of :class:`PowerDivergence` with ``lambda_="pearson"``.
    For the contingency-table construction, conditional-case aggregation, and p-value computation,
    see :class:`PowerDivergence`.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset on which to test the independence condition.

    Attributes
    ----------
    statistic_ : float
        The Pearson chi-square test statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.
    dof_ : int
        Degrees of freedom for the test. Set after calling the test.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Chi-squared_test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> test = ChiSquare(data)
    >>> test("A", "C", [], significance_level=0.05)
    True
    >>> test("A", "B", ["D"], significance_level=0.05)
    True
    >>> test("A", "B", ["D", "E"], significance_level=0.05)
    False
    """

    _tags = {
        "name": "chi_square",
        "data_types": ("discrete",),
        "default_for": "discrete",
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame):
        super().__init__(data=data, lambda_="pearson")
