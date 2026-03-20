import pandas as pd

from .power_divergence import PowerDivergence


class GSq(PowerDivergence):
    """
    G squared test for conditional independence. Also commonly known as G-test,
    likelihood-ratio or maximum likelihood statistical significance test.
    Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset on which to test the independence condition.

    Attributes
    ----------
    statistic_ : float
        The G-squared (log-likelihood ratio) test statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.
    dof_ : int
        Degrees of freedom for the test. Set after calling the test.

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/G-test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.ci_tests import GSq
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> test = GSq(data)
    >>> test("A", "C", [], significance_level=0.05)
    True
    >>> test("A", "B", ["D"], significance_level=0.05)
    True
    >>> test("A", "B", ["D", "E"], significance_level=0.05)
    False
    """

    _tags = {
        "name": "g_sq",
        "data_types": ("discrete",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame):
        super().__init__(data=data, lambda_="log-likelihood")
