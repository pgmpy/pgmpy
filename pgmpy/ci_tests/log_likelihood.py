import pandas as pd

from .power_divergence import PowerDivergence


class LogLikelihood(PowerDivergence):
    """
    Log likelihood ratio test for conditional independence. Also commonly known
    as G-test, G-squared test or maximum likelihood statistical significance
    test.  Tests the null hypothesis that X is independent of Y given Zs.

    Parameters
    ----------
    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    Returns
    -------
    CI Test Results: tuple or bool
        If boolean = False, Returns a tuple (chi, p_value, dof). `chi` is the
        chi-squared test statistic. The `p_value` for the test, i.e. the
        probability of observing the computed chi-square statistic (or an even
        higher value), given the null hypothesis that X ⊥⊥ Y | Zs is True.
        If boolean = True, returns True if the p_value of the test is greater
        than `significance_level` else returns False.

    References
    ----------
    [1] https://en.wikipedia.org/wiki/G-test

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> test = LogLikelihood(data)
    >>> test("A", "C", [], boolean=True, significance_level=0.05)
    True
    >>> test("A", "B", ["D"], boolean=True, significance_level=0.05)
    True
    >>> test("A", "B", ["D", "E"], boolean=True, significance_level=0.05)
    False
    """

    _tags = {
        "name": "log_likelihood",
        "data_types": ("discrete",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame):
        super().__init__(data=data, lambda_="log-likelihood")
