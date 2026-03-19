import pandas as pd

from .power_divergence import PowerDivergence


class ChiSquare(PowerDivergence):
    """
    Perform Chi-square conditional independence test.

    Tests the null hypothesis that X is independent from Y given Zs.

    Parameters
    ----------

    data: pandas.DataFrame
        The dataset on which to test the independence condition.

    Returns
    -------
    result : bool or tuple
        If boolean=False, returns (chi, p_value, dof).
        If boolean=True, returns True if p_value > significance_level.

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
    >>> test("A", "C", [], boolean=True, significance_level=0.05)
    True
    >>> test("A", "B", ["D"], boolean=True, significance_level=0.05)
    True
    >>> test("A", "B", ["D", "E"], boolean=True, significance_level=0.05)
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
