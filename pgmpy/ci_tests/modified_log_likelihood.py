import pandas as pd

from .power_divergence import PowerDivergence


class ModifiedLogLikelihood(PowerDivergence):
    """
    Modified log-likelihood ratio test for conditional independence on discrete data.

    This class is a thin specialization of :class:`PowerDivergence` with
    ``lambda_="mod-log-likelihood"``. For the contingency-table construction,
    conditional-case aggregation, and p-value computation, see :class:`PowerDivergence`.

    Parameters
    ----------
    data : pandas.DataFrame
        The dataset on which to test the independence condition.

    Attributes
    ----------
    statistic_ : float
        The modified log-likelihood ratio test statistic. Set after calling the test.
    p_value_ : float
        The p-value for the test. Set after calling the test.
    dof_ : int
        Degrees of freedom for the test. Set after calling the test.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> np.random.seed(42)
    >>> data = pd.DataFrame(
    ...     data=np.random.randint(low=0, high=2, size=(50000, 4)), columns=list("ABCD")
    ... )
    >>> data["E"] = data["A"] + data["B"] + data["C"]
    >>> test = ModifiedLogLikelihood(data=data)
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
        "name": "modified_log_likelihood",
        "data_types": ("discrete",),
        "default_for": None,
        "requires_data": True,
    }

    def __init__(self, data: pd.DataFrame):
        super().__init__(data=data, lambda_="mod-log-likelihood")
