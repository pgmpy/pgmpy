from pgmpy.independencies import IndependenceAssertion

from ._base import _BaseCITest


class IndependenceMatch(_BaseCITest):
    """
    Check if `X ⊥⊥ Y | Z` is in `independences`.

    This method is implemented to have a uniform API when the independences
    are provided explicitly instead of being inferred from data.

    Parameters
    ----------
    independencies : pgmpy.independencies.Independencies
        The object containing the known independencies.
    """

    _tags = {
        "name": "independence_match",
        "data_types": ("discrete", "continuous", "mixed"),
        "default_for": None,
        "requires_data": False,
    }

    def __init__(self, independencies=None):
        self.independencies = independencies
        super().__init__()

    def run_test(self, X, Y, Z):
        """
        Check whether the independence assertion X ⊥⊥ Y | Z is present.

        Parameters
        ----------
        X : str
            The first variable.
        Y : str
            The second variable.
        Z : list
            Conditioning variables.

        Returns
        -------
        statistic : None
            No test statistic (this is a lookup, not a statistical test).
        p_value : float
            1.0 if the assertion is found, 0.0 otherwise.
        """
        if self.independencies is None:
            raise ValueError("independencies must be provided in __init__.")

        self.statistic_ = None
        self.p_value_ = 1.0 if IndependenceAssertion(X, Y, list(Z)) in self.independencies else 0.0

        return self.statistic_, self.p_value_
