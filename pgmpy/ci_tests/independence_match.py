from typing import Optional

from pgmpy.independencies import IndependenceAssertion

from ._base import _BaseCITest


class IndependenceMatch(_BaseCITest):
    """
    Check if `X ⊥⊥ Y | Z` is in `independences`.

    This method is implemented to have a uniform API when the independences
    are provided explicitly instead of being inferred from data.

    Parameters
    ----------
    X : str
        The first variable for testing the independence condition X ⊥⊥ Y | Z.

    Y : str
        The second variable for testing the independence condition X ⊥⊥ Y | Z.
    Z : list or array-like
        A list of conditional variables for testing the condition X ⊥⊥ Y | Z.
    independencies : pgmpy.independencies.Independencies
        The object containing the known independences.

    Returns
    -------
    bool
        True if the independence assertion is present in `independences`, else False.
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

    def test(
        self,
        X: str,
        Y: str,
        Z: Optional[list] = None,
        boolean: bool = True,
        significance_level: float = 0.05,
        **kwargs,
    ) -> bool:
        """
        Test independence assertion.

        Returns
        -------
        bool
            True if the independence assertion is present in `independencies`, else False.
        """
        Z = [] if Z is None else list(Z)
        self._validate_inputs(X, Y, Z)

        independencies = kwargs.get("independencies") or self.independencies

        if independencies is None:
            raise ValueError(
                "independencies must be provided either in __init__ or as a keyword argument."
            )

        return IndependenceAssertion(X, Y, Z) in independencies

    def _compute_statistic(self, X, Y, Z, **kwargs):
        """Not used for IndependenceMatch as it returns boolean directly."""
        raise NotImplementedError("Use test() method instead.")
