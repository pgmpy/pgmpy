# This extension template provides instructions to add new structure scores to pgmpy.
#
# Please follow the following steps:
# 1. Copy this file to `pgmpy/structure_score/` and rename it as `your_score_name.py` (e.g., `my_score.py`).
#    Note: Do NOT start the filename with an underscore `_`, otherwise it won't be discovered.
# 2. Go through the file and address all the TODOs.
# 3. Add an import in `pgmpy/structure_score/__init__.py` (e.g. `from .my_score import MyScore`).
# 4. Add the class name to the `__all__` list in `pgmpy/structure_score/__init__.py`.
# 5. If you would like to contribute the score to pgmpy, please add tests in
#    `pgmpy/tests/test_structure_score/test_your_score_name.py`.

# TODO: Add any other necessary imports here (e.g., numpy, scipy).
# import numpy as np

from collections.abc import Hashable

from pgmpy.structure_score import BaseStructureScore

# TODO: If your score builds on an existing score (e.g., BIC extends LogLikelihood),
# import and inherit from that class instead of BaseStructureScore.


class MyStructureScore(BaseStructureScore):
    r"""
    [One-line description of the structure score.]

    [Detailed description. Include the local score formula if possible, e.g.:]

    .. math::
        \operatorname{Score}(X_i, \Pi_i) = \ldots

    where :math:`X_i` is the variable, :math:`\Pi_i` is its parent set, and ...

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a variable. Missing values should be
        set to ``numpy.nan``.
    state_names : dict, optional
        Dictionary mapping each variable to its allowed states. If not provided,
        the unique values observed in the data are used (discrete variables only).

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.base import DAG
    >>> from pgmpy.structure_score import MyStructureScore
    >>> data = pd.DataFrame(
    ...     {"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]}
    ... )
    >>> model = DAG([("A", "B"), ("A", "C")])
    >>> score = MyStructureScore(data)
    >>> score.score(model)       # global score summed over all nodes
    >>> score.local_score("B", ("A",))   # local score for one node

    Raises
    ------
    ValueError
        If the data type is incompatible with this score, or if model variables
        are not present in the data.

    References
    ----------
    - :cite:p:`TODO_lastname_year`

    .. note::
        Add a BibTeX entry for your reference to ``docs/references.bib`` using
        the key format ``lastname_year`` (e.g. ``cooper_herskovits_1992``), then
        replace ``TODO_lastname_year`` above with that key.
    """

    # Required: metadata used for auto-discovery by get_scoring_method().
    _tags = {
        # Unique lowercase string key. Passed as scoring_method="my-score" by users.
        "name": "my-score",
        # Data type this score supports. One of: "discrete", "continuous", "mixed".
        "supported_datatype": "discrete",
        # Set to "discrete", "continuous", or "mixed" to make this the default score
        # for that data type. Set to None if this should not be a default.
        "default_for": None,
        # Set to True if this score estimates parameters (e.g., MLE-based penalties).
        "is_parametric": False,
    }

    def __init__(self, data, state_names=None):
        # TODO: Add extra hyperparameters as keyword arguments here (e.g., equivalent_sample_size=10).
        # TODO: Store extra hyperparameters as instance attributes before calling super().
        super().__init__(data, state_names=state_names)
        # TODO: Optionally precompute anything that depends on self.data or self.state_names here
        #       (e.g., BDeu precomputes encoded columns and cardinality arrays for performance).

    def _local_score(self, variable: Hashable, parents: tuple[Hashable, ...]) -> float:
        """
        Compute the local score for `variable` given `parents`.

        Called internally by the cached `local_score()` wrapper — do not call
        this method directly from outside the class.

        Parameters
        ----------
        variable : hashable
            The target variable whose local score is computed.
        parents : tuple of hashable
            The parent variables of `variable` in the candidate structure.

        Returns
        -------
        float
            Local score value. Higher values indicate a better fit.
        """
        # Useful attributes set by BaseStructureScore.__init__:
        #   self.data        — the preprocessed DataFrame
        #   self.variables   — list of all variable names
        #   self.state_names — dict mapping each variable to its list of states
        #   self.dtypes      — dict of variable -> inferred dtype

        # Useful utility for discrete scores:
        #   from pgmpy.utils import get_state_counts
        #   state_counts = get_state_counts(self.data, self.state_names, variable, parents)

        # TODO: Implement the local score computation.
        raise NotImplementedError

    # -------------------------------------------------------------------------
    # Optional overrides
    # -------------------------------------------------------------------------

    # Override structure_prior() to return the absolute log prior for a full graph.
    # Used by score(model). Default returns 0 (flat prior).
    #
    # def structure_prior(self, model) -> float:
    #     return 0.0

    # Override structure_prior_ratio() to return the log prior delta for a single
    # edge operation ("+", "-", "flip"). Used by Hill-Climb/GES per candidate operation.
    # Must be consistent with structure_prior(). Default returns 0.
    #
    # def structure_prior_ratio(self, operation) -> float:
    #     return 0.0
