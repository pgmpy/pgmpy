from numbers import Real

import numpy as np
from numpy.typing import ArrayLike
from skbase.base import BaseObject


class BaseTabularEstimator(BaseObject):
    """Base class for the estimators of a ``TabularCPD``'s table from its weighted counts.

    ``TabularCPD.fit`` counts the rows with each state of the target for each combination of parent states, with any
    sample weights, and passes the counts to ``estimate``, which returns the table. With complete data, every estimate
    of the table that uses only this node's data depends on the data only through these counts, so subclasses only
    implement ``estimate``, and set a ``name`` and the other tags registered for local estimators in
    :mod:`pgmpy.registry`.
    """

    _tags = {
        "object_type": "local_estimator",
        "name": None,
        "parameterization": ["tabular_cpd"],
        "supports_weighted_data": True,
    }

    def estimate(self, counts: np.ndarray) -> np.ndarray:
        """Return the table estimated from the counts.

        Parameters
        ----------
        counts : numpy.ndarray of shape (n_states, n_parent_combinations)
            Weighted number of rows with each state of the target (rows) for each combination of parent states
            (columns), in the order of ``TabularCPD.cpt_``.

        Returns
        -------
        numpy.ndarray of shape (n_states, n_parent_combinations)
            Probability of each state of the target for each combination of parent states; each column sums to 1.
        """
        raise NotImplementedError


class TabularMLE(BaseTabularEstimator):
    """Maximum likelihood estimate of a ``TabularCPD``'s table: each column of the counts divided by its total, with a
    uniform distribution for a combination of parent states without data.

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.parameterization import TabularMLE
    >>> TabularMLE().estimate(np.array([[3.0, 0.0], [1.0, 0.0]])).tolist()
    [[0.75, 0.5], [0.25, 0.5]]
    """

    _tags = {"name": "tabular_mle"}

    def estimate(self, counts: np.ndarray) -> np.ndarray:
        return _normalize(counts)


class TabularBayesian(BaseTabularEstimator):
    """Bayesian estimate of a ``TabularCPD``'s table with a Dirichlet prior: the posterior mean, the counts plus the
    prior's pseudo-counts, with each column divided by its total.

    The prior adds 1 to every cell for ``"K2"``, the ``equivalent_sample_size`` spread evenly over the cells for
    ``"BDeu"``, and ``pseudo_counts`` for ``"dirichlet"``. Sample weights count rows, so a weight of 2 counts a row
    twice against the prior.

    Parameters
    ----------
    prior_type : {"BDeu", "K2", "dirichlet"}, default="BDeu"
        The prior, in any case.
    equivalent_sample_size : float, default=5
        Total pseudo-count of the BDeu prior, spread evenly over the table's cells. Only used by ``"BDeu"``.
    pseudo_counts : float or array-like of shape (n_states, n_parent_combinations), optional
        Pseudo-count of every cell, or of each cell in the order of ``TabularCPD.cpt_``. Only for ``"dirichlet"``, which
        needs it.

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.parameterization import TabularBayesian
    >>> TabularBayesian(prior_type="K2").estimate(np.array([[3.0, 0.0], [1.0, 0.0]])).round(2).tolist()
    [[0.67, 0.5], [0.33, 0.5]]

    In a ``TabularCPD``:

    >>> import pandas as pd
    >>> from pgmpy.parameterization import TabularCPD
    >>> data = pd.DataFrame({"rain": ["no", "yes", "no", "no"], "wet": ["no", "yes", "yes", "no"]})
    >>> cpd = TabularCPD(estimator=TabularBayesian(prior_type="K2")).fit(data[["rain"]], data["wet"])
    >>> cpd.cpt_.round(2).tolist()
    [[0.6, 0.33], [0.4, 0.67]]
    """

    _tags = {"name": "tabular_bayesian"}

    def __init__(
        self,
        prior_type: str = "BDeu",
        equivalent_sample_size: float = 5,
        pseudo_counts: float | ArrayLike | None = None,
    ) -> None:
        self.prior_type = prior_type
        self.equivalent_sample_size = equivalent_sample_size
        self.pseudo_counts = pseudo_counts
        super().__init__()

        prior = str(prior_type).lower()
        if prior not in ("bdeu", "k2", "dirichlet"):
            raise ValueError(f"prior_type must be 'BDeu', 'K2' or 'dirichlet', but is {prior_type!r}.")
        if prior == "dirichlet" and pseudo_counts is None:
            raise ValueError("prior_type='dirichlet' needs pseudo_counts.")
        if prior != "dirichlet" and pseudo_counts is not None:
            raise ValueError(f"pseudo_counts is only for prior_type='dirichlet', but prior_type is {prior_type!r}.")
        if isinstance(equivalent_sample_size, bool) or not isinstance(equivalent_sample_size, Real):
            raise ValueError(f"equivalent_sample_size must be a positive number, but is {equivalent_sample_size!r}.")
        if not (np.isfinite(equivalent_sample_size) and equivalent_sample_size > 0):
            raise ValueError(f"equivalent_sample_size must be a positive number, but is {equivalent_sample_size!r}.")
        if pseudo_counts is not None:
            values = np.asarray(pseudo_counts, dtype=float)
            if values.ndim not in (0, 2) or not np.isfinite(values).all() or (values < 0).any():
                raise ValueError("pseudo_counts must be a non-negative number, or a 2-D table of them.")

    def estimate(self, counts: np.ndarray) -> np.ndarray:
        prior = self.prior_type.lower()
        if prior == "k2":
            pseudo_counts = 1.0
        elif prior == "bdeu":
            pseudo_counts = self.equivalent_sample_size / counts.size
        else:
            pseudo_counts = np.asarray(self.pseudo_counts, dtype=float)
            if pseudo_counts.ndim == 2 and pseudo_counts.shape != counts.shape:
                raise ValueError(
                    f"pseudo_counts must have the table's shape {counts.shape}, but has shape {pseudo_counts.shape}."
                )
        return _normalize(counts + pseudo_counts)


def _normalize(counts: np.ndarray) -> np.ndarray:
    """Return each column of the counts divided by its total, and a uniform column where the total is 0."""
    counts = np.array(counts, dtype=float)
    counts[:, counts.sum(axis=0) == 0] = 1.0
    return counts / counts.sum(axis=0)
