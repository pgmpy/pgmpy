import numpy as np
from sklearn.linear_model import LinearRegression

from pgmpy.parameterization._base import BaseLocalEstimator


class BaseLinearGaussianEstimator(BaseLocalEstimator):
    """Base class for the estimators of a ``LinearGaussianCPD``'s coefficients and standard deviation from its data.

    ``LinearGaussianCPD.fit`` turns the parents and the target into float arrays, drops the rows with weight 0, and
    passes them to ``estimate``, which returns the coefficients and the standard deviation. The estimator gets the rows
    themselves, not their sums of squares: some estimators, such as robust regression, need the rows, and least squares
    on the rows is more accurate than on their sums of squares. Subclasses implement ``estimate``, and set a ``name``.
    """

    _tags = {"parameterization": ["linear_gaussian_cpd"], "supports_weighted_data": True}

    def estimate(
        self, parents: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None
    ) -> tuple[np.ndarray, float]:
        """Return the coefficients and the standard deviation estimated from the data.

        Parameters
        ----------
        parents : numpy.ndarray of shape (n_rows, n_parents)
            The parents' values, in the order of ``LinearGaussianCPD.evidence_``; no columns for a root.
        target : numpy.ndarray of shape (n_rows,)
            The target's values.
        sample_weight : numpy.ndarray of shape (n_rows,) or None
            Positive frequency of each row: weight 2 counts a row twice. ``None`` for unweighted data.

        Returns
        -------
        beta : numpy.ndarray of shape (1 + n_parents,)
            The intercept, then one coefficient per parent.
        std : float
            Standard deviation of the target given its parents.
        """
        raise NotImplementedError


class LinearGaussianOLS(BaseLinearGaussianEstimator):
    """Least squares estimate of a ``LinearGaussianCPD``: the coefficients by weighted least squares, and the standard
    deviation from the residuals. A root gets its weighted mean.

    Parameters
    ----------
    std_estimator : {"unbiased", "mle"}, default="unbiased"
        ``"unbiased"`` divides the sum of squared residuals by the number of rows minus the number of independent
        coefficients: the intercept plus the rank of the parent data, so collinear parents count once. ``"mle"`` divides
        by the number of rows. With sample weights, their total replaces the number of rows.

    Examples
    --------
    >>> import numpy as np
    >>> from pgmpy.parameterization import LinearGaussianOLS
    >>> parents = np.array([[0.0], [1.0], [2.0], [3.0]])
    >>> target = np.array([1.5, 2.5, 4.5, 7.5])
    >>> beta, std = LinearGaussianOLS().estimate(parents, target, None)
    >>> beta.round(2).tolist(), round(std, 3)
    ([1.0, 2.0], 0.707)

    In a ``LinearGaussianCPD``:

    >>> import pandas as pd
    >>> from pgmpy.parameterization import LinearGaussianCPD
    >>> X, y = pd.DataFrame({"x": parents[:, 0]}), pd.Series(target, name="y")
    >>> cpd = LinearGaussianCPD(estimator=LinearGaussianOLS(std_estimator="mle")).fit(X, y)
    >>> round(cpd.std_, 3)
    0.5
    """

    _tags = {"name": "linear_gaussian_ols"}

    def __init__(self, std_estimator: str = "unbiased") -> None:
        self.std_estimator = std_estimator
        super().__init__()

        if std_estimator not in ("unbiased", "mle"):
            raise ValueError(f"std_estimator must be 'unbiased' or 'mle', but is {std_estimator!r}.")

    def estimate(
        self, parents: np.ndarray, target: np.ndarray, sample_weight: np.ndarray | None
    ) -> tuple[np.ndarray, float]:
        # Step 1: Estimate the coefficients by weighted least squares; a root has only its mean.
        weights = np.ones(len(target)) if sample_weight is None else sample_weight
        if parents.shape[1]:
            regression = LinearRegression().fit(parents, target, sample_weight=sample_weight)
            beta = np.concatenate([[regression.intercept_], regression.coef_])
            # rank_ is the rank of the centred parent data, which leaves out the intercept.
            n_coefficients = 1 + regression.rank_
        else:
            beta = np.array([np.average(target, weights=weights)])
            n_coefficients = 1

        # Step 2: Estimate the std from the residuals. The unbiased estimate divides by the total weight minus the
        # independent coefficients; weights meant to total n_coefficients can sum to just above it, so allow for
        # rounding.
        total = weights.sum()
        if self.std_estimator == "unbiased" and total <= n_coefficients * (1 + 1e-9):
            raise ValueError(
                f"The unbiased std needs a total weight, or number of rows, above the {n_coefficients} independent "
                f"coefficients, but it is {total:g}. Fit more data, or use std_estimator='mle'."
            )
        means = beta[0] + parents @ beta[1:]
        ddof = 0 if self.std_estimator == "mle" else n_coefficients
        return beta, float(np.sqrt(np.sum(weights * (target - means) ** 2) / (total - ddof)))
