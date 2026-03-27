import numpy as np

from pgmpy.structure_score.log_likelihood_gauss import LogLikelihoodGauss


class BICGauss(LogLikelihoodGauss):
    r"""
    BIC structure score for Gaussian Bayesian networks.

    This score penalizes the Gaussian log-likelihood to discourage overfitting. The local score is computed as:

    .. math::
        \operatorname{BIC}(X_i, \Pi_i) = \ell(X_i, \Pi_i) - \frac{d_i}{2} \log n,

    where :math:`\ell(X_i, \Pi_i)` is the fitted Gaussian log-likelihood, :math:`d_i = \text{df\_model} + 2` is the
    effective parameter count used by the implementation, and :math:`n` is the number of rows in `self.data`.

    Here `df_model` is the statsmodels degree-of-freedom count for the fitted regressors and excludes the intercept. The
    additional `+ 2` accounts for one intercept parameter and one Gaussian variance parameter.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.
    state_names : dict, optional
        Accepted for API consistency but not typically used for Gaussian networks.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.structure_score import BICGauss
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": rng.normal(size=100),
    ...         "B": rng.normal(size=100),
    ...         "C": rng.normal(size=100),
    ...     }
    ... )
    >>> score = BICGauss(data)
    >>> round(score.local_score("B", ("A", "C")), 3)
    np.float64(-146.37)

    Raises
    ------
    ValueError
        If the model cannot be fitted because the data contains incompatible or non-numeric variables.
    """

    _tags = {
        "name": "bic-g",
        "supported_datatype": "continuous",
        "default_for": "continuous",
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        return ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))
