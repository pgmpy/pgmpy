from pgmpy.structure_score.log_likelihood_gauss import LogLikelihoodGauss


class AICGauss(LogLikelihoodGauss):
    r"""
    AIC structure score for Gaussian Bayesian networks.

    This score penalizes the Gaussian log-likelihood using a sample-size independent complexity term. The local score
    is defined as:

    .. math::
        \operatorname{AIC}(X_i, \Pi_i) = \ell(X_i, \Pi_i) - d_i,

    where :math:`\ell(X_i, \Pi_i)` is the fitted Gaussian log-likelihood and :math:`d_i = \text{df\_model} + 2` is the
    effective parameter count used by the implementation.

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
    >>> from pgmpy.structure_score import AICGauss
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": rng.normal(size=100),
    ...         "B": rng.normal(size=100),
    ...         "C": rng.normal(size=100),
    ...     }
    ... )
    >>> score = AICGauss(data)
    >>> round(score.local_score("B", ("A", "C")), 3)
    np.float64(-141.16)

    Raises
    ------
    ValueError
        If the model cannot be fitted because the data contains incompatible or non-numeric
        variables.
    """

    _tags = {
        "name": "aic-g",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        return ll - (df_model + 2)
