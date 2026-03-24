from pgmpy.structure_score.log_likelihood_cond_gauss import LogLikelihoodCondGauss


class AICCondGauss(LogLikelihoodCondGauss):
    r"""
    AIC structure score for Bayesian networks with mixed discrete and continuous variables.

    This score penalizes the conditional-Gaussian log-likelihood by the number of free parameters. The local score
    computed by `local_score(variable, parents)` is

    .. math::
        \operatorname{AIC}(X_i, \Pi_i) = \ell(X_i, \Pi_i) - k_i,

    where :math:`\ell(X_i, \Pi_i)` is the local conditional-Gaussian log-likelihood and :math:`k_i` is the number of
    free parameters returned by `_get_num_parameters`.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where columns may be discrete or continuous variables.
    state_names : dict, optional
        Dictionary mapping discrete variable names to their possible states.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.structure_score import AICCondGauss
    >>> rng = np.random.default_rng(0)
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": rng.normal(size=100),
    ...         "B": rng.integers(0, 2, size=100),
    ...         "C": rng.normal(size=100),
    ...     }
    ... )
    >>> score = AICCondGauss(data)
    >>> round(score.local_score("A", ("B", "C")), 3)
    np.float64(-141.319)

    Raises
    ------
    ValueError
        If the log-likelihood or parameter count cannot be computed for the given local configuration.

    References
    ----------
    .. [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian Networks of Mixed Variables. International
        Journal of Data Science and Analytics, 6(1), 3-18. https://doi.org/10.1007/s41060-017-0085-7
    """

    _tags = {
        "name": "aic-cg",
        "supported_datatype": "mixed",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - k
