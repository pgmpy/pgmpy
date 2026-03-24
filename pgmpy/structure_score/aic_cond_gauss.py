from pgmpy.structure_score.log_likelihood_cond_gauss import LogLikelihoodCondGauss


class AICCondGauss(LogLikelihoodCondGauss):
    """AIC structure score for mixed Bayesian networks."""

    _tags = {
        "name": "aic-cg",
        "supported_datatype": "mixed",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        r"""
        Compute the local mixed-data AIC score for ``variable`` given ``parents``.

        The method computes

        .. math::
            \operatorname{AIC}(X_i, \Pi_i)
            = \ell(X_i, \Pi_i) - k_i,

        where :math:`\ell(X_i, \Pi_i)` is the local
        conditional-Gaussian log-likelihood and :math:`k_i` is the number of
        free parameters computed by ``_get_num_parameters``.
        """
        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - k
