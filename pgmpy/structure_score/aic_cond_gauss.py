from pgmpy.structure_score.log_likelihood_cond_gauss import LogLikelihoodCondGauss


class AICCondGauss(LogLikelihoodCondGauss):
    """AIC structure score for mixed Bayesian networks."""

    _tags = {
        "name": "aic-cg",
        "supported_datatype": "mixed",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the local mixed-data AIC score for `variable`."""
        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - k
