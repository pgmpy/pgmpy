from pgmpy.structure_score.log_likelihood_gauss import LogLikelihoodGauss


class AICGauss(LogLikelihoodGauss):
    """AIC structure score for Gaussian Bayesian networks."""

    _tags = {
        "name": "aic-g",
        "supported_datatype": "continuous",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the local Gaussian AIC score for `variable`."""
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        return ll - (df_model + 2)
