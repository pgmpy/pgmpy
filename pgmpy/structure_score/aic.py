from pgmpy.structure_score.log_likelihood import LogLikeliHood


class AIC(LogLikeliHood):
    """AIC structure score for discrete Bayesian networks."""

    _tags = {
        "name": "aic-d",
        "supported_datatype": "discrete",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the local AIC score for `variable`."""
        ll, num_parents_states, var_cardinality = self._log_likelihood(variable=variable, parents=parents)
        score = ll - num_parents_states * (var_cardinality - 1)

        return score
