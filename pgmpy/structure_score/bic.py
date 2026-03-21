from math import log

from pgmpy.base import DAG
from pgmpy.structure_score.log_likelihood import LogLikeliHood


class BIC(LogLikeliHood):
    """BIC structure score for discrete Bayesian networks."""

    _tags = {
        "name": "bic_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": True,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def local_score(self, variable: str, parents: list[str]) -> float:
        """Compute the local BIC score for `variable`."""
        sample_size = len(self.data)
        ll, num_parents_states, var_cardinality = self._log_likelihood(variable=variable, parents=parents)
        score = ll - 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score
