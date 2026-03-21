import numpy as np

from pgmpy.base import DAG
from pgmpy.structure_score.base import BaseStructureScore


class LogLikeliHood(BaseStructureScore):
    """Discrete log-likelihood structure score."""

    _tags = {
        "name": "log_likelihood_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def _log_likelihood(self, variable: str, parents: list[str]) -> tuple[float, int, int]:
        var_cardinality = len(self.state_names[variable])
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        np.log(counts, out=log_likelihoods, where=counts > 0)

        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        return (np.sum(log_likelihoods), num_parents_states, var_cardinality)

    def local_score(self, variable: str, parents: list[str]) -> float:
        """Compute the local log-likelihood score for `variable`."""
        ll, _, _ = self._log_likelihood(variable=variable, parents=parents)
        return ll
