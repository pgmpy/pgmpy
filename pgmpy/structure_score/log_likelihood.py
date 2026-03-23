import numpy as np

from pgmpy.structure_score._base import BaseStructureScore


class LogLikelihood(BaseStructureScore):
    """Discrete log-likelihood structure score."""

    _tags = {
        "name": "ll-d",
        "supported_datatype": "discrete",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None):
        super().__init__(data, state_names=state_names)

    def _log_likelihood(self, variable: str, parents: tuple[str, ...]) -> tuple[float, int, int]:
        var_cardinality = len(self.state_names[variable])
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

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        """Compute the local log-likelihood score for `variable`."""
        ll, _, _ = self._log_likelihood(variable=variable, parents=parents)
        return ll
