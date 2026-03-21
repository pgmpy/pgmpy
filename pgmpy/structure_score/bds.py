from math import lgamma, log

import numpy as np
from scipy.special import gammaln

from pgmpy.structure_score.bdeu import BDeu


class BDs(BDeu):
    """BDs structure score for discrete Bayesian networks."""

    _tags = {
        "name": "bds",
        "supported_datatype": "discrete",
        "default_for": None,
        "is_parameteric": True,
    }

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        super().__init__(data, equivalent_sample_size, **kwargs)

    def structure_prior_ratio(self, operation) -> float:
        """Compute the prior ratio for a graph edit."""
        if operation == "+":
            return -log(2.0)
        if operation == "-":
            return log(2.0)
        return 0

    def structure_prior(self, model) -> float:
        """Compute the marginal uniform prior for a structure."""
        nedges = float(len(model.edges()))
        nnodes = float(len(model.nodes()))
        possible_edges = nnodes * (nnodes - 1) / 2.0
        score = -(nedges + possible_edges) * log(2.0)
        return score

    def local_score(self, variable: str, parents: list[str]) -> float:
        """Compute the local BDs score for `variable` given `parents`."""
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        counts_size = num_parents_states * len(self.state_names[variable])
        log_gamma_counts = np.zeros_like(counts, dtype=float)
        alpha = self.equivalent_sample_size / state_counts.shape[1]
        beta = self.equivalent_sample_size / counts_size
        gammaln(counts + beta, out=log_gamma_counts)

        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        gamma_counts_adj = (num_parents_states - counts.shape[1]) * len(self.state_names[variable]) * gammaln(beta)
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + state_counts.shape[1] * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score
