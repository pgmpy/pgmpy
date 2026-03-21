from math import lgamma

import numpy as np
from scipy.special import gammaln

from pgmpy.base import DAG
from pgmpy.structure_score.base import BaseStructureScore


class BDeu(BaseStructureScore):
    """BDeu structure score for discrete Bayesian networks."""

    _tags = {
        "name": "bdeu_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": True,
        "is_default": False,
    }

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        self.equivalent_sample_size = equivalent_sample_size
        super().__init__(data, **kwargs)

    def local_score(self, variable: str, parents: list[str]) -> float:
        """Compute the local BDeu score for `variable` given `parents`."""
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        counts_size = num_parents_states * len(self.state_names[variable])
        log_gamma_counts = np.zeros_like(counts, dtype=float)
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts_size
        gammaln(counts + beta, out=log_gamma_counts)

        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        gamma_counts_adj = (num_parents_states - counts.shape[1]) * len(self.state_names[variable]) * gammaln(beta)
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + num_parents_states * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score
