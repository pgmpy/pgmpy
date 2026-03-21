from math import lgamma

import numpy as np
from scipy.special import gammaln

from pgmpy.base import DAG
from pgmpy.structure_score.base import BaseStructureScore


class K2(BaseStructureScore):
    """K2 structure score for discrete Bayesian networks."""

    _tags = {
        "name": "k2_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super().__init__(data, **kwargs)

    def local_score(self, variable: str, parents: list[str]) -> float:
        """Compute the local K2 score for `variable` given `parents`."""
        var_cardinality = len(self.state_names[variable])
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_gamma_counts = np.zeros_like(counts, dtype=float)

        gammaln(counts + 1, out=log_gamma_counts)

        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + var_cardinality, out=log_gamma_conds)

        score = np.sum(log_gamma_counts) - np.sum(log_gamma_conds) + num_parents_states * lgamma(var_cardinality)

        return score
