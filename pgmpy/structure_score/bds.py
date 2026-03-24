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

    def __init__(self, data, equivalent_sample_size=10, state_names=None):
        super().__init__(data, equivalent_sample_size, state_names=state_names)

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

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        r"""
        Compute the local BDs score for ``variable`` given ``parents``.

        The method computes the BDs score using only observed parent configurations and explicit correction terms for
        the unobserved ones:

        .. math::
            \operatorname{BDs}(X_i, \Pi_i) = \left[ \sum_{j \in \mathcal{O}_i} \sum_{k=1}^{r_i} \log \Gamma(N_{ijk} +
            \beta) + (q_i - \tilde{q}_i) r_i \log \Gamma(\beta) \right]
              - \left[ \sum_{j \in \mathcal{O}_i} \log \Gamma(N_{ij} + \alpha) + (q_i - \tilde{q}_i) \log \Gamma(\alpha)
                \right] + \tilde{q}_i \log \Gamma(\alpha)
              - q_i r_i \log \Gamma(\beta),

        where :math:`\mathcal{O}_i` is the set of observed parent configurations, :math:`\tilde{q}_i = |\mathcal{O}_i|`,
        :math:`q_i` is the total number of parent configurations, :math:`r_i` is the cardinality of :math:`X_i`,
        :math:`\alpha = \text{equivalent_sample_size} / \tilde{q}_i`, :math:`\beta = \text{equivalent_sample_size} /
        (r_i q_i)`, and :math:`N_{ij} = \sum_{k=1}^{r_i} N_{ijk}`.
        """
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
