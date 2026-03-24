from math import lgamma, log

import numpy as np
from scipy.special import gammaln

from pgmpy.structure_score.bdeu import BDeu


class BDs(BDeu):
    r"""
    BDs structure score for discrete Bayesian networks.

    BDs is a sparse-data variant of BDeu that reallocates the equivalent sample size over the observed parent
    configurations instead of all possible configurations. This makes it better suited to discrete datasets with many
    unobserved parent configurations. The local score computed as:

    .. math::
        \operatorname{BDs}(X_i, \Pi_i) =
        \left[
            \sum_{j \in \mathcal{O}_i} \sum_{k=1}^{r_i} \log \Gamma(N_{ijk} + \beta)
            + (q_i - \tilde{q}_i) r_i \log \Gamma(\beta)
        \right]
        - \left[
            \sum_{j \in \mathcal{O}_i} \log \Gamma(N_{ij} + \alpha)
            + (q_i - \tilde{q}_i) \log \Gamma(\alpha)
        \right]
        + \tilde{q}_i \log \Gamma(\alpha)
        - q_i r_i \log \Gamma(\beta),

    where :math:`\mathcal{O}_i` is the set of observed parent configurations, :math:`\tilde{q}_i = |\mathcal{O}_i|`,
    :math:`q_i` is the total number of parent configurations, :math:`r_i` is the cardinality of :math:`X_i`,
    :math:`\alpha = \text{equivalent_sample_size} / \tilde{q}_i`, :math:`\beta = \text{equivalent_sample_size} / (r_i
    q_i)`, and :math:`N_{ij} = \sum_{k=1}^{r_i} N_{ijk}`.

    In the implementation, `state_counts(..., reindex=False)` keeps only the observed parent configurations. The
    `gamma_counts_adj` and `gamma_conds_adj` terms restore the missing contributions from the unobserved ones so the
    returned score matches the full BDs formula. This class also uses the marginal uniform graph prior from Scutari
    (2016).

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable. Missing values should be set to `numpy.nan`.
    equivalent_sample_size : int, optional
        Equivalent sample size used to define the Dirichlet hyperparameters.
    state_names : dict, optional
        Dictionary mapping each variable to its discrete states. If not specified, the unique values observed in the
        data are used.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.structure_score import BDs
    >>> data = pd.DataFrame(
    ...     {"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]}
    ... )
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> score = BDs(data, equivalent_sample_size=5)
    >>> round(score.score(model), 3)
    np.float64(-12.857)
    >>> round(score.local_score("B", ("A",)), 3)
    np.float64(-3.446)

    Raises
    ------
    ValueError
        If the data contains non-discrete variables, or if the model variables are not present
        in the data.

    References
    ----------
    .. [1] Scutari, Marco. An Empirical-Bayes Score for Discrete Bayesian Networks. Journal of Machine Learning
        Research, 2016, pp. 438-48.
    """

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
