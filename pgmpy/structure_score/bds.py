from collections.abc import Hashable
from math import lgamma, log

import numpy as np
from scipy.special import gammaln

from pgmpy.structure_score.bdeu import BDeu
from pgmpy.utils import get_state_counts_array


class BDs(BDeu):
    r"""
    BDs structure score for discrete Bayesian networks.

    BDs is a sparse-data variant of BDeu that reallocates the equivalent sample size over the observed parent
    configurations instead of all possible configurations. This makes it better suited to discrete datasets with many
    unobserved parent configurations. The local score computed as:

    .. math::
        \operatorname{BDs}(X_i, \Pi_i) = \sum_{j \in \mathcal{O}_i} \left[
            \log \Gamma(\alpha)
            - \log \Gamma(N_{ij} + \alpha)
            + \sum_{k=1}^{r_i} \left(
                \log \Gamma(N_{ijk} + \beta)
                - \log \Gamma(\beta)
            \right)
        \right],

    where :math:`\mathcal{O}_i` is the set of observed parent configurations (those with :math:`N_{ij} > 0`),
    :math:`\tilde{q}_i = |\mathcal{O}_i|`, :math:`r_i` is the cardinality of :math:`X_i`,
    :math:`\alpha = \text{equivalent_sample_size} / \tilde{q}_i`, :math:`\beta = \text{equivalent_sample_size} / (r_i
    \tilde{q}_i)`, :math:`N_{ijk}` is the count of :math:`X_i = k` in parent configuration :math:`j`, and
    :math:`N_{ij} = \sum_{k=1}^{r_i} N_{ijk}`.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable. Missing values should be set to `numpy.nan`.
    equivalent_sample_size : int, optional
        Equivalent sample size used to define the Dirichlet hyperparameters.
    state_names : dict, optional
        Dictionary mapping each variable to its discrete states. If not specified, the unique values observed in the
        data are used.
    max_cache_size : int or None, default=10000
        Maximum number of local scores to cache. If None, the cache is unlimited.

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
    - :footcite:t:`scutari_2016a`
    """

    _tags = {
        "name": "bds",
        "data_types": ["discrete"],
        "default_for": None,
    }

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

    def _local_score(self, variable: Hashable, parents: tuple[Hashable, ...]) -> float:
        counts = get_state_counts_array(self._codes, self._cardinalities, variable, parents)
        var_cardinality = self._cardinalities[variable]

        # BDs reallocates the equivalent sample size over only the observed parent configs.
        col_sums = np.sum(counts, axis=0, dtype=float)
        observed = col_sums > 0
        n_observed = int(np.count_nonzero(observed))

        alpha = self.equivalent_sample_size / n_observed
        beta = self.equivalent_sample_size / (n_observed * var_cardinality)

        log_gamma_counts = gammaln(counts[:, observed] + beta)
        log_gamma_conds = gammaln(col_sums[observed] + alpha)

        score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + n_observed * lgamma(alpha)
            - n_observed * var_cardinality * lgamma(beta)
        )
        return score
