from math import lgamma

import numpy as np
from scipy.special import gammaln

from pgmpy.structure_score._base import BaseStructureScore
from pgmpy.utils import encode_columns, get_state_counts_array


class K2(BaseStructureScore):
    r"""
    K2 structure score for discrete Bayesian networks using uniform Dirichlet priors.

    The K2 score evaluates a Bayesian network structure on fully discrete data under a Dirichlet prior in which all
    pseudo-counts are equal to 1. The local score is computed as:

    .. math::
        \operatorname{K2}(X_i, \Pi_i) = \sum_{j=1}^{q_i} \left[ \log \Gamma(r_i)
            - \log \Gamma(N_{ij} + r_i) + \sum_{k=1}^{r_i} \log \Gamma(N_{ijk} + 1) \right],

    where :math:`r_i` is the cardinality of :math:`X_i`, :math:`q_i` is the number of parent configurations of
    :math:`\Pi_i`, :math:`N_{ijk}` is the count of :math:`X_i = k` in parent configuration :math:`j`, and
    :math:`N_{ij} = \sum_{k=1}^{r_i} N_{ijk}`.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable. Missing values should be set to `numpy.nan`.
    state_names : dict, optional
        Dictionary mapping each variable to its discrete states. If not specified, the unique values observed in the
        data are used.
    max_cache_size : int or None, default=10000
        Maximum number of local scores to cache. If None, the cache is unlimited.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.structure_score import K2
    >>> data = pd.DataFrame(
    ...     {"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]}
    ... )
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> score = K2(data)
    >>> round(score.score(model), 3)
    np.float64(-9.875)
    >>> round(score.local_score("B", ("A",)), 3)
    np.float64(-3.584)

    Raises
    ------
    ValueError
        If the data contains non-discrete variables, or if the model variables are not present in the data.

    References
    ----------
    - :cite:p:`koller_friedman_2009`
    - :cite:p:`liao_2022`
    """

    _tags = {
        "name": "k2",
        "supported_datatype": "discrete",
        "default_for": None,
        "is_parameteric": False,
    }

    def __init__(self, data, state_names=None, max_cache_size=10000):
        super().__init__(data, state_names=state_names, max_cache_size=max_cache_size)
        self._codes, self._cardinalities = encode_columns(self.data, self.state_names)

    def _local_score(self, variable: str, parents: tuple[str, ...]) -> float:
        var_cardinality = self._cardinalities[variable]
        counts = get_state_counts_array(self._codes, self._cardinalities, variable, parents)
        num_parents_states = counts.shape[1]

        log_gamma_counts = np.zeros_like(counts)
        gammaln(counts + 1, out=log_gamma_counts)

        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + var_cardinality, out=log_gamma_conds)

        score = np.sum(log_gamma_counts) - np.sum(log_gamma_conds) + num_parents_states * lgamma(var_cardinality)

        return score
