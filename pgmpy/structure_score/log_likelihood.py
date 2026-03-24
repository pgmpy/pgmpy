import numpy as np

from pgmpy.structure_score._base import BaseStructureScore


class LogLikelihood(BaseStructureScore):
    r"""
    Log-likelihood structure score for discrete Bayesian networks.

    This score evaluates a discrete Bayesian network structure by computing the unpenalized log-likelihood of the
    observed data. The local score is computed as:

    .. math::
        \ell(X_i, \Pi_i) = \sum_{j=1}^{q_i} \sum_{k=1}^{r_i} N_{ijk} \log \frac{N_{ijk}}{N_{ij}},

    with the convention :math:`0 \log 0 = 0`, where :math:`r_i` is the cardinality of :math:`X_i`, :math:`q_i` is the
    number of parent configurations of :math:`\Pi_i`, :math:`N_{ijk}` is the count of :math:`X_i = k` in parent
    configuration :math:`j`, and :math:`N_{ij} = \sum_{k=1}^{r_i} N_{ijk}`.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable. Missing values should be set to `numpy.nan`.
    state_names : dict, optional
        Dictionary mapping each variable to its discrete states. If not specified, the unique values observed in the
        data are used.

    Examples
    --------
    >>> import pandas as pd
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.structure_score import LogLikelihood
    >>> data = pd.DataFrame(
    ...     {"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]}
    ... )
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> score = LogLikelihood(data)
    >>> round(score.score(model), 3)
    np.float64(-6.931)
    >>> round(score.local_score("B", ("A",)), 3)
    np.float64(-2.773)

    Raises
    ------
    ValueError
        If the data contains non-discrete variables, or if the model variables are not present in the data.
    """

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
        ll, _, _ = self._log_likelihood(variable=variable, parents=parents)
        return ll
