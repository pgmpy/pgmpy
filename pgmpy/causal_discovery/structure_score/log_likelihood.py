#!/usr/bin/env python
from math import log
from typing import List

import numpy as np

from pgmpy.base import DAG
from pgmpy.causal_discovery.structure_score import BaseStructureScore


class LogLikeliHood(BaseStructureScore):
    """
    Log-likelihood structure score for Discrete Bayesian networks.

    This score evaluates the fit of a Discrete Bayesian network structure
    by computing the (unpenalized) log-likelihood of the observed data given the model.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)
    """

    _tags = {
        "name": "log_likelihood_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super(LogLikeliHood, self).__init__(data, **kwargs)

    def _log_likelihood(self, variable, parents):

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_likelihoods = np.zeros_like(counts, dtype=float)

        # Compute the log-counts
        np.log(counts, out=log_likelihoods, where=counts > 0)

        # Compute the log-conditional sample size
        log_conditionals = np.sum(counts, axis=0, dtype=float)
        np.log(log_conditionals, out=log_conditionals, where=log_conditionals > 0)

        # Compute the log-likelihoods
        log_likelihoods -= log_conditionals
        log_likelihoods *= counts

        return (np.sum(log_likelihoods), num_parents_states, var_cardinality)

    def local_score(self, variable: str, parents: List[str]) -> float:
        ll, num_parents_states, var_cardinality = self._log_likelihood(
            variable=variable, parents=parents
        )
        return ll


class BIC(LogLikeliHood):
    """
    BIC (Bayesian Information Criterion) structure score for discrete Bayesian networks.

    The BIC score, also known as the Minimal Descriptive Length (MDL) score, evaluates
    Bayesian network structures using a log-likelihood term with a complexity penalty to
    discourage overfitting. Use this score for structure learning when you want to balance
    model fit with simplicity.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable.
        Missing values should be set as `numpy.nan`.
        Note: pandas converts such columns to dtype float.
    state_names : dict, optional
        Dictionary mapping variable names to their discrete states.
        If not specified, unique values observed in the data are used as possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import BIC
    >>> data = pd.DataFrame({"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]})
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> bic_score = BIC(data)
    >>> print(bic_score.score(model))
    -151.47

    Raises
    ------
    ValueError
        If the data contains continuous variables, or if the model variables are not present in the data.

    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009,
        Section 18.3.4–18.3.6 (esp. page 802).
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    _tags = {
        "name": "bic_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": True,
    }

    def __init__(self, data, **kwargs):
        super(BIC, self).__init__(data, **kwargs)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local BIC/MDL score for a variable and its parent variables.

        This method quantifies the fit of a variable to its parent set in the network,
        balancing log-likelihood with a complexity penalty to discourage overfitting.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local BIC score for the specified variable and parent configuration.

        Examples
        --------
        >>> variable = "B"
        >>> parents = ["A"]
        >>> score = bic_score.local_score(variable, parents)
        >>> print(score)
        -19.315

        Raises
        ------
        ValueError
            If `variable` or any parent is not present in `state_names` or data, or if
            the data contains unsupported types (e.g., continuous values).
        """

        sample_size = len(self.data)
        ll, num_parents_states, var_cardinality = self._log_likelihood(
            variable=variable, parents=parents
        )
        score = ll - 0.5 * log(sample_size) * num_parents_states * (var_cardinality - 1)

        return score


class AIC(LogLikeliHood):
    """
    AIC (Akaike Information Criterion) structure score for discrete Bayesian networks.

    The AIC score evaluates Bayesian network structures using a log-likelihood term
    with a penalty for model complexity to discourage overfitting. Unlike BIC,
    the penalty term is independent of sample size, making AIC more sensitive to
    goodness of fit in smaller datasets.

    Use this score when you want to select a network structure that balances model
    fit with simplicity, especially in contexts with moderate or small sample sizes.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable.
        Missing values should be set as `numpy.nan`.
        Note: pandas converts such columns to dtype float.
    state_names : dict, optional
        Dictionary mapping variable names to their discrete states.
        If not specified, unique values observed in the data are used as possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import AIC
    >>> data = pd.DataFrame({"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]})
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> aic_score = AIC(data)
    >>> print(aic_score.score(model))
    -140.12

    Raises
    ------
    ValueError
        If the data contains continuous variables, or if the model variables are not present in the data.

    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009,
        Section 18.3.4–18.3.6 (esp. page 802).
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    _tags = {
        "name": "aic_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super(AIC, self).__init__(data, **kwargs)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local AIC score for a variable and its parent variables.

        This method quantifies the fit of a variable to its parent set in the network,
        balancing log-likelihood with a complexity penalty to avoid overfitting.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local AIC score for the specified variable and parent configuration.

        Examples
        --------
        >>> variable = "B"
        >>> parents = ["A"]
        >>> score = aic_score.local_score(variable, parents)
        >>> print(score)
        -17.032

        Raises
        ------
        ValueError
            If `variable` or any parent is not present in `state_names` or data, or if
            the data contains unsupported types (e.g., continuous values).
        """

        ll, num_parents_states, var_cardinality = self._log_likelihood(
            variable=variable, parents=parents
        )
        score = ll - num_parents_states * (var_cardinality - 1)

        return score
