#!/usr/bin/env python
from math import lgamma, log
from typing import List

import numpy as np
from scipy.special import gammaln

from pgmpy.base import DAG
from pgmpy.causal_discovery.structure_score import BaseStructureScore


class K2(BaseStructureScore):
    """
    K2 structure score for discrete Bayesian networks using Dirichlet priors.

    The K2 score is commonly used to evaluate the fit of a Bayesian network structure
    on fully discrete data, assuming all Dirichlet hyperparameters (pseudo-counts) are set to 1.
    This metric is suitable for structure learning when variables are categorical and no
    prior preference for particular parameterizations is assumed.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable. Missing values
        should be set to `numpy.nan`. (Note: pandas will convert columns with `numpy.nan` to dtype float.)
    state_names : dict, optional
        Dictionary mapping each variable to its discrete states. If not specified, the unique
        values observed in the data are used as possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import K2
    >>> data = pd.DataFrame({"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]})
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> k2_score = K2(data)
    >>> print(k2_score.score(model))
    -356.1785

    Raises
    ------
    ValueError
        If the data contains continuous variables, or if the model variables are not present in the data.

    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009,
        Section 18.3.4–18.3.6 (esp. page 806).
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    _tags = {
        "name": "k2_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        super(K2, self).__init__(data, **kwargs)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local K2 score for a discrete variable and its parent variables.

        The K2 local score measures how well the conditional probability distribution
        of `variable` given its parents fits the observed data, assuming uniform Dirichlet
        priors (all hyperparameters set to 1). The calculation is based on marginal and
        conditional counts, and is suitable for fully discrete Bayesian networks.

        Parameters
        ----------
        variable : str
            The name of the target variable (child node).
        parents : list of str
            List of parent variable names (categorical/discrete).

        Returns
        -------
        score : float
            The local K2 score for the specified variable and parent configuration.

        Examples
        --------
        >>> variable = "B"
        >>> parents = ["A"]
        >>> s = k2_score.local_score(variable, parents)
        >>> print(s)
        -42.18

        Raises
        ------
        ValueError
            If `variable` or any parent is not present in `state_names` or data, or if the data
            is not fully discrete.

        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009,
            Section 18.3.4–18.3.6 (esp. page 806).
        """

        var_states = self.state_names[variable]
        var_cardinality = len(var_states)
        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        log_gamma_counts = np.zeros_like(counts, dtype=float)

        # Compute log(gamma(counts + 1))
        gammaln(counts + 1, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + var_cardinality, out=log_gamma_conds)

        # TODO: Check why is this needed
        #
        # Adjustments when using reindex=False as it drops columns of 0 state counts
        # gamma_counts_adj = (
        #     (num_parents_states - counts.shape[1]) * var_cardinality * gammaln(1)
        # )
        # gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(
        #     var_cardinality
        # )
        # log_gamma_counts += gamma_counts_adj
        # log_gamma_conds += gamma_conds_adj

        score = (
            np.sum(log_gamma_counts)
            - np.sum(log_gamma_conds)
            + num_parents_states * lgamma(var_cardinality)
        )

        return score


class BDeu(BaseStructureScore):
    """
    BDeu structure score for discrete Bayesian networks with Dirichlet priors.

    The BDeu score evaluates Bayesian network structures using an "equivalent sample size"
    to define Dirichlet prior hyperparameters, making it flexible for various data sizes
    and uncertainty levels. Use this score when you want to control the influence of your prior
    belief through the equivalent sample size.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable.
        Missing values should be set as `numpy.nan`.
        Note: pandas converts such columns to dtype float.

    equivalent_sample_size : int, optional (default: 10)
        The equivalent (imaginary) sample size for the Dirichlet hyperparameters.
        The score is sensitive to this value; experiment with different values as needed.

    state_names : dict, optional
        Dictionary mapping variable names to their discrete states.
        If not specified, unique values observed in the data are used as possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import BDeu
    >>> data = pd.DataFrame({"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]})
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> bdeu_score = BDeu(data, equivalent_sample_size=5)
    >>> print(bdeu_score.score(model))
    -241.872

    Raises
    ------
    ValueError
        If the data contains continuous variables, or if the model variables are not present in the data.

    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009,
        Section 18.3.4–18.3.6 (esp. page 806).
    [2] AM Carvalho, Scoring functions for learning Bayesian networks,
        http://www.lx.it.pt/~asmc/pub/talks/09-TA/ta_pres.pdf
    """

    _tags = {
        "name": "bdeu_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": True,
        "is_default": False,
    }

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        self.equivalent_sample_size = equivalent_sample_size
        super(BDeu, self).__init__(data, **kwargs)

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local BDeu score for a given variable and its parent variables.

        This method calculates how well a given variable is explained by its parents
        according to the BDeu scoring metric, incorporating the equivalent sample size
        as the Dirichlet prior.

        Parameters
        ----------
        variable : str
            The name of the variable for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local BDeu score for the specified variable and parent configuration.

        Raises
        ------
        ValueError
            If `variable` or any parent is not found in state_names or data.
        """

        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        # The counts_size reflects the full possible table, including dropped zero-count columns.
        counts_size = num_parents_states * len(self.state_names[variable])
        log_gamma_counts = np.zeros_like(counts, dtype=float)
        alpha = self.equivalent_sample_size / num_parents_states
        beta = self.equivalent_sample_size / counts_size
        # Compute log(gamma(counts + beta)) for the observed state counts.
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma of the conditional sample size.
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        # Adjustment for missing zero-count columns (when using reindex=False to save memory).
        gamma_counts_adj = (
            (num_parents_states - counts.shape[1])
            * len(self.state_names[variable])
            * gammaln(beta)
        )
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        # Final BDeu local score calculation.
        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + num_parents_states * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score


class BDs(BDeu):
    """
    BDs (Bayesian Dirichlet sparse) structure score for discrete Bayesian networks.

    The BDs score is a variant of the BDeu score that sets Dirichlet hyperparameters
    (pseudo-counts) proportional to the number of observed parent configurations,
    leading to improved scoring in sparse or partially observed data scenarios.

    Use this score when you expect many possible parent configurations in your data
    to be unobserved (common in sparse or high-dimensional discrete datasets).

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a discrete variable.
        Missing values should be set as `numpy.nan`.
        Note: pandas converts such columns to dtype float.
    equivalent_sample_size : int, optional (default: 10)
        The equivalent (imaginary) sample size for the Dirichlet hyperparameters.
        The score is sensitive to this value; try different values if needed.
    state_names : dict, optional
        Dictionary mapping variable names to their discrete states.
        If not specified, unique values observed in the data are used as possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import BDs
    >>> data = pd.DataFrame({"A": [0, 1, 1, 0], "B": [1, 0, 1, 0], "C": [1, 1, 1, 0]})
    >>> model = DiscreteBayesianNetwork([("A", "B"), ("A", "C")])
    >>> bds_score = BDs(data, equivalent_sample_size=5)
    >>> print(bds_score.score(model))
    -210.314

    Raises
    ------
    ValueError
        If the data contains continuous variables, or if the model variables are not present in the data.

    References
    ----------
    [1] Scutari, Marco. An Empirical-Bayes Score for Discrete Bayesian Networks.
        Journal of Machine Learning Research, 2016, pp. 438–48
    """

    _tags = {
        "name": "bds_structure_score",
        "supported_datatype": (DAG,),
        "is_parameteric": True,
        "is_default": False,
    }

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        super(BDs, self).__init__(data, equivalent_sample_size, **kwargs)

    def structure_prior_ratio(self, operation):
        """
        Computes the log ratio of prior probabilities for a proposed change to the DAG structure.

        This method implements the marginal uniform prior for the graph structure, where the
        log prior probability ratio is -log(2) for adding an edge, log(2) for removing an edge,
        and 0 otherwise.

        Parameters
        ----------
        operation : str
            The proposed operation on the Directed Acyclic Graph (DAG).
            Use "+" for adding an edge, "-" for removing an edge, or other values for no change.

        Returns
        -------
        prior_ratio : float
            The log ratio of the prior probabilities for the proposed operation.

        Examples
        --------
        >>> from pgmpy.estimators import BDs
        >>> score = BDs(data)
        >>> score.structure_prior_ratio("+")
        -0.6931471805599453
        >>> score.structure_prior_ratio("-")
        0.6931471805599453
        >>> score.structure_prior_ratio("noop")
        0
        """
        if operation == "+":
            return -log(2.0)
        if operation == "-":
            return log(2.0)
        return 0

    def structure_prior(self, model):
        """
        Computes the marginal uniform prior for a Bayesian network structure.

        This method assigns a marginal uniform prior to the graph structure, where
        the probability of an arc (edge) between any two nodes (in either direction) is 1/4,
        and the probability of no arc between any two nodes is 1/2. The returned value
        is the log prior probability for the given model structure.

        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The Bayesian network model for which to compute the structure prior.

        Returns
        -------
        score : float
            The log prior probability of the given network structure under the marginal uniform prior.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import BDs
        >>> model = DiscreteBayesianNetwork([("A", "B"), ("C", "D")])
        >>> score = BDs(data)
        >>> prior = score.structure_prior(model)
        >>> print(prior)
        -4.1588830833596715
        """
        nedges = float(len(model.edges()))
        nnodes = float(len(model.nodes()))
        possible_edges = nnodes * (nnodes - 1) / 2.0
        score = -(nedges + possible_edges) * log(2.0)
        return score

    def local_score(self, variable: str, parents: List[str]) -> float:
        """
        Computes the local BDs score for a variable and its parent variables.

        The BDs local score quantifies how well the given variable is explained by its
        specified parent set, using a Bayesian Dirichlet sparse prior. The hyperparameters
        are adjusted based on the number of observed parent configurations, making the score
        more robust in sparse data scenarios.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local BDs score for the specified variable and parent configuration.

        Examples
        --------
        >>> variable = "B"
        >>> parents = ["A"]
        >>> score = bds_score.local_score(variable, parents)
        >>> print(score)
        -38.215

        Raises
        ------
        ValueError
            If `variable` or any parent is not present in `state_names` or data, or if
            the data contains unsupported types (e.g., continuous values).
        """

        parents = list(parents)
        state_counts = self.state_counts(variable, parents, reindex=False)
        num_parents_states = np.prod([len(self.state_names[var]) for var in parents])

        counts = np.asarray(state_counts)
        # counts size is different because reindex=False is dropping columns.
        counts_size = num_parents_states * len(self.state_names[variable])
        log_gamma_counts = np.zeros_like(counts, dtype=float)
        alpha = self.equivalent_sample_size / state_counts.shape[1]
        beta = self.equivalent_sample_size / counts_size
        # Compute log(gamma(counts + beta))
        gammaln(counts + beta, out=log_gamma_counts)

        # Compute the log-gamma conditional sample size
        log_gamma_conds = np.sum(counts, axis=0, dtype=float)
        gammaln(log_gamma_conds + alpha, out=log_gamma_conds)

        # Adjustment because of missing 0 columns when using reindex=False for computing state_counts to save memory.
        gamma_counts_adj = (
            (num_parents_states - counts.shape[1])
            * len(self.state_names[variable])
            * gammaln(beta)
        )
        gamma_conds_adj = (num_parents_states - counts.shape[1]) * gammaln(alpha)

        score = (
            (np.sum(log_gamma_counts) + gamma_counts_adj)
            - (np.sum(log_gamma_conds) + gamma_conds_adj)
            + state_counts.shape[1] * lgamma(alpha)
            - counts_size * lgamma(beta)
        )
        return score
