#!/usr/bin/env python
from math import lgamma, log
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.special import gammaln
from scipy.stats import multivariate_normal

from pgmpy.estimators import BaseEstimator
from pgmpy.utils import get_dataset_type


def get_scoring_method(
    scoring_method: Optional[Union[str, "StructureScore"]],
    data: pd.DataFrame,
    use_cache: bool,
) -> Tuple["StructureScore", "StructureScore"]:
    available_methods = {
        "continuous": {
            "bic-g": BICGauss,
            "ll-g": LogLikelihoodGauss,
            "aic-g": AICGauss,
        },
        "discrete": {
            "bic-d": BIC,
            "k2": K2,
            "bdeu": BDeu,
            "bds": BDs,
            "aic-d": AIC,
            "ll-d": LogLikeliHood,
        },
        "mixed": {
            "bic-cg": BICCondGauss,
            "ll-cg": LogLikelihoodCondGauss,
            "aic-cg": AICCondGauss,
        },
    }
    all_available_methods = [
        key for subdict in available_methods.values() for key in subdict.keys()
    ]

    var_type = get_dataset_type(data)
    supported_methods = available_methods[var_type] | available_methods["mixed"]

    if isinstance(scoring_method, str):
        if scoring_method.lower() in [
            "k2score",
            "bdeuscore",
            "bdsscore",
            "bicscore",
            "aicscore",
        ]:
            raise ValueError(
                "The scoring method names have been changed. Please refer the documentation."
            )
        elif scoring_method.lower() not in list(all_available_methods):
            raise ValueError(
                "Unknown scoring method. Please refer documentation for a list of supported score metrics."
            )
        elif scoring_method.lower() not in list(supported_methods.keys()):
            raise ValueError(
                f"Incorrect scoring method for {var_type}, scoring_method should be one of"
                f"{list(supported_methods.keys())}, received {scoring_method}. {data.dtypes.unique()}"
            )
    elif isinstance(scoring_method, type(None)):
        # automatically determine scoring method, pick first one
        scoring_method = list(available_methods[var_type].keys())[0]

    elif not isinstance(scoring_method, StructureScore):
        raise ValueError(
            f"scoring_method should either be one of {all_available_methods} or an instance of StructureScore"
        )

    score: StructureScore
    if isinstance(scoring_method, str):
        score = supported_methods[scoring_method.lower()](data=data)
    else:
        score = scoring_method

    if use_cache:
        from pgmpy.estimators.ScoreCache import ScoreCache

        score_c = ScoreCache(score, data)
    else:
        score_c = score

    return score, score_c


class StructureScore(BaseEstimator):
    """
    Abstract base class for structure scoring in pgmpy.

    Structure scores are used to evaluate how well a given Bayesian network structure
    fits observed data. This class should not be used directly. Use one of the derived
    classes such as `K2`, `BDeu`, `BIC`, or `AIC` for concrete scoring methods.

    Structure scores are central to model selection in Bayesian networks and are
    particularly useful when comparing candidate network structures in discrete data scenarios.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame in which each column represents a variable. Missing values should
        be marked as `numpy.nan`. Note: Columns with `numpy.nan` will have dtype `float`.

    state_names : dict, optional
        Dictionary mapping each variable name to the set of its discrete states.
        If not specified, the observed values in the data are used as possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import K2
    >>> # Create random data sample with 3 variables, where B and C are identical:
    >>> data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 2)), columns=list("AB"))
    >>> data["C"] = data["B"]
    >>> model1 = DiscreteBayesianNetwork([["A", "B"], ["A", "C"]])
    >>> model2 = DiscreteBayesianNetwork([["A", "B"], ["B", "C"]])
    >>> K2(data).score(model1)
    -24242.367348745247
    >>> K2(data).score(model2)
    -16273.793897051042

    Notes
    -----
    - Use this class as a base for implementing custom structure scores.
    - Use derived classes (`K2`, `BDeu`, `BIC`, `AIC`) for standard scoring approaches.
    - If you provide data with continuous variables or incompatible states, a `ValueError` may be raised.
    - For best results, ensure all variables are discrete and states are correctly specified.

    Raises
    ------
    ValueError
        If data contains unsupported (non-discrete) types, or if the variables
        in the model do not match the data columns.

    References
    ----------
    Koller & Friedman, Probabilistic Graphical Models: Principles and Techniques, 2009, Section 18.3.
    """

    def __init__(self, data, **kwargs):
        super(StructureScore, self).__init__(data, **kwargs)

    def score(self, model):
        """
        Computes a structure score for a given Bayesian network model.

        This method evaluates how well the specified `DiscreteBayesianNetwork`
        fits the observed data, using the structure score metric implemented in the subclass.
        The higher (or less negative) the score, the better the fit between the model and the data.

        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The Bayesian network whose structure is to be scored. All nodes in the
            model must correspond to columns in the input data.

        Returns
        -------
        score : float
            The computed structure score representing the model's fit to the data.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import K2
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(
        ...     np.random.randint(0, 5, size=(5000, 2)), columns=list("AB")
        ... )
        >>> data["C"] = data["B"]
        >>> K2(data).score(DiscreteBayesianNetwork([["A", "B"], ["A", "C"]]))
        -24242.367348745247
        >>> K2(data).score(DiscreteBayesianNetwork([["A", "B"], ["B", "C"]]))
        -16273.793897051042

        Raises
        ------
        ValueError
            If the model contains nodes not present in the data columns, or if the
            data contains unsupported variable types.
        """

        score = 0
        for node in model.nodes():
            score += self.local_score(node, list(model.predecessors(node)))
        score += self.structure_prior(model)
        return score

    def structure_prior(self, model):
        """
        Computes the (log) prior distribution over Bayesian network structures.

        This method returns a uniform prior by default and is currently unused in scoring.
        Override this method in subclasses to implement custom prior distributions
        over network structures.

        Parameters
        ----------
        model : DiscreteBayesianNetwork
            The Bayesian network model for which the structure prior is to be computed.

        Returns
        -------
        prior : float
            The log prior probability of the given model structure. By default, returns 0.

        Examples
        --------
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import K2
        >>> model = DiscreteBayesianNetwork([("A", "B")])
        >>> score = K2(data)
        >>> prior = score.structure_prior(model)
        >>> print(prior)
        0
        """
        return 0

    def structure_prior_ratio(self, operation):
        """
        Computes the log ratio of prior probabilities for a proposed change to the model structure.

        This method returns the log prior probability ratio for a structural operation
        (e.g., adding, removing, or reversing an edge) in the Bayesian network. By default,
        it assumes a uniform prior and returns 0, meaning no structural operation is favored.

        Parameters
        ----------
        operation : tuple or object
            The proposed operation on the Directed Acyclic Graph (DAG), typically represented as a tuple
            describing the change (such as ('add', 'A', 'B') for adding an edge from A to B).

        Returns
        -------
        prior_ratio : float
            The log ratio of the prior probabilities for the proposed operation. By default, returns 0.

        Examples
        --------
        >>> from pgmpy.estimators import K2
        >>> op = ("add", "A", "B")  # Example operation
        >>> score = K2(data)
        >>> ratio = score.structure_prior_ratio(op)
        >>> print(ratio)
        0
        """
        return 0


class K2(StructureScore):
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

    def __init__(self, data, **kwargs):
        super(K2, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
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


class BDeu(StructureScore):
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

    def __init__(self, data, equivalent_sample_size=10, **kwargs):
        self.equivalent_sample_size = equivalent_sample_size
        super(BDeu, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
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

    def local_score(self, variable, parents):
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


class LogLikeliHood(StructureScore):
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

    def local_score(self, variable, parents):
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

    def __init__(self, data, **kwargs):
        super(BIC, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
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

    def __init__(self, data, **kwargs):
        super(AIC, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
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


class LogLikelihoodGauss(StructureScore):
    """
    Log-likelihood structure score for Gaussian Bayesian networks.

    This score evaluates the fit of a continuous (Gaussian) Bayesian network structure
    by computing the (unpenalized) log-likelihood of the observed data given the model,
    using generalized linear modeling. It is suitable for networks with continuous variables.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.

    state_names : dict, optional
        Dictionary mapping variable names to possible states. Not typically used for Gaussian networks.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import LogLikelihoodGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randn(100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = LogLikelihoodGauss(data)
    >>> ll = score.local_score("B", ["A", "C"])
    >>> print(ll)
    -142.125

    Raises
    ------
    ValueError
        If the data contains discrete or non-numeric variables.
    """

    def __init__(self, data, **kwargs):
        super(LogLikelihoodGauss, self).__init__(data, **kwargs)

    def _log_likelihood(self, variable, parents):
        """
        Computes the log-likelihood and degrees of freedom for a Gaussian model.

        This internal method fits a generalized linear model (GLM) for the specified variable
        as a function of its parent variables, using the statsmodels library, and returns the
        log-likelihood and degrees of freedom of the fitted model.

        Parameters
        ----------
        variable : str
            The name of the variable (node) to be predicted.
        parents : list of str
            List of variable names to be used as predictors (parents). If empty, fits an intercept-only model.

        Returns
        -------
        llf : float
            The log-likelihood of the fitted model.
        df_model : int
            The degrees of freedom of the fitted model (number of model parameters estimated).

        Examples
        --------
        >>> llf, df = score._log_likelihood("B", ["A", "C"])
        >>> print(llf, df)
        -142.125 2

        Raises
        ------
        ValueError
            If the GLM cannot be fitted due to missing or non-numeric data.
        """
        if len(parents) == 0:
            glm_model = smf.glm(formula=f"{variable} ~ 1", data=self.data).fit()
        else:
            glm_model = smf.glm(
                formula=f"{variable} ~ {' + '.join(parents)}", data=self.data
            ).fit()

        return (glm_model.llf, glm_model.df_model)

    def local_score(self, variable, parents):
        """
        Computes the log-likelihood score for a variable given its parent variables.

        Fits a generalized linear model (GLM) for the variable as a function of its parents,
        and returns the resulting log-likelihood as the structure score.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The log-likelihood score for the specified variable and parent configuration.

        Examples
        --------
        >>> ll = score.local_score("B", ["A", "C"])
        >>> print(ll)
        -142.125

        Raises
        ------
        ValueError
            If the GLM cannot be fitted due to non-numeric data or missing columns.
        """
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        return ll


class BICGauss(LogLikelihoodGauss):
    """
    BIC (Bayesian Information Criterion) structure score for Gaussian Bayesian networks.

    The BICGauss score evaluates continuous Bayesian network structures by penalizing
    the log-likelihood with a term proportional to the number of model parameters,
    discouraging overfitting. This is the Gaussian version of the BIC/MDL score,
    suitable for networks where all variables are continuous.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import BICGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randn(100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = BICGauss(data)
    >>> s = score.local_score("B", ["A", "C"])
    >>> print(s)
    -111.42

    Raises
    ------
    ValueError
        If the GLM cannot be fitted due to missing or non-numeric data.
    """

    def __init__(self, data, **kwargs):
        super(BICGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """
        Computes the local BIC/MDL score for a variable and its parent variables
        in a Gaussian Bayesian network.

        The score is the log-likelihood minus a penalty term that increases
        with the number of model parameters and sample size.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local BICGauss score for the specified variable and parent configuration.

        Examples
        --------
        >>> s = score.local_score("B", ["A", "C"])
        >>> print(s)
        -111.42

        Raises
        ------
        ValueError
            If the GLM cannot be fitted due to missing or non-numeric data.
        """
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        # Adding +2 to model df to compute the likelihood df.
        return ll - (((df_model + 2) / 2) * np.log(self.data.shape[0]))


class AICGauss(LogLikelihoodGauss):
    """
    AIC (Akaike Information Criterion) structure score for Gaussian Bayesian networks.

    The AICGauss score evaluates continuous Bayesian network structures by penalizing
    the log-likelihood with a term proportional to the number of model parameters.
    The penalty is less severe than BIC and does not depend on sample size, making AIC
    preferable for model selection with smaller datasets.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where each column represents a continuous variable.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import AICGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randn(100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = AICGauss(data)
    >>> s = score.local_score("B", ["A", "C"])
    >>> print(s)
    -97.53

    Raises
    ------
    ValueError
        If the GLM cannot be fitted due to missing or non-numeric data.
    """

    def __init__(self, data, **kwargs):
        super(AICGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """
        Computes the local AIC score for a variable and its parent variables
        in a Gaussian Bayesian network.

        The score is the log-likelihood minus a penalty term that increases with
        the number of model parameters (but not sample size).

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local AICGauss score for the specified variable and parent configuration.

        Examples
        --------
        >>> s = score.local_score("B", ["A", "C"])
        >>> print(s)
        -97.53

        Raises
        ------
        ValueError
            If the GLM cannot be fitted due to missing or non-numeric data.
        """
        ll, df_model = self._log_likelihood(variable=variable, parents=parents)

        # Adding +2 to model df to compute the likelihood df.
        return ll - (df_model + 2)


class LogLikelihoodCondGauss(StructureScore):
    """
    Log-likelihood score for Bayesian networks with mixed discrete and continuous variables.

    This score is based on conditional Gaussian distributions and supports networks
    with both discrete and continuous variables, using the methodology described in [1].
    The local score computes the log-likelihood of the observed data given the
    network structure, handling mixed parent sets as described in the reference.

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where columns can be discrete or continuous variables.
        Variable types should be consistent with the structure.

    state_names : dict, optional
        Dictionary mapping discrete variable names to their possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import LogLikelihoodCondGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randint(0, 2, 100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = LogLikelihoodCondGauss(data)
    >>> ll = score.local_score("A", ["B", "C"])
    >>> print(ll)
    -98.452

    Raises
    ------
    ValueError
        If the data or variable types are not suitable for conditional Gaussian modeling.

    References
    ----------
    [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian
        Networks of Mixed Variables. International journal of data science and
        analytics, 6(1), 3–18. https://doi.org/10.1007/s41060-017-0085-7
    """

    def __init__(self, data, **kwargs):
        super(LogLikelihoodCondGauss, self).__init__(data, **kwargs)

    @staticmethod
    def _adjusted_cov(df):
        """
        Computes an adjusted covariance matrix from the given DataFrame.

        This method returns the sample covariance matrix for the columns in `df`, making sure
        the result is always positive semi-definite. If there are not enough rows to estimate
        covariance (i.e., fewer rows than variables), the identity matrix is returned. If the
        covariance matrix is not positive semi-definite, a small value is added to the diagonal.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame whose columns are the variables for which the covariance matrix is computed.

        Returns
        -------
        cov_matrix : pandas.DataFrame
            The adjusted covariance matrix. If not enough data, returns the identity matrix.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> df = pd.DataFrame(np.random.randn(5, 3), columns=["A", "B", "C"])
        >>> cov = LogLikelihoodCondGauss._adjusted_cov(df)
        >>> print(cov)
                A         B         C
        A  0.802359  0.100722 -0.006956
        B  0.100722  0.818795  0.154614
        C -0.006956  0.154614  0.540758
        """
        # If a number of rows less than number of variables, return variance 1 with no covariance.
        if (df.shape[0] == 1) or (df.shape[0] < len(df.columns)):
            return pd.DataFrame(
                np.eye(len(df.columns)), index=df.columns, columns=df.columns
            )

        # If the matrix is not positive semidefinite, add a small error to make it.
        df_cov = df.cov()
        if np.any(np.isclose(np.linalg.eig(df_cov)[0], 0)):
            df_cov = df_cov + 1e-6
        return df_cov

    def _cat_parents_product(self, parents):
        """
        Computes the product of the number of unique states for each categorical parent.

        For each parent in `parents` that is discrete (not continuous), this method multiplies
        the number of observed unique states. Parents with only one unique value are ignored
        (i.e., do not contribute to the product).

        Parameters
        ----------
        parents : list of str
            List of parent variable names to consider.

        Returns
        -------
        k : int
            The product of unique state counts for each discrete parent in `parents`.

        Examples
        --------
        >>> score._cat_parents_product(["A", "B", "C"])
        6
        """
        k = 1
        for pa in parents:
            if self.dtypes[pa] != "N":
                n_states = self.data[pa].nunique()
                if n_states > 1:
                    k *= self.data[pa].nunique()
        return k

    def _get_num_parameters(self, variable, parents):
        """
        Computes the number of free parameters required for the conditional distribution
        of a variable given its parents in a mixed (discrete and continuous) Bayesian network.

        For a continuous variable, the number of parameters depends on the number of continuous
        parents and the number of configurations of discrete parents. For a discrete variable,
        it depends on the number of categories and parent configurations.

        Parameters
        ----------
        variable : str
            The name of the target variable (child node).
        parents : list of str
            List of parent variable names.

        Returns
        -------
        k : int
            The number of free parameters for the conditional distribution of `variable`
            given its parents.

        Examples
        --------
        >>> score._get_num_parameters("A", ["B", "C"])
        12
        """
        parent_dtypes = [self.dtypes[pa] for pa in parents]
        n_cont_parents = parent_dtypes.count("N")

        if self.dtypes[variable] == "N":
            k = self._cat_parents_product(parents=parents) * (n_cont_parents + 2)
        else:
            if n_cont_parents == 0:
                k = self._cat_parents_product(parents=parents) * (
                    self.data[variable].nunique() - 1
                )
            else:
                k = (
                    self._cat_parents_product(parents=parents)
                    * (self.data[variable].nunique() - 1)
                    * (n_cont_parents + 2)
                )

        return k

    def _log_likelihood(self, variable, parents):
        """
        Computes the conditional log-likelihood for a variable given its parent set,
        supporting both continuous and discrete variables (mixed Bayesian networks).

        For a continuous variable, computes the log-likelihood using conditional Gaussian
        distributions as described in [1]. For a discrete variable, computes the
        log-likelihood based on the joint and marginal probabilities involving both
        discrete and continuous parents.

        Parameters
        ----------
        variable : str
            The variable (node) for which the log-likelihood is computed.
        parents : list of str
            List of parent variable names.

        Returns
        -------
        log_like : float
            The log-likelihood value for the specified variable and parent set.

        Examples
        --------
        >>> ll = score._log_likelihood("A", ["B", "C"])
        >>> print(ll)
        -99.242

        Raises
        ------
        ValueError
            If data is not suitable for log-likelihood computation (e.g., unsupported variable types).

        References
        ----------
        [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian
            Networks of Mixed Variables. International journal of data science and
            analytics, 6(1), 3–18. https://doi.org/10.1007/s41060-017-0085-7
        """
        df = self.data.loc[:, [variable] + parents]

        # If variable is continuous, the probability is computed as:
        # P(C1 | C2, D) = p(C1, C2 | D) / p(C2 | D)
        if self.dtypes[variable] == "N":
            c1 = variable
            c2 = [var for var in parents if self.dtypes[var] == "N"]
            d = list(set(parents) - set(c2))

            # If D = {}, p(C1, C2 | D) = p(C1, C2) and p(C2 | D) = p(C2)
            if len(d) == 0:
                # If C2 = {}, p(C1, C2 | D) = p(C1) and p(C2 | D) = 1.
                if len(c2) == 0:
                    p_c1c2_d = multivariate_normal.pdf(
                        x=df,
                        mean=df.mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df),
                        allow_singular=True,
                    )
                    return np.sum(np.log(p_c1c2_d))
                else:
                    p_c1c2_d = multivariate_normal.pdf(
                        x=df,
                        mean=df.mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df),
                        allow_singular=True,
                    )
                    df_c2 = df.loc[:, c2]
                    p_c2_d = np.maximum(
                        1e-8,
                        multivariate_normal.pdf(
                            x=df_c2,
                            mean=df_c2.mean(axis=0),
                            cov=LogLikelihoodCondGauss._adjusted_cov(df_c2),
                            allow_singular=True,
                        ),
                    )

                    return np.sum(np.log(p_c1c2_d / p_c2_d))
            else:
                log_like = 0
                for d_states, df_d in df.groupby(d, observed=True):
                    p_c1c2_d = multivariate_normal.pdf(
                        x=df_d.loc[:, [c1] + c2],
                        mean=df_d.loc[:, [c1] + c2].mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(
                            df_d.loc[:, [c1] + c2]
                        ),
                        allow_singular=True,
                    )
                    if len(c2) == 0:
                        p_c2_d = 1
                    else:
                        p_c2_d = np.maximum(
                            1e-8,
                            multivariate_normal.pdf(
                                x=df_d.loc[:, c2],
                                mean=df_d.loc[:, c2].mean(axis=0),
                                cov=LogLikelihoodCondGauss._adjusted_cov(
                                    df_d.loc[:, c2]
                                ),
                                allow_singular=True,
                            ),
                        )

                    log_like += np.sum(np.log(p_c1c2_d / p_c2_d))
                return log_like

        # If variable is discrete, the probability is computed as:
        # P(D1 | C, D2) = (p(C| D1, D2) p(D1, D2)) / (p(C| D2) p(D2))
        else:
            d1 = variable
            c = [var for var in parents if self.dtypes[var] == "N"]
            d2 = list(set(parents) - set(c))

            log_like = 0
            for d_states, df_d1d2 in df.groupby([d1] + d2, observed=True):
                # Check if df_d1d2 also has the discrete variables.
                # If C={}, p(C | D1, D2) = 1.
                if len(c) == 0:
                    p_c_d1d2 = 1
                else:
                    p_c_d1d2 = multivariate_normal.pdf(
                        x=df_d1d2.loc[:, c],
                        mean=df_d1d2.loc[:, c].mean(axis=0),
                        cov=LogLikelihoodCondGauss._adjusted_cov(df_d1d2.loc[:, c]),
                        allow_singular=True,
                    )

                # P(D1, D2)
                p_d1d2 = np.repeat(df_d1d2.shape[0] / df.shape[0], df_d1d2.shape[0])

                # If D2 = {}, p(D1 | C, D2) = (p(C | D1, D2) p(D1, D2)) / p(C)
                if len(d2) == 0:
                    if len(c) == 0:
                        p_c_d2 = 1
                    else:
                        p_c_d2 = np.maximum(
                            1e-8,
                            multivariate_normal.pdf(
                                x=df_d1d2.loc[:, c],
                                mean=df.loc[:, c].mean(axis=0),
                                cov=LogLikelihoodCondGauss._adjusted_cov(df.loc[:, c]),
                                allow_singular=True,
                            ),
                        )

                    log_like += np.sum(np.log(p_c_d1d2 * p_d1d2 / p_c_d2))
                else:
                    if len(c) == 0:
                        p_c_d2 = 1
                    else:
                        df_d2 = df
                        for var, state in zip(d2, d_states[1:]):
                            df_d2 = df_d2.loc[df_d2[var] == state]

                        p_c_d2 = np.maximum(
                            1e-8,
                            multivariate_normal.pdf(
                                x=df_d1d2.loc[:, c],
                                mean=df_d2.loc[:, c].mean(axis=0),
                                cov=LogLikelihoodCondGauss._adjusted_cov(
                                    df_d2.loc[:, c]
                                ),
                                allow_singular=True,
                            ),
                        )

                    p_d2 = df.groupby(d2, observed=True).count() / df.shape[0]
                    for var, value in zip(d2, d_states[1:]):
                        p_d2 = p_d2.loc[p_d2.index.get_level_values(var) == value]

                    log_like += np.sum(
                        np.log((p_c_d1d2 * p_d1d2) / (p_c_d2 * p_d2.values.ravel()[0]))
                    )
            return log_like

    def local_score(self, variable, parents):
        """
        Computes the local log-likelihood score for a variable given its parent variables
        in a mixed (discrete and continuous) Bayesian network.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local conditional Gaussian log-likelihood score for the specified variable and parent configuration.

        Examples
        --------
        >>> ll = score.local_score("A", ["B", "C"])
        >>> print(ll)
        -98.452

        Raises
        ------
        ValueError
            If the log-likelihood cannot be computed due to incompatible data or variable types.
        """
        ll = self._log_likelihood(variable=variable, parents=parents)
        return ll


class BICCondGauss(LogLikelihoodCondGauss):
    """
    BIC (Bayesian Information Criterion) score for Bayesian networks with mixed (discrete and continuous) variables.

    The BICCondGauss score evaluates network structures by penalizing the conditional log-likelihood
    with a term proportional to the number of free parameters and the logarithm of sample size.
    This approach generalizes the classic BIC to handle mixed discrete/continuous data as
    described in [1].

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where columns may be discrete or continuous variables.

    state_names : dict, optional
        Dictionary mapping discrete variable names to possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import BICCondGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randint(0, 2, 100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = BICCondGauss(data)
    >>> s = score.local_score("A", ["B", "C"])
    >>> print(s)
    -115.37

    Raises
    ------
    ValueError
        If the log-likelihood or number of parameters cannot be computed for the provided variables.

    References
    ----------
    [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian
        Networks of Mixed Variables. International journal of data science and
        analytics, 6(1), 3–18. https://doi.org/10.1007/s41060-017-0085-7
    """

    def __init__(self, data, **kwargs):
        super(BICCondGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """
        Computes the local BIC score for a variable and its parent set in a mixed Bayesian network.

        The score is calculated as the log-likelihood minus a complexity penalty, which
        is proportional to the number of free parameters and the log of the sample size.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local BICCondGauss score for the specified variable and parent configuration.

        Examples
        --------
        >>> s = score.local_score("A", ["B", "C"])
        >>> print(s)
        -115.37

        Raises
        ------
        ValueError
            If the log-likelihood or parameter count cannot be computed for the given configuration.
        """

        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - ((k / 2) * np.log(self.data.shape[0]))


class AICCondGauss(LogLikelihoodCondGauss):
    """
    AIC (Akaike Information Criterion) score for Bayesian networks with mixed (discrete and continuous) variables.

    The AICCondGauss score evaluates network structures by penalizing the conditional log-likelihood
    with a term equal to the number of free parameters. This generalizes the classic AIC
    to handle Bayesian networks with both discrete and continuous variables [1].

    Parameters
    ----------
    data : pandas.DataFrame
        DataFrame where columns may be discrete or continuous variables.

    state_names : dict, optional
        Dictionary mapping discrete variable names to possible states.

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import AICCondGauss
    >>> data = pd.DataFrame(
    ...     {
    ...         "A": np.random.randn(100),
    ...         "B": np.random.randint(0, 2, 100),
    ...         "C": np.random.randn(100),
    ...     }
    ... )
    >>> score = AICCondGauss(data)
    >>> s = score.local_score("A", ["B", "C"])
    >>> print(s)
    -99.75

    Raises
    ------
    ValueError
        If the log-likelihood or number of parameters cannot be computed for the provided variables.

    References
    ----------
    [1] Andrews, B., Ramsey, J., & Cooper, G. F. (2018). Scoring Bayesian
        Networks of Mixed Variables. International journal of data science and
        analytics, 6(1), 3–18. https://doi.org/10.1007/s41060-017-0085-7
    """

    def __init__(self, data, **kwargs):
        super(AICCondGauss, self).__init__(data, **kwargs)

    def local_score(self, variable, parents):
        """
        Computes the local AIC score for a variable and its parent set in a mixed Bayesian network.

        The score is calculated as the log-likelihood minus the number of free parameters.

        Parameters
        ----------
        variable : str
            The name of the variable (node) for which the local score is to be computed.
        parents : list of str
            List of variable names considered as parents of `variable`.

        Returns
        -------
        score : float
            The local AICCondGauss score for the specified variable and parent configuration.

        Examples
        --------
        >>> s = score.local_score("A", ["B", "C"])
        >>> print(s)
        -99.75

        Raises
        ------
        ValueError
            If the log-likelihood or parameter count cannot be computed for the given configuration.
        """
        ll = self._log_likelihood(variable=variable, parents=parents)
        k = self._get_num_parameters(variable=variable, parents=parents)

        return ll - k
