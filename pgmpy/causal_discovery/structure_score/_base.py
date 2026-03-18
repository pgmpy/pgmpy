#!/usr/bin/env python
from typing import Union

import pandas as pd
from skbase.base import BaseObject

from pgmpy.estimators import BaseEstimator
from pgmpy.utils import get_dataset_type


class BaseStructureScore(BaseEstimator, BaseObject):
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

    _tags = {
        "name": "base_structure_score",
        "supported_datatype": None,
        "is_parametric": False,
        "is_default": False,
    }

    def __init__(self, data, **kwargs):
        BaseEstimator.__init__(self, data, **kwargs)

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


class ScoreCacheMixin(BaseStructureScore):
    """
    A mixin class for StructureScore instances, which implement a decomposable score,
    that caches local scores.
    Based on the global decomposition property of Bayesian networks for decomposable scores.

    Parameters
    ----------
    base_scorer: BaseStructureScore instance
         Has to be a decomposable score.
    data: pandas DataFrame instance
        DataFrame instance where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)
    max_size: int (optional, default 10_000)
        The maximum number of elements allowed in the cache. When the limit is reached, the least recently used
        entries will be discarded.
    **kwargs
        Additional arguments that will be handed to the super constructor.

    Reference
    ---------
    Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
    Section 18.3
    """

    def __init__(self, base_scorer, data, max_size=10000, **kwargs):
        assert isinstance(base_scorer, BaseStructureScore), "Base scorer has to be of type BaseStructureScore."

        self.base_scorer = base_scorer
        self.cache = LRUCache(original_function=self._wrapped_original, max_size=int(max_size))
        super().__init__(data, **kwargs)

    def local_score(self, variable, parents):
        hashable = tuple(parents)
        return self.cache(variable, hashable)

    def _wrapped_original(self, variable, parents):
        expected = list(parents)
        return self.base_scorer.local_score(variable, expected)


# link fields
_PREV, _NEXT, _KEY, _VALUE = 0, 1, 2, 3


class LRUCache:
    """
    Least-Recently-Used cache.
    Acts as a wrapper around an arbitrary function and caches the return values.

    Based on the implementation of Raymond Hettinger
    (https://stackoverflow.com/questions/2437617/limiting-the-size-of-a-python-dictionary)

    Parameters
    ----------
    original_function: callable
        The original function that will be wrapped. Return values will be cached.
        The function parameters have to be hashable.
    max_size: int (optional, default 10_000)
        The maximum number of elements allowed within the cache. If the size would be exceeded,
        the least recently used element will be removed from the cache.
    """

    def __init__(self, original_function, max_size=10000):
        self.original_function = original_function
        self.max_size = max_size
        self.mapping = {}

        # oldest
        self.head = [None, None, None, None]
        # newest
        self.tail = [self.head, None, None, None]
        self.head[_NEXT] = self.tail

    def __call__(self, *key):
        mapping, head, tail = self.mapping, self.head, self.tail

        link = mapping.get(key, head)
        if link is head:
            # Not yet in map
            value = self.original_function(*key)
            if len(mapping) >= self.max_size:
                # Unlink the least recently used element
                old_prev, old_next, old_key, old_value = head[_NEXT]
                head[_NEXT] = old_next
                old_next[_PREV] = head
                del mapping[old_key]
            # Add new value as most recently used element
            last = tail[_PREV]
            link = [last, tail, key, value]
            mapping[key] = last[_NEXT] = tail[_PREV] = link
        else:
            # Unlink element from current position
            link_prev, link_next, key, value = link
            link_prev[_NEXT] = link_next
            link_next[_PREV] = link_prev
            # Add as most recently used element
            last = tail[_PREV]
            last[_NEXT] = tail[_PREV] = link
            link[_PREV] = last
            link[_NEXT] = tail
        return value


def get_scoring_method(
    scoring_method: Union[str, "BaseStructureScore"] | None,
    data: pd.DataFrame,
    use_cache: bool,
    **kwargs,
) -> tuple["BaseStructureScore", "BaseStructureScore"]:
    from pgmpy.causal_discovery.structure_score._conditional_gaussian import (
        AICCondGauss,
        BICCondGauss,
        LogLikelihoodCondGauss,
    )
    from pgmpy.causal_discovery.structure_score._discrete import (
        AIC,
        BIC,
        K2,
        BDeu,
        BDs,
        LogLikeliHood,
    )
    from pgmpy.causal_discovery.structure_score._gaussian import (
        AICGauss,
        BICGauss,
        LogLikelihoodGauss,
    )

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
    all_available_methods = [key for subdict in available_methods.values() for key in subdict.keys()]

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
            raise ValueError("The scoring method names have been changed. Please refer the documentation.")
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

    elif not isinstance(scoring_method, BaseStructureScore):
        raise ValueError(
            f"scoring_method should either be one of {all_available_methods} or an instance of BaseStructureScore"
        )

    score: BaseStructureScore
    if isinstance(scoring_method, str):
        score = supported_methods[scoring_method.lower()](data=data, **kwargs)
    else:
        score = scoring_method

    if use_cache:
        score_c = ScoreCacheMixin(score, data)
    else:
        score_c = score

    return score, score_c
