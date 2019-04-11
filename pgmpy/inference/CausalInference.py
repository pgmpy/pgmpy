import networkx as nx

import numpy as np

from pgmpy.models.CausalGraph import CausalGraph
from pgmpy.estimators.LinearModel import LinearEstimator


class CausalInference(object):
    """
    This is an inference class for performing Causal Inference over Bayesian Networks or Strucural Equation Models.

    This class will accept queries of the form: P(Y | do(X)) and utilize it's methods to provide an estimand which:
     * Identifies adjustment variables
     * Backdoor Adjustment
     * Front Door Adjustment
     * Instrumental Variable Adjustment

    Parameters
    ----------
    model : CausalGraph
        The model that we'll perform inference over.
    set_nodes : list[node:str] or None
        A list (or set/tuple) of nodes in the Bayesian Network which have been set to a specific value per the
        do-operator.

    Examples
    --------
    Create a small Bayesian Network.
    >>> from pgmpy.models.BayesianModel import BayesianModel
    >>> game = CausalGraph([('X', 'A'),
                            ('A', 'Y'),
                            ('A', 'B')])
    Load the graph into the CausalInference object to make causal queries.
    >>> from pgmpy.inference.causal_inferece import CausalInference
    >>> inference = CausalInference(game)
    >>> inference.get_all_backdoor_adjustment_sets(X="X", Y="Y")
    >>> inference.get_all_frontdoor_adjustment_sets(X="X", Y="Y")

    References
    ----------
    'Causality: Models, Reasoning, and Inference' - Judea Pearl (2000)

    Many thanks to @ijmbarr for their implementation of Causal Graphical models available. It served as an invaluable
    reference. Available on GitHub: https://github.com/ijmbarr/causalgraphicalmodels
    """
    def __init__(self, model):
        assert isinstance(model, CausalGraph)
        self.dag = model

    def __repr__(self):
        variables = ", ".join(map(str, sorted(self.observed_variables)))
        return ("{classname}({vars})".format(self.__class__.__name__, variables))

    def simple_decision(self, adjustment_sets):
        """
        Implements a simple decision rule to select a set from all calculated adjustment sets.  
        """
        adjustment_list = list(adjustment_sets)
        if (adjustment_list is None) | (adjustment_list == []):
            return frozenset([])
        return adjustment_list[np.argmin(adjustment_list)]

    def estimate_ate(self, X, Y, data, estimand_strategy="smallest", estimator_type="linear", **kwargs):
        """
        Estimate the average treatment effect of X on Y.

        Parameters
        ----------
        X: str
            Intervention Variable
        Y: str
            Target Variable
        data: pandas DataFrame
            All observed data for this Bayesian Network.
        estimand_strategy: str or frozenset
            Either specify a specific backdoor adjustment set or a strategy.  The available options are:
                smallest:
                    Use the smallest estimand of observed variables
                all:
                    Estimate the ATE from each identified estimand
        estimator_type: str
            The type of model to be used to estimate the ATE.  Right now just linear is supported, but we'll add more
            as use cases arise.
        **kwargs: dict
            Keyward arguments specific to the selected estimator.
            linear:
              missing: str
                Available options are "none", "drop", or "raise"
        """
        valid_estimators = ['linear']
        try:
            assert estimator_type in valid_estimators
        except AssertionError:
            print("{} if not a valid estimator_type.  Please select from {}".format(estimator_type, valid_estimators))

        if isinstance(estimand_strategy, frozenset):
            adjustment_set = frozenset({estimand_strategy})
            assert self.dag.is_valid_backdoor_adjustment_set(X, Y, Z=adjustment_set)
        elif estimand_strategy in ['smallest', 'all']:
            adjustment_sets = self.dag.get_all_backdoor_adjustment_sets(X, Y)
            if estimand_strategy == 'smallest':
                adjustment_sets = frozenset({self.simple_decision(adjustment_sets)})

        if estimator_type == "linear":
            self.estimator = LinearEstimator(self.dag)

        ate = [
            self.estimator.fit(X=X, Y=Y, Z=s, data=data, **kwargs)._get_ate()
            for s in adjustment_sets
        ]
        return np.mean(ate)
