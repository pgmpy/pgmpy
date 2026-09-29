from __future__ import annotations

from itertools import chain

import numpy as np
from joblib import Parallel, delayed

from pgmpy.factors.discrete import TabularCPD
from pgmpy.utils import get_state_counts

from .base import DiscreteParameterEstimator


class DiscreteMLE(DiscreteParameterEstimator):
    """
    Computes parameters for a given discrete model using Maximum Likelihood Estimation.

    Parameters
    ----------
    state_names: dict, optional
        A dict indicating, for each variable, the discrete set of states that the variable can take. If unspecified, the
        observed values in the data set are taken to be the only possible states.

    n_jobs: int, default=1
        Number of jobs to run in parallel. Using `n_jobs > 1` for small models might be slower.

    Attributes
    ----------
    parameters_ : list of TabularCPD
        Learned conditional probability distributions, one per variable in the
        model, ordered by `self._model.nodes()`. Populated by `fit`.

    state_names_ : dict
        Mapping from variable name to the list of states for that variable,
        inferred from the data (or taken from the `state_names` constructor
        argument when supplied). Populated by `fit`.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.parameter_estimator import DiscreteMLE
    >>> data = load_dataset("college_plans").data
    >>> model = DiscreteBayesianNetwork(
    ...     [("ses", "iq"), ("sex", "pe"), ("ses", "pe"), ("iq", "cp"), ("pe", "cp")]
    ... )
    >>> estimator = DiscreteMLE()
    >>> estimator.fit(model, data)
    DiscreteMLE()
    """

    _tags = {
        "supports_latent_variables": False,
        "supports_weighted_data": True,
    }

    def __init__(
        self,
        state_names: dict | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.n_jobs = n_jobs
        super().__init__(state_names=state_names)

    @staticmethod
    def _estimate_cpd(model, data, state_names: dict, node, sample_weight=None) -> TabularCPD:
        parents = sorted(model.get_parents(node))
        state_counts = get_state_counts(
            data=data,
            state_names=state_names,
            variable=node,
            parents=parents,
            sample_weight=sample_weight,
        )
        state_counts.iloc[:, (state_counts.values == 0).all(axis=0)] = 1.0

        parents_cardinalities = [len(state_names[parent]) for parent in parents]
        node_cardinality = len(state_names[node])

        cpd = TabularCPD(
            node,
            node_cardinality,
            np.array(state_counts),
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names={var: state_names[var] for var in chain([node], parents)},
        )
        cpd.normalize()
        return cpd

    def fit(self, model, data, sample_weight=None):
        """
        Estimate model parameters using Maximum Likelihood Estimation.

        Parameters
        ----------
        model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork
            The model structure for which to estimate CPDs.

        data: pandas.DataFrame
            DataFrame object with column names identical to the variable names of the network.

        sample_weight: array-like of shape (n_samples,), optional
            Per-row weights for `data`. If None, each row is weighted equally.

        Returns
        -------
        self: DiscreteMLE
            Fitted estimator with learned CPDs stored in `parameters_`.

        Examples
        --------
        >>> from pgmpy.datasets import load_dataset
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.parameter_estimator import DiscreteMLE
        >>> data = load_dataset("college_plans").data
        >>> model = DiscreteBayesianNetwork(
        ...     [("ses", "iq"), ("sex", "pe"), ("ses", "pe"), ("iq", "cp"), ("pe", "cp")]
        ... )
        >>> estimator = DiscreteMLE()
        >>> estimator.fit(model, data).parameters_  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(ses:4) at 0x...>,
         <TabularCPD representing P(iq:4 | ses:4) at 0x...>,
         <TabularCPD representing P(sex:2) at 0x...>,
         <TabularCPD representing P(pe:2 | ses:4, sex:2) at 0x...>,
         <TabularCPD representing P(cp:2 | iq:4, pe:2) at 0x...>]
        """
        self._initialize_fit(model, data, sample_weight=sample_weight)

        parameters = Parallel(n_jobs=self.n_jobs)(
            delayed(type(self)._estimate_cpd)(
                model=self._model,
                data=self._data,
                state_names=self.state_names_,
                node=node,
                sample_weight=self._sample_weight,
            )
            for node in self._model.nodes()
        )
        self.parameters_ = self._sort_parameters(parameters)
        return self
