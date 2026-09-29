from __future__ import annotations

import numbers
from collections.abc import Hashable, Iterable
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numpy.typing import ArrayLike

from pgmpy.base import DAG
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.utils import get_state_counts
from pgmpy.utils._warnings import _warn_external

from .base import DiscreteParameterEstimator


class DiscreteBayesianEstimator(DiscreteParameterEstimator):
    """
    Class used to compute parameters for a model using Bayesian Parameter Estimation.

    Parameters
    ----------
    state_names: dict, optional
        A dict indicating, for each variable, the discrete set of states that the variable can take. If unspecified, the
        observed values in the data set are taken to be the only possible states.

    prior_type: {"dirichlet", "BDeu", "K2"}, default="BDeu"
        String indicating which type of prior to use for the model parameters. If `prior_type` is `"dirichlet"`,
        `pseudo_counts` specifies the Dirichlet hyperparameters. If `prior_type` is `"BDeu"`, then
        `equivalent_sample_size` is used to construct uniform pseudo counts. `"K2"` is a shorthand for a Dirichlet prior
        with all pseudo counts set to 1.

    equivalent_sample_size: int, float, or dict, default=5
        Equivalent sample size used for the BDeu prior. Can be a single value or a dict specifying the size for each
        variable separately.

    pseudo_counts: int, float, dict, or None, default=None
        Pseudo counts used with the Dirichlet prior. Can be a single value or a dict containing, for each variable, a
        2-D array of shape `(node_cardinality, product(parents_cardinalities))`.

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
    >>> from pgmpy.parameter_estimator import DiscreteBayesianEstimator
    >>> data = load_dataset("college_plans").data
    >>> model = DiscreteBayesianNetwork(
    ...     [("ses", "iq"), ("sex", "pe"), ("ses", "pe"), ("iq", "cp"), ("pe", "cp")]
    ... )
    >>> estimator = DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=5)
    >>> estimator.fit(model, data).parameters_  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [<TabularCPD representing P(ses:4) at 0x...>,
     <TabularCPD representing P(iq:4 | ses:4) at 0x...>,
     <TabularCPD representing P(sex:2) at 0x...>,
     <TabularCPD representing P(pe:2 | ses:4, sex:2) at 0x...>,
     <TabularCPD representing P(cp:2 | iq:4, pe:2) at 0x...>]
    """

    _tags = {
        "supports_latent_variables": False,
        "supports_weighted_data": True,
    }

    def __init__(
        self,
        state_names: dict | None = None,
        prior_type: str = "BDeu",
        equivalent_sample_size: int | float | dict[Any, int | float] = 5,
        pseudo_counts: int | float | dict[Any, np.ndarray | list[list[float]]] | None = None,
        n_jobs: int = 1,
    ) -> None:
        self.prior_type = prior_type
        self.equivalent_sample_size = equivalent_sample_size
        self.pseudo_counts = pseudo_counts
        self.n_jobs = n_jobs
        super().__init__(state_names=state_names)

    @staticmethod
    def _has_ignored_pseudo_counts(
        nodes: Iterable[Hashable],
        prior_type: str,
        pseudo_counts: ArrayLike | dict[Hashable, ArrayLike | None] | None,
    ) -> bool:
        """Check whether pseudo-counts for any requested node will be ignored."""
        if prior_type.lower() == "dirichlet":
            return False

        for node in nodes:
            node_pseudo_counts = pseudo_counts.get(node) if isinstance(pseudo_counts, dict) else pseudo_counts
            if node_pseudo_counts is not None and np.array(node_pseudo_counts).size > 0:
                return True
        return False

    @staticmethod
    def _resolve_pseudo_counts(
        model,
        state_names: dict,
        node,
        prior_type: str,
        equivalent_sample_size: int | float | dict[Any, int | float],
        pseudo_counts: int | float | dict[Any, np.ndarray | list[list[float]]] | None,
    ) -> tuple[np.ndarray, list, list[int], int]:
        node_cardinality = len(state_names[node])
        parents = sorted(model.get_parents(node))
        parents_cardinalities = [len(state_names[parent]) for parent in parents]
        cpd_shape = (node_cardinality, np.prod(parents_cardinalities, dtype=int))

        prior_type = prior_type.lower()
        node_pseudo_counts = pseudo_counts
        if isinstance(pseudo_counts, dict) and not isinstance(pseudo_counts, numbers.Real):
            node_pseudo_counts = pseudo_counts.get(node)

        if prior_type == "k2":
            resolved_pseudo_counts = np.ones(cpd_shape, dtype=int)
        elif prior_type == "bdeu":
            equivalent_sample_size_val = (
                equivalent_sample_size.get(node, 0)
                if isinstance(equivalent_sample_size, dict)
                else equivalent_sample_size
            )
            alpha = float(equivalent_sample_size_val) / (node_cardinality * np.prod(parents_cardinalities, dtype=int))
            resolved_pseudo_counts = np.ones(cpd_shape, dtype=float) * alpha
        elif prior_type == "dirichlet":
            if isinstance(node_pseudo_counts, numbers.Real):
                resolved_pseudo_counts = np.ones(cpd_shape, dtype=int) * node_pseudo_counts
            else:
                node_pseudo_counts = np.array([]) if node_pseudo_counts is None else np.array(node_pseudo_counts)
                if node_pseudo_counts.size == 0:
                    resolved_pseudo_counts = np.zeros(cpd_shape, dtype=float)
                else:
                    if node_pseudo_counts.shape != cpd_shape:
                        raise ValueError(
                            f"The shape of pseudo_counts for the node: {node} must be of shape: {str(cpd_shape)}"
                        )
                    resolved_pseudo_counts = node_pseudo_counts
        else:
            raise ValueError("'prior_type' not specified")

        return resolved_pseudo_counts, parents, parents_cardinalities, node_cardinality

    @staticmethod
    def _estimate_cpd(
        model,
        data,
        state_names: dict,
        node,
        prior_type: str = "BDeu",
        equivalent_sample_size: int | float | dict[Any, int | float] = 5,
        pseudo_counts: int | float | dict[Any, np.ndarray | list[list[float]]] | None = None,
        sample_weight=None,
    ) -> TabularCPD:
        resolved_pseudo_counts, parents, parents_cardinalities, node_cardinality = (
            DiscreteBayesianEstimator._resolve_pseudo_counts(
                model=model,
                state_names=state_names,
                node=node,
                prior_type=prior_type,
                equivalent_sample_size=equivalent_sample_size,
                pseudo_counts=pseudo_counts,
            )
        )
        state_counts = get_state_counts(
            data=data,
            state_names=state_names,
            variable=node,
            parents=parents,
            sample_weight=sample_weight,
        )
        bayesian_counts = state_counts + resolved_pseudo_counts

        cpd = TabularCPD(
            node,
            node_cardinality,
            np.array(bayesian_counts),
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names={var: state_names[var] for var in chain([node], parents)},
        )
        cpd.normalize()
        return cpd

    def fit(
        self,
        model: DAG | DiscreteBayesianNetwork,
        data: pd.DataFrame,
        sample_weight: ArrayLike | None = None,
    ) -> DiscreteBayesianEstimator:
        """
        Estimate model parameters using Bayesian Parameter Estimation.

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
        self: DiscreteBayesianEstimator
            Fitted estimator with learned CPDs stored in `parameters_`.

        Warns
        -----
        UserWarning
            If nonempty `pseudo_counts` for a model node are ignored because the prior is not Dirichlet.

        Examples
        --------
        >>> from pgmpy.datasets import load_dataset
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.parameter_estimator import DiscreteBayesianEstimator
        >>> data = load_dataset("college_plans").data
        >>> model = DiscreteBayesianNetwork(
        ...     [("ses", "iq"), ("sex", "pe"), ("ses", "pe"), ("iq", "cp"), ("pe", "cp")]
        ... )
        >>> estimator = DiscreteBayesianEstimator(prior_type="BDeu", equivalent_sample_size=5)
        >>> estimator.fit(model, data).parameters_  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(ses:4) at 0x...>,
         <TabularCPD representing P(iq:4 | ses:4) at 0x...>,
         <TabularCPD representing P(sex:2) at 0x...>,
         <TabularCPD representing P(pe:2 | ses:4, sex:2) at 0x...>,
         <TabularCPD representing P(cp:2 | iq:4, pe:2) at 0x...>]
        """
        self._initialize_fit(model, data, sample_weight=sample_weight)

        if self._has_ignored_pseudo_counts(self._model.nodes(), self.prior_type, self.pseudo_counts):
            _warn_external(
                f"pseudo_counts is ignored with prior_type={self.prior_type.lower()!r}. "
                "Use prior_type='dirichlet' to specify pseudo_counts.",
                UserWarning,
            )

        parameters = Parallel(n_jobs=self.n_jobs)(
            delayed(DiscreteBayesianEstimator._estimate_cpd)(
                model=self._model,
                data=self._data,
                state_names=self.state_names_,
                node=node,
                prior_type=self.prior_type,
                equivalent_sample_size=self.equivalent_sample_size,
                pseudo_counts=self.pseudo_counts,
                sample_weight=self._sample_weight,
            )
            for node in self._model.nodes()
        )
        self.parameters_ = self._sort_parameters(parameters)
        return self
