from __future__ import annotations

from itertools import chain, product
from math import log
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config, logger
from pgmpy.factors.discrete import TabularCPD

from .base import DiscreteParameterEstimator
from .discrete_mle import DiscreteMLE


class DiscreteEM(DiscreteParameterEstimator):
    """
    Class used to compute parameters for a model using Expectation Maximization (EM).

    EM is an iterative algorithm commonly used for estimation when there are latent variables in the model. The
    algorithm iteratively improves the parameter estimates, maximizing the likelihood of the given data.

    Parameters
    ----------
    state_names: dict, optional
        A dict indicating, for each variable, the discrete set of states that the variable can take. If unspecified, the
        observed values in the data set are taken to be the only possible states.

    latent_card: dict, optional
        A dictionary of the form `{latent_var: cardinality}` specifying the cardinality (number of states) of each
        latent variable. If None, assumes `2` states for each latent variable.

    m_step_estimator: discrete parameter estimator instance, optional
        Estimator instance to use in the M-step. The estimator must support weighted data. If not specified, uses
        `DiscreteMLE()`.

    max_iter: int, default=100
        The maximum number of iterations the algorithm is allowed to run for. If `max_iter` is reached, returns the last
        value of parameters.

    atol: float, default=1e-08
        Absolute tolerance used for checking convergence. If the parameter change is less than `atol` in an iteration,
        the algorithm exits.

    n_jobs: int, default=1
        Number of jobs to run in parallel. Using `n_jobs > 1` for small models or datasets might be slower.

    batch_size: int, default=1000
        Number of data points used to compute weights in a batch.

    seed: int, optional
        Random seed to use for generating initial CPDs.

    init_cpds: dict or {"uniform", "random"}, optional
        Initial CPDs for the optimizer. If not specified, CPDs involving latent variables are initialized randomly and
        CPDs involving only observed variables are initialized using the unweighted M-step estimator.

    show_progress: bool, default=True
        Whether to show a progress bar for iterations.

    Attributes
    ----------
    parameters_ : list of TabularCPD
        Learned conditional probability distributions, one per variable in the
        model (including latent variables), ordered by `self._model.nodes()`.
        Populated by `fit`.

    state_names_ : dict
        Mapping from variable name to the list of states for that variable.
        For observed variables the states are inferred from the data; for
        latent variables they are taken from `latent_card` (or default to
        `[0, 1]` if unspecified). Populated by `fit`.

    Examples
    --------
    >>> from pgmpy.datasets import load_dataset
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.parameter_estimator import DiscreteEM
    >>> # Drop the "pe" column so it will be treated as a latent variable.
    >>> data = load_dataset("college_plans").data.drop(columns=["pe"])
    >>> model = DiscreteBayesianNetwork(
    ...     [("ses", "iq"), ("sex", "pe"), ("ses", "pe"), ("iq", "cp"), ("pe", "cp")],
    ...     latents={"pe"},
    ... )
    >>> estimator = DiscreteEM(show_progress=False)
    >>> estimator.fit(model, data).parameters_  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    [<TabularCPD representing P(ses:4) at 0x...>,
     <TabularCPD representing P(iq:4 | ses:4) at 0x...>,
     <TabularCPD representing P(sex:2) at 0x...>,
     <TabularCPD representing P(pe:2 | ses:4, sex:2) at 0x...>,
     <TabularCPD representing P(cp:2 | iq:4, pe:2) at 0x...>]
    """

    _tags = {
        "supports_latent_variables": True,
        "supports_weighted_data": False,
    }

    def __init__(
        self,
        state_names: dict | None = None,
        latent_card: dict[str, int] | None = None,
        m_step_estimator: DiscreteParameterEstimator | None = None,
        max_iter: int = 100,
        atol: float = 1e-08,
        n_jobs: int = 1,
        batch_size: int = 1000,
        seed: int | None = None,
        init_cpds: dict[str, TabularCPD] | str | None = None,
        show_progress: bool = True,
    ) -> None:
        self.latent_card = latent_card
        self.m_step_estimator = m_step_estimator
        self.max_iter = max_iter
        self.atol = atol
        self.n_jobs = n_jobs
        self.batch_size = batch_size
        self.seed = seed
        self.init_cpds = init_cpds
        self.show_progress = show_progress
        super().__init__(state_names=state_names)

    def _get_log_likelihood(self, datapoint: dict[str, Any]) -> float:
        likelihood = 0.0
        for cpd in self._model_copy.cpds:
            scope = set(cpd.scope())
            likelihood += log(
                max(
                    cpd.get_value(**{key: value for key, value in datapoint.items() if key in scope}),
                    1e-10,
                )
            )
        return likelihood

    def _parallel_compute_weights(
        self,
        data_unique,
        latent_card: dict[str, int],
        n_counts: dict[tuple, int],
        offset: int,
        batch_size: int,
    ):
        cache = []
        for i in range(offset, min(offset + batch_size, data_unique.shape[0])):
            latent_combinations = np.array(list(product(*[range(card) for card in latent_card.values()])), dtype=int)
            if latent_combinations.size == 0:
                latent_combinations = np.empty((1, 0), dtype=int)

            df = data_unique.iloc[[i] * latent_combinations.shape[0]].reset_index(drop=True)
            for index, latent_var in enumerate(latent_card.keys()):
                df[latent_var] = latent_combinations[:, index]

            weights = np.e ** (df.apply(lambda t: self._get_log_likelihood(dict(t)), axis=1))
            df["_weight"] = (weights / weights.sum()) * n_counts[tuple(data_unique.iloc[i])]
            cache.append(df)

        return pd.concat(cache)

    def _fit_parameters(self, model, data, sample_weight=None) -> list[TabularCPD]:
        base = self.m_step_estimator if self.m_step_estimator is not None else DiscreteMLE()

        if not isinstance(base, DiscreteParameterEstimator):
            raise TypeError(
                "m_step_estimator should be an instance of a discrete parameter estimator. "
                "Pass an initialized estimator, for example `DiscreteMLE()`."
            )

        if not bool(base.get_tag("supports_weighted_data")):
            raise ValueError(f"{type(base).__name__} doesn't support weighted data and can't be used in EM.")

        params = base.get_params(deep=False)
        params["state_names"] = self.state_names_
        estimator = type(base)(**params)
        estimator.fit(model, data, sample_weight=sample_weight)
        return estimator.parameters_

    def fit(self, model, data, sample_weight=None):
        """
        Estimate model parameters using Expectation Maximization.

        Parameters
        ----------
        model: pgmpy.base.DAG or pgmpy.models.DiscreteBayesianNetwork
            The model structure for which to estimate CPDs.

        data: pandas.DataFrame
            DataFrame object with column names identical to the observed variable names of the network. Fully missing
            columns are treated as latent variables if they are not already marked as latent.

        Returns
        -------
        self: DiscreteEM
            Fitted estimator with learned CPDs stored in `parameters_`.

        Examples
        --------
        >>> from pgmpy.datasets import load_dataset
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.parameter_estimator import DiscreteEM
        >>> # Drop the "pe" column so it will be treated as a latent variable.
        >>> data = load_dataset("college_plans").data.drop(columns=["pe"])
        >>> model = DiscreteBayesianNetwork(
        ...     [("ses", "iq"), ("sex", "pe"), ("ses", "pe"), ("iq", "cp"), ("pe", "cp")],
        ...     latents={"pe"},
        ... )
        >>> estimator = DiscreteEM(show_progress=False)
        >>> estimator.fit(model, data).parameters_  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        [<TabularCPD representing P(ses:4) at 0x...>,
         <TabularCPD representing P(iq:4 | ses:4) at 0x...>,
         <TabularCPD representing P(sex:2) at 0x...>,
         <TabularCPD representing P(pe:2 | ses:4, sex:2) at 0x...>,
         <TabularCPD representing P(cp:2 | iq:4, pe:2) at 0x...>]
        """
        # Step 1: Preprocess model and data.
        #         EM-specific handling first: copy the model so user-supplied latents aren't mutated; promote
        #         fully-missing columns to latent variables; drop rows with partial missingness. Then run the
        #         shared input validation + state-name setup.
        model = model.copy()

        original_cols = set(data.columns)
        data = data.dropna(axis=1, how="all")
        dropped_cols = original_cols - set(data.columns)
        new_latents = [col for col in dropped_cols if col not in model.latents]
        if new_latents:
            logger.warning(
                f"Columns {new_latents} have all missing values and are not marked as latent. "
                "Treating them as latent variables."
            )
            model.latents.update(new_latents)

        original_rows_count = data.shape[0]
        data = data.dropna()
        dropped_rows_count = original_rows_count - data.shape[0]
        if dropped_rows_count:
            logger.warning(
                f"{dropped_rows_count} rows with missing values in partially "
                "missing columns were dropped from the dataset."
            )

        self._initialize_fit(model, data, sample_weight=sample_weight)

        # Step 2: Resolve latent cardinalities and build helper model copies.
        #         `_model_copy` holds the running CPDs across EM iterations; `complete_model` treats latents as
        #         observed for the weighted M-step; `observed_model` drops latents entirely for the initial MLE.
        if self.latent_card is None:
            latent_card = dict.fromkeys(self._model.latents, 2)
        else:
            latent_card = {var: self.latent_card.get(var, 2) for var in self._model.latents}

        for var in self._model.latents:
            if var in self.state_names_:
                if len(self.state_names_[var]) != latent_card[var]:
                    raise ValueError(
                        f"Conflicting cardinality for latent variable {var}: "
                        f"state_names specifies {len(self.state_names_[var])}, "
                        f"latent_card specifies {latent_card[var]}."
                    )
            else:
                self.state_names_[var] = list(range(latent_card[var]))

        self._model_copy = self._model.copy()
        complete_model = self._model.copy()
        complete_model.latents = set()
        observed_model = complete_model.copy()
        observed_model.remove_nodes_from(list(self._model.latents))

        n_states_dict = {key: len(value) for key, value in self.state_names_.items()}
        init_cpds = {} if self.init_cpds is None else self.init_cpds

        # Step 3: Initialize CPDs.
        # Step 3.0: If `init_cpds` is a string, expand it into a dict of random or uniform CPDs for every node.
        if isinstance(init_cpds, str):
            parents_dict = {var: self._model.get_parents(var) for var in self._model.nodes()}
            if init_cpds == "random":
                init_cpds = {
                    var: TabularCPD.get_random(
                        variable=var,
                        evidence=parents_dict[var],
                        cardinality={v: n_states_dict[v] for v in ([var] + parents_dict[var])},
                        state_names={v: self.state_names_[v] for v in ([var] + parents_dict[var])},
                        seed=self.seed,
                    )
                    for var in self._model.nodes()
                }
            elif init_cpds == "uniform":
                init_cpds = {
                    var: TabularCPD.get_uniform(
                        variable=var,
                        evidence=parents_dict[var],
                        cardinality={v: n_states_dict[v] for v in ([var] + parents_dict[var])},
                        state_names={v: self.state_names_[v] for v in ([var] + parents_dict[var])},
                        seed=self.seed,
                    )
                    for var in self._model.nodes()
                }
            else:
                raise ValueError(
                    f"If `init_cpds` is a string, it must be either 'random' or 'uniform'. Got: {init_cpds}"
                )

        # Step 3.1: Partition nodes.
        #           `fixed_cpd_vars` = nodes with no latent involvement; their CPDs are the EM fixed point and
        #           can be estimated once from observed data. `updatable_vars` are refit every EM iteration.
        children_of_latents = set(chain.from_iterable(self._model.get_children(var) for var in self._model.latents))
        fixed_cpd_vars = [
            var
            for var in self._model.nodes()
            if (var not in self._model.latents) and (var not in children_of_latents) and (var not in init_cpds)
        ]
        updatable_vars = [var for var in self._model.nodes() if var not in fixed_cpd_vars]

        fixed_cpds = [
            cpd
            for cpd in self._fit_parameters(observed_model, self._data, sample_weight=None)
            if cpd.variable in fixed_cpd_vars
        ]

        # Step 3.2: Randomly initialize CPDs for updatable variables that don't have a user-supplied init.
        latent_cpds = []
        for node in updatable_vars:
            if node in init_cpds:
                continue

            parents = list(self._model_copy.predecessors(node))
            latent_cpds.append(
                TabularCPD.get_random(
                    variable=node,
                    evidence=parents,
                    cardinality={var: n_states_dict[var] for var in chain([node], parents)},
                    state_names={var: self.state_names_[var] for var in chain([node], parents)},
                    seed=self.seed,
                )
            )

        self._model_copy.add_cpds(*list(chain(fixed_cpds, latent_cpds, list(init_cpds.values()))))

        # Step 4: Run the EM algorithm.
        #         `data_unique` and `n_counts` are iteration-invariant so we precompute them once.
        data_unique = self._data.drop_duplicates()
        n_counts = self._data.groupby(list(self._data.columns), observed=True).size().to_dict()

        disable_pbar = not (self.show_progress and config.SHOW_PROGRESS)
        for _ in tqdm(range(self.max_iter), disable=disable_pbar):
            # Step 4.1: E-step — expand each observation over all latent combinations and weight each augmented
            #           row by the current posterior P(h | x_obs).
            cache = Parallel(n_jobs=self.n_jobs)(
                delayed(self._parallel_compute_weights)(data_unique, latent_card, n_counts, i, self.batch_size)
                for i in range(0, data_unique.shape[0], self.batch_size)
            )
            weighted_data = pd.concat(cache)
            iter_sample_weight = weighted_data.pop("_weight").to_numpy()

            # Step 4.2: M-step — weighted MLE on the completed data, keeping the fixed CPDs and overwriting only
            #           the updatable ones.
            new_cpds = fixed_cpds.copy()
            new_cpds.extend(
                cpd
                for cpd in self._fit_parameters(complete_model, weighted_data, sample_weight=iter_sample_weight)
                if cpd.variable in updatable_vars
            )

            # Step 4.3: Check convergence (parameter-change tolerance). Early-return once all CPDs are within atol.
            new_cpds = self._sort_parameters(new_cpds)
            if all(cpd.__eq__(self._model_copy.get_cpds(node=cpd.scope()[0]), atol=self.atol) for cpd in new_cpds):
                self.parameters_ = new_cpds
                return self

            self._model_copy.cpds = new_cpds

        self.parameters_ = new_cpds
        return self
