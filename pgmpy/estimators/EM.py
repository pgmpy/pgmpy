from itertools import chain, product
from math import log

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import DAG
from pgmpy.estimators import MaximumLikelihoodEstimator, ParameterEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.global_vars import logger
from pgmpy.models import DiscreteBayesianNetwork


class ExpectationMaximization(ParameterEstimator):
    """
    Class used to compute parameters for a model using Expectation
    Maximization (EM).

    EM is an iterative algorithm commonly used for
    estimation in the case when there are latent variables in the model.
    The algorithm iteratively improves the parameter estimates maximizing
    the likelihood of the given data.

    Parameters
    ----------
    model: A pgmpy.models.DiscreteBayesianNetwork instance

    data: pandas DataFrame object
        DataFrame object with column names identical to the variable names
        of the network.  (If some values in the data are missing the data
        cells should be set to `numpy.nan`.  Note that pandas converts each
        column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states
        that the variable can take. If unspecified, the observed values in
        the data set are taken to be the only possible states.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import ExpectationMaximization
    >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
    ...                       columns=['A', 'B', 'C', 'D', 'E'])
    >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
    >>> estimator = ExpectationMaximization(model, data)
    """

    def __init__(self, model, data, **kwargs):
        if not isinstance(model, (DAG, DiscreteBayesianNetwork)):
            raise NotImplementedError(
                "Expectation Maximization is only implemented for DAG or DiscreteBayesianNetwork"
            )

        if isinstance(model, DAG):
            model_bn = DiscreteBayesianNetwork(model.edges())
            model_bn.add_nodes_from(model.nodes())
            model_bn.latents = model.latents
            model = model_bn

        # Drop fully missing columns and treat them as latent if not already
        original_cols = set(data.columns)
        fully_missing_cols = data.columns[data.isna().all()]
        data = data.drop(columns=fully_missing_cols)
        dropped_cols = original_cols - set(data.columns)
        new_latents = [col for col in dropped_cols if col not in model.latents]

        if new_latents:
            logger.warning(
                f"Columns {new_latents} have all missing values and are not marked as latent. "
                "Treating them as latent variables."
            )
            model.latents.update(new_latents)

        # Do NOT drop rows with missing values - the EM algorithm will handle them
        if data.isna().any().any():
            logger.info(
                f"Dataset contains missing values. These will be treated as values to be estimated by EM."
            )

        super(ExpectationMaximization, self).__init__(model, data, **kwargs)
        self.model_copy = self.model.copy()

        # Track which values are missing in the data
        self.missing_mask = data.isna()

    def _get_log_likelihood(self, datapoint):
        """
        Computes the likelihood of a given datapoint. Goes through each
        CPD matching the combination of states to get the value and multiplies
        them together.
        """
        likelihood = 0
        for cpd in self.model_copy.cpds:
            scope = set(cpd.scope())

            # Get relevant datapoint values for this CPD
            scope_values = {
                key: value
                for key, value in datapoint.items()
                if key in scope and not pd.isna(value)
            }

            # Only compute likelihood if we have complete information for this CPD
            if len(scope_values) == len(scope):
                likelihood += log(
                    max(
                        cpd.get_value(**scope_values),
                        1e-10,
                    )
                )
        return likelihood

    def _safe_tuple_key(self, row):
        """Create a safe tuple key for dictionary lookups that handles NaN and categorical values."""
        result = []
        for x in row:
            if pd.isna(x):
                result.append("__NA__")  # Use a string placeholder for NaN
            else:
                result.append(str(x))  # Convert all values to strings
        return tuple(result)

    def _parallel_compute_weights(
        self, data_unique, latent_card, missing_patterns, n_counts, offset, batch_size
    ):
        cache = []
        MAX_MISSING_VARS = (
            8  # Limit the number of missing variables we try to fill at once
        )

        # First, let's check for categorical columns and get their categories
        categorical_columns = {
            col: data_unique[col].cat.categories.tolist()
            for col in data_unique.columns
            if hasattr(data_unique[col], "cat")
        }

        for i in range(offset, min(offset + batch_size, len(missing_patterns))):
            pattern, indices = missing_patterns[i]
            pattern_data = data_unique.iloc[indices]

            # Get the list of variables that need to be filled
            missing_vars = pattern.index[pattern].tolist()
            latent_vars = list(latent_card.keys())

            # Combine both types of missing values
            all_missing_vars = missing_vars + [
                v for v in latent_vars if v not in missing_vars
            ]

            if not all_missing_vars:
                # If nothing is missing for this pattern, just add the original data
                pattern_data["_weight"] = pattern_data.apply(
                    lambda row: n_counts.get(self._safe_tuple_key(row), 1), axis=1
                )
                cache.append(pattern_data)
                continue

            # If we have too many missing variables, only use the first MAX_MISSING_VARS
            # to keep computation tractable
            if len(all_missing_vars) > MAX_MISSING_VARS:
                print(
                    f"Warning: Pattern has {len(all_missing_vars)} missing variables, limiting to {MAX_MISSING_VARS}"
                )
                all_missing_vars = all_missing_vars[:MAX_MISSING_VARS]

            # Create combinations of all possible values for missing variables
            var_cards = {
                v: latent_card.get(v, len(self.state_names.get(v, [0, 1])))
                for v in all_missing_vars
            }
            combinations = list(
                product(*[range(var_cards[v]) for v in all_missing_vars])
            )

            for idx in indices:
                row = data_unique.iloc[
                    idx
                ].copy()  # Make a copy to prevent modifying original
                row_tuple_key = self._safe_tuple_key(row)
                expanded_rows = []

                # Convert to dictionary for manipulation
                row_dict = {}
                for col in row.index:
                    # Handle categorical columns specially
                    if pd.isna(row[col]):
                        row_dict[col] = None
                    else:
                        row_dict[col] = row[col]

                for combo in combinations:
                    new_row_dict = row_dict.copy()
                    for j, var in enumerate(all_missing_vars):
                        # Handle categorical columns properly
                        if var in categorical_columns:
                            # Make sure we're using valid categories from the column
                            cat_value = combo[j] % len(categorical_columns[var])
                            new_row_dict[var] = categorical_columns[var][cat_value]
                        else:
                            new_row_dict[var] = combo[j]

                    # Calculate the likelihood using the dictionary
                    log_likelihood = self._get_log_likelihood(new_row_dict)
                    new_row_dict["_log_likelihood"] = log_likelihood
                    expanded_rows.append(new_row_dict)

                if expanded_rows:
                    # Create DataFrame from row dictionaries to avoid category issues
                    expanded_df = pd.DataFrame(expanded_rows)

                    # Calculate weights based on likelihood
                    likelihoods = np.exp(expanded_df["_log_likelihood"].values)
                    sum_likelihood = likelihoods.sum()
                    if sum_likelihood > 0:
                        expanded_df["_weight"] = (
                            likelihoods / sum_likelihood
                        ) * n_counts.get(row_tuple_key, 1)
                    else:
                        # Handle case where all likelihoods are very small
                        expanded_df["_weight"] = (
                            np.ones(len(likelihoods))
                            / len(likelihoods)
                            * n_counts.get(row_tuple_key, 1)
                        )

                    expanded_df = expanded_df.drop(columns=["_log_likelihood"])
                    cache.append(expanded_df)

        if not cache:
            return pd.DataFrame()

        # Combine all dataframes
        result = pd.concat(cache, ignore_index=True, copy=False)

        # Convert back to categorical if needed
        for col, categories in categorical_columns.items():
            if col in result.columns:
                result[col] = pd.Categorical(result[col], categories=categories)

        return result

    def _compute_weights(self, n_jobs, latent_card, batch_size):
        """
        For each data pattern, creates extra data points for each possible combination
        of states of latent variables and missing values, and assigns weights to each of them.
        """
        data_unique = self.data.drop_duplicates().reset_index(drop=True)

        # Use a more efficient way to count occurrences
        n_counts = {}
        for _, row in self.data.iterrows():
            row_key = self._safe_tuple_key(row)
            n_counts[row_key] = n_counts.get(row_key, 0) + 1

        # Group data by missing patterns manually instead of using groupby
        missing_mask = data_unique.isna()
        pattern_dict = {}

        for idx, row in missing_mask.iterrows():
            # Create a safe pattern tuple that won't cause issues
            pattern_tuple = tuple([bool(x) for x in row])
            if pattern_tuple not in pattern_dict:
                pattern_dict[pattern_tuple] = [idx]
            else:
                pattern_dict[pattern_tuple].append(idx)

        pattern_groups = [
            (missing_mask.iloc[indices[0]], indices)
            for pattern_tuple, indices in pattern_dict.items()
        ]

        # Process in batches
        batch_pattern_groups = [
            pattern_groups[i : i + batch_size]
            for i in range(0, len(pattern_groups), batch_size)
        ]

        if n_jobs > 1:
            cache = Parallel(n_jobs=n_jobs)(
                delayed(self._parallel_compute_weights)(
                    data_unique, latent_card, batch, n_counts, 0, len(batch)
                )
                for batch in batch_pattern_groups
            )
        else:
            # Process sequentially for debugging or if n_jobs=1
            cache = [
                self._parallel_compute_weights(
                    data_unique, latent_card, batch, n_counts, 0, len(batch)
                )
                for batch in batch_pattern_groups
            ]

        return pd.concat(cache, copy=False) if cache else pd.DataFrame()

    def _is_converged(self, new_cpds, atol=1e-08):
        """
        Checks if the values of `new_cpds` is within tolerance limits of current
        model cpds.
        """
        for cpd in new_cpds:
            if not cpd.__eq__(self.model_copy.get_cpds(node=cpd.scope()[0]), atol=atol):
                return False
        return True

    def get_parameters(
        self,
        latent_card=None,
        max_iter=100,
        atol=1e-08,
        n_jobs=1,
        batch_size=1000,
        seed=None,
        init_cpds={},
        show_progress=True,
    ):
        """
        Method to estimate all model parameters (CPDs) using Expecation Maximization.

        Parameters
        ----------
        latent_card: dict (default: None)
            A dictionary of the form {latent_var: cardinality} specifying the
            cardinality (number of states) of each latent variable. If None,
            assumes `2` states for each latent variable.

        max_iter: int (default: 100)
            The maximum number of iterations the algorithm is allowed to run for.
            If max_iter is reached, return the last value of parameters.

        atol: int (default: 1e-08)
            The absolute accepted tolerance for checking convergence. If the parameters
            change is less than atol in an iteration, the algorithm will exit.

        n_jobs: int (default: 1)
            Number of jobs to run in parallel.
            Using n_jobs > 1 for small models or datasets might be slower.

        batch_size: int (default: 1000)
            Number of data used to compute weights in a batch.

        seed: int
            The random seed to use for generating the intial values.

        init_cpds: dict or str
            dict: A dictionary of the form {variable: instance of TabularCPD}
            specifying the initial CPD values for the EM optimizer to start
            with. If not specified, CPDs involving latent variables are
            initialized randomly, and CPDs involving only observed variables are
            initialized with their MLE estimates.

            str: `uniform`, all CPDs will be initialized to have a uniform distribution.
                 `random`, all CPDs will be initialized randomly.

        show_progress: boolean (default: True)
            Whether to show a progress bar for iterations.

        Returns
        -------
        Estimated paramters (CPDs): list
            A list of estimated CPDs for the model.

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import ExpectationMaximization as EM
        >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 3)),
        ...                       columns=['A', 'C', 'D'])
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D')], latents={'B'})
        >>> estimator = EM(model, data)
        >>> estimator.get_parameters(latent_card={'B': 3})
        [<TabularCPD representing P(C:2) at 0x7f7b534251d0>,
        <TabularCPD representing P(B:3 | C:2, A:2) at 0x7f7b4dfd4da0>,
        <TabularCPD representing P(A:2) at 0x7f7b4dfd4fd0>,
        <TabularCPD representing P(D:2 | C:2) at 0x7f7b4df822b0>]
        """
        # Step 1: Parameter checks
        if latent_card is None:
            latent_card = {var: 2 for var in self.model_copy.latents}

        # Add cardinality for variables with missing values
        for col in self.data.columns:
            if (
                col not in latent_card
                and col not in self.model_copy.latents
                and self.missing_mask[col].any()
            ):
                if col in self.state_names:
                    latent_card[col] = len(self.state_names[col])
                else:
                    # Default to 2 states if not specified
                    latent_card[col] = 2

        # Step 2: Create structures/variables to be used later.
        n_states_dict = {key: len(value) for key, value in self.state_names.items()}
        n_states_dict.update(latent_card)
        for var in self.model_copy.latents:
            self.state_names[var] = list(range(n_states_dict[var]))

        # Step 3: Initialize CPDs.
        # Step 3.0: Check if init_cpds is a string and if so, initialize the CPDs.
        if isinstance(init_cpds, str):
            parents_dict = {
                var: self.model.get_parents(var) for var in self.model.nodes()
            }
            if init_cpds == "random":
                init_cpds = {
                    var: TabularCPD.get_random(
                        variable=var,
                        evidence=parents_dict[var],
                        cardinality={
                            v: n_states_dict[v] for v in ([var] + parents_dict[var])
                        },
                        state_names={
                            v: self.state_names[v] for v in ([var] + parents_dict[var])
                        },
                        seed=seed,
                    )
                    for var in self.model.nodes()
                }
            elif init_cpds == "uniform":
                init_cpds = {
                    var: TabularCPD.get_uniform(
                        variable=var,
                        evidence=parents_dict[var],
                        cardinality={
                            v: n_states_dict[v] for v in ([var] + parents_dict[var])
                        },
                        state_names={
                            v: self.state_names[v] for v in ([var] + parents_dict[var])
                        },
                        seed=seed,
                    )
                    for var in self.model.nodes()
                }
            else:
                raise ValueError(
                    f"If `init_cpds` is a string, it must be either 'random' or 'uniform'. Got: {init_cpds}"
                )

        # Step 3.1: Learn the CPDs of variables which don't involve
        #           latent variables using MLE if their init_cpd is
        #           not specified.
        fixed_cpds = []
        fixed_cpd_vars = (
            set(self.model.nodes())
            - self.model.latents
            - set(chain(*[self.model.get_children(var) for var in self.model.latents]))
            - set(init_cpds.keys())
        )

        mle = MaximumLikelihoodEstimator.__new__(MaximumLikelihoodEstimator)
        mle.model = self.model
        mle.data = self.data
        mle.state_names = self.state_names

        for var in fixed_cpd_vars:
            fixed_cpds.append(mle.estimate_cpd(var))

        # Step 3.2: Randomly initialize the CPDs involving latent variables if init_cpds is not specified.
        latent_cpds = []
        vars_with_latents = (
            set(self.model_copy.nodes()) - fixed_cpd_vars - set(init_cpds.keys())
        )
        for node in vars_with_latents:
            parents = list(self.model_copy.predecessors(node))
            latent_cpds.append(
                TabularCPD.get_random(
                    variable=node,
                    evidence=parents,
                    cardinality={
                        var: n_states_dict[var] for var in chain([node], parents)
                    },
                    state_names={
                        var: self.state_names[var] for var in chain([node], parents)
                    },
                    seed=seed,
                )
            )

        self.model_copy.add_cpds(
            *list(chain(fixed_cpds, latent_cpds, list(init_cpds.values())))
        )

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_iter)

        mle.model = self.model_copy
        # Step 4: Run the EM algorithm.
        for _ in range(max_iter):
            # Step 4.1: E-step: Expands the dataset and computes the likelihood of each
            #           possible state of latent variables.
            weighted_data = self._compute_weights(n_jobs, latent_card, batch_size)
            # Step 4.2: M-step: Uses the weights of the dataset to do a weighted MLE.
            new_cpds = fixed_cpds.copy()
            mle.data = weighted_data
            for var in vars_with_latents.union(set(init_cpds.keys())):
                new_cpds.append(mle.estimate_cpd(var, weighted=True))

            # Step 4.3: Check of convergence and max_iter
            if self._is_converged(new_cpds, atol=atol):
                if show_progress and config.SHOW_PROGRESS:
                    pbar.close()
                return new_cpds

            else:
                self.model_copy.cpds = new_cpds
                if show_progress and config.SHOW_PROGRESS:
                    pbar.update(1)

        return new_cpds
