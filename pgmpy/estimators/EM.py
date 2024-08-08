from itertools import chain, product
from math import log

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.estimators import MaximumLikelihoodEstimator, ParameterEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianNetwork


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
    model: A pgmpy.models.BayesianNetwork instance

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
    >>> from pgmpy.models import BayesianNetwork
    >>> from pgmpy.estimators import ExpectationMaximization
    >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
    ...                       columns=['A', 'B', 'C', 'D', 'E'])
    >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
    >>> estimator = ExpectationMaximization(model, data)
    """

    def __init__(self, model, data, **kwargs):
        if not isinstance(model, BayesianNetwork):
            raise NotImplementedError(
                "Expectation Maximization is only implemented for BayesianNetwork"
            )

        super(ExpectationMaximization, self).__init__(model, data, **kwargs)
        self.model_copy = self.model.copy()

    def _get_log_likelihood(self, datapoint):
        """
        Computes the likelihood of a given datapoint. Goes through each
        CPD matching the combination of states to get the value and multiplies
        them together.
        """
        likelihood = 0
        for cpd in self.model_copy.cpds:
            scope = set(cpd.scope())
            likelihood += log(
                max(
                    cpd.get_value(
                        **{
                            key: value
                            for key, value in datapoint.items()
                            if key in scope
                        }
                    ),
                    1e-10,
                )
            )
        return likelihood

    def _parallel_compute_weights(
        self, data_unique, latent_card, n_counts, offset, batch_size
    ):
        cache = []

        for i in range(offset, min(offset + batch_size, data_unique.shape[0])):
            v = list(product(*[range(card) for card in latent_card.values()]))
            latent_combinations = np.array(v, dtype=int)
            df = data_unique.iloc[[i] * latent_combinations.shape[0]].reset_index(
                drop=True
            )
            for index, latent_var in enumerate(latent_card.keys()):
                df[latent_var] = latent_combinations[:, index]
            weights = np.e ** (
                df.apply(lambda t: self._get_log_likelihood(dict(t)), axis=1)
            )
            df["_weight"] = (weights / weights.sum()) * n_counts[
                tuple(data_unique.iloc[i])
            ]
            cache.append(df)

        return pd.concat(cache, copy=False)

    def _compute_weights(self, n_jobs, latent_card, batch_size):
        """
        For each data point, creates extra data points for each possible combination
        of states of latent variables and assigns weights to each of them.
        """

        data_unique = self.data.drop_duplicates()
        n_counts = self.data.groupby(list(self.data.columns)).size().to_dict()

        cache = Parallel(n_jobs=n_jobs)(
            delayed(self._parallel_compute_weights)(
                data_unique, latent_card, n_counts, i, batch_size
            )
            for i in range(0, data_unique.shape[0], batch_size)
        )

        return pd.concat(cache, copy=False)

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

        init_cpds: dict
            A dictionary of the form {variable: instance of TabularCPD}
            specifying the initial CPD values for the EM optimizer to start
            with. If not specified, CPDs involving latent variables are
            initialized randomly, and CPDs involving only observed variables are
            initialized with their MLE estimates.

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
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.estimators import ExpectationMaximization as EM
        >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 3)),
        ...                       columns=['A', 'C', 'D'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D')], latents={'B'})
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

        # Step 2: Create structures/variables to be used later.
        n_states_dict = {key: len(value) for key, value in self.state_names.items()}
        n_states_dict.update(latent_card)
        for var in self.model_copy.latents:
            self.state_names[var] = list(range(n_states_dict[var]))

        # Step 3: Initialize CPDs.
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
