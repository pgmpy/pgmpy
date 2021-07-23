import warnings
from itertools import product, chain

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.global_vars import SHOW_PROGRESS


class ExpectationMaximization(ParameterEstimator):
    def __init__(self, model, data, **kwargs):
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
            cells should be set to `numpy.NaN`.  Note that pandas converts each
            column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values in
            the data set are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to
            `True` all rows that contain `np.NaN` somewhere are ignored. If
            `False` then, for each variable, every row where neither the
            variable nor its parents are `np.NaN` is used.

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
        if not isinstance(model, BayesianNetwork):
            raise NotImplementedError(
                "Expectation Maximization is only implemented for BayesianNetwork"
            )

        super(ExpectationMaximization, self).__init__(model, data, **kwargs)
        self.model_copy = self.model.copy()

    def _get_likelihood(self, datapoint):
        """
        Computes the likelihood of a given datapoint. Goes through each
        CPD matching the combination of states to get the value and multiplies
        them together.
        """
        likelihood = 1
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for cpd in self.model_copy.cpds:
                scope = set(cpd.scope())
                likelihood *= cpd.get_value(
                    **{key: value for key, value in datapoint.items() if key in scope}
                )
        return likelihood

    def _compute_weights(self, latent_card):
        """
        For each data point, creates extra data points for each possible combination
        of states of latent variables and assigns weights to each of them.
        """
        cache = []

        data_unique = self.data.drop_duplicates()
        n_counts = self.data.groupby(list(self.data.columns)).size().to_dict()

        for i in range(data_unique.shape[0]):
            v = list(product(*[range(card) for card in latent_card.values()]))
            latent_combinations = np.array(v, dtype=int)
            df = data_unique.iloc[[i] * latent_combinations.shape[0]].reset_index(
                drop=True
            )
            for index, latent_var in enumerate(latent_card.keys()):
                df[latent_var] = latent_combinations[:, index]

            weights = df.apply(lambda t: self._get_likelihood(dict(t)), axis=1)
            df["_weight"] = (weights / weights.sum()) * n_counts[
                tuple(data_unique.iloc[i])
            ]
            cache.append(df)

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
        n_jobs=-1,
        seed=None,
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

        n_jobs: int (default: -1)
            Number of jobs to run in parallel. Default: -1 uses all the processors.

        seed: int
            The random seed to use for generating the intial values.

        show_progress: boolean (default: True)
            Whether to show a progress bar for iterations.

        Returns
        -------
        list: A list of estimated CPDs for the model.

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

        # Step 3: Initialize random CPDs if starting values aren't provided.
        if seed is not None:
            np.random.seed(seed)

        cpds = []
        for node in self.model_copy.nodes():
            parents = list(self.model_copy.predecessors(node))
            cpds.append(
                TabularCPD.get_random(
                    variable=node,
                    evidence=parents,
                    cardinality={
                        var: n_states_dict[var] for var in chain([node], parents)
                    },
                    state_names={
                        var: self.state_names[var] for var in chain([node], parents)
                    },
                )
            )

        self.model_copy.add_cpds(*cpds)

        if show_progress and SHOW_PROGRESS:
            pbar = tqdm(total=max_iter)

        # Step 4: Run the EM algorithm.
        for _ in range(max_iter):
            # Step 4.1: E-step: Expands the dataset and computes the likelihood of each
            #           possible state of latent variables.
            weighted_data = self._compute_weights(latent_card)
            # Step 4.2: M-step: Uses the weights of the dataset to do a weighted MLE.
            new_cpds = MaximumLikelihoodEstimator(
                self.model_copy, weighted_data
            ).get_parameters(n_jobs=n_jobs, weighted=True)

            # Step 4.3: Check of convergence and max_iter
            if self._is_converged(new_cpds, atol=atol):
                if show_progress and SHOW_PROGRESS:
                    pbar.close()
                return new_cpds

            else:
                self.model_copy.cpds = new_cpds
                if show_progress and SHOW_PROGRESS:
                    pbar.update(1)

        return cpds
