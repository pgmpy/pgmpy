from itertools import product, chain

import numpy as np
import pandas as pd

from pgmpy.estimators import ParameterEstimator, MaximumLikelihoodEstimator
from pgmpy.models import BayesianModel
from pgmpy.factors.discrete import TabularCPD


class ExpectationMaximization(ParameterEstimator):
    def __init__(self, model, data, **kwargs):
        """
        Class used to compute parameters for a model using Expectation
        Maximization (EM).  EM is an iterative algorithm commonly used for
        estimation in the case when there are latent variables in the model.
        The algorithm iteratively improves the parameter estimates maximizing
        the likelihood of the given data.

        Parameters
        ----------
        model: A pgmpy.models.BayesianModel instance

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
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import ExpectationMaximization
        >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> estimator = ExpectationMaximization(model, data)
        """
        if not isinstance(model, BayesianModel):
            raise NotImplementedError(
                "Expectation Maximization is only implemented for BayesianModel"
            )

        super(ExpectationMaximization, self).__init__(model, data, **kwargs)

    def _get_likelihood(self, datapoint):
        """
        Computes the likelihood of a given datapoint. Goes through each
        CPD matching the combination of states to get the value and multiplies
        them together.
        """
        likelihood = 1
        for cpd in self.model.cpds:
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
        cache = {}
        for i in range(0, self.data.shape[0]):
            if tuple(self.data.iloc[i]) not in cache.keys():
                v = list(product(*[range(card) for card in latent_card.values()]))
                latent_combinations = np.array(v, dtype=int)
                df = self.data.iloc[[i] * latent_combinations.shape[0]].reset_index(
                    drop=True
                )
                for index, latent_var in enumerate(latent_card.keys()):
                    df[latent_var] = latent_combinations[:, index]

                weights = df.apply(lambda t: self._get_likelihood(dict(t)), axis=1)
                df["_weight"] = weights / weights.sum()

                cache[tuple(self.data.iloc[i])] = df

        return pd.concat(
            [cache[tuple(self.data.iloc[i])] for i in range(self.data.shape[0])]
        )

    def _is_converged(self, new_cpds, atol=1e-08):
        """
        Checks if the values of `new_cpds` is within tolerance limits of current
        model cpds.
        """
        for cpd in new_cpds:
            if not cpd.__eq__(self.model.get_cpds(node=cpd.scope()[0]), atol=atol):
                return False
        return True

    def get_parameters(self, latent_card=None, n_jobs=-1, max_iter=100, atol=1e-08):
        """
        Method to estimate all model parameters (CPDs) using Expecation Maximization.

        Parameters
        ----------
        latent_card: dict (default: None)
            A dictionary of the form {latent_var: cardinality} specifying the
            cardinality (number of states) of each latent variable. If None,
            assumes `2` states for each latent variable.

        n_jobs: int (default: -1)
            Number of jobs to run in parallel. Default: -1 uses all the processors.

        Returns
        -------
        parameters: list
            List of TabularCPDs, one for each variable of the model

        n_jobs: int
            Number of processes to spawn

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import ExpectationMaximization as EM
        >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 3)),
        ...                       columns=['A', 'C', 'D'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D')], latents={'B'})
        >>> estimator = EM(model, data)
        >>> estimator.get_parameters(latent_card={'B': 3})
        [<TabularCPD representing P(C:2) at 0x7f7b534251d0>,
        <TabularCPD representing P(B:3 | C:2, A:2) at 0x7f7b4dfd4da0>,
        <TabularCPD representing P(A:2) at 0x7f7b4dfd4fd0>,
        <TabularCPD representing P(D:2 | C:2) at 0x7f7b4df822b0>]
        """
        if latent_card is None:
            latent_card = {var: 2 for var in self.model.latents}

        n_states_dict = {key: len(value) for key, value in self.state_names.items()}
        n_states_dict.update(latent_card)
        for var in self.model.latents:
            self.state_names[var] = list(range(n_states_dict[var]))

        cpds = []
        for node in self.model.nodes():
            parents = list(self.model.predecessors(node))
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

        self.model.add_cpds(*cpds)

        for i in range(max_iter):
            # Expectation Step: Computes the likelihood of each data point.
            weighted_data = self._compute_weights(latent_card)
            # Maximization Step: Uses the weights of the dataset for estimation.
            new_cpds = MaximumLikelihoodEstimator(
                self.model, weighted_data
            ).get_parameters(weighted=True)
            if self._is_converged(new_cpds):
                return new_cpds
            else:
                self.model.add_cpds(*new_cpds)
