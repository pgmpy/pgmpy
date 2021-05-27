# coding:utf-8

import copy
import numpy as np
import pandas as pd

from pgmpy.estimators import ParameterEstimator
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel


class ExpectationMaximization(ParameterEstimator):
    def __init__(self, model, data, **kwargs):
        """
        Class used to compute parameters for a model using Expectation Maximization


        Parameters
        ----------
        model: A pgmpy.models.BayesianModel instance
                note that the prior CPDs must be specified in this model, this is
                because the prior CPDs are used along with a VariableElimination class
                to compute the Expectation Step for the missing data instances

        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the
                network.
            (If some values in the data are missing the data cells should be set to
                `numpy.NaN`. Note that pandas converts each column containing
                `numpy.NaN`s to dtype `float`.)

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import (ExpectationMaximization,
        >>>                               MaximumLikelihoodEstimator)
        >>> data = pd.DataFrame(np.random.randint(0,2,(1000,4)),
        >>>                     columns=['A','B','C','D'])
        >>> model = BayesianModel([('A','C'),('B','C'),('C','D')])
        >>> estimator = MaximumLikelihoodEstimator(model,data)
        >>> model.add_cpds(*estimator.get_parameters())

        # now we will generate new data set some values to missing and then run
        #   inference
        >>> data = pd.DataFrame(np.random.randint(0,2,(1000,4)),
        >>>                     columns=['A','B','C','D'])
        >>>
        >>> i = np.random.randint(0,2,data.shape)
        >>> i = i.astype(bool)
        >>> data[i] = np.nan
        >>> em_estimator = ExpectationMaximization(model,data)
        """
        if not isinstance(model, BayesianModel):
            raise NotImplementedError(
                "Maximum Likelihood Estimate is only implemented for BayesianModel"
            )

        model.check_model()

        super(ExpectationMaximization, self).__init__(model, data, **kwargs)

    def get_parameters(self,n_iter=1):
        """
        Method to estimate teh model parameters (CPDs) using Expectation Maximization.

        Parameters
        ----------
        n_iter: int, the number of iterations to perform the EM algorithm

        Returns
        -------
        parameters: list
            List of TabularCPDs one for each variable in the model

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import (ExpectationMaximization,
        >>>                               MaximumLikelihoodEstimator)
        >>> data = pd.DataFrame(np.random.randint(0,2,(1000,4)),
        >>>                     columns=['A','B','C','D'])
        >>> model = BayesianModel([('A','C'),('B','C'),('C','D')])
        >>> estimator = MaximumLikelihoodEstimator(model,data)
        >>> model.add_cpds(*estimator.get_parameters())

        # now we will generate new data set some values to missing and then run
        #   inference
        >>> data = pd.DataFrame(np.random.randint(0,2,(1000,4)),
        >>>                     columns=['A','B','C','D'])
        >>>
        >>> i = np.random.randint(0,2,data.shape)
        >>> i = i.astype(bool)
        >>> data[i] = np.nan
        >>> em_estimator = ExpectationMaximization(model,data)

        >>> em_estimator.get_parameters()
        [<TabularCPD representing P(A:2) at 0x7f76b14ebeb8>,
         <TabularCPD representing P(B:2) at 0x7f76b11f9a58>,
         <TabularCPD representing P(C:2 | A:2, B:2) at 0x7f76b11f9cf8>,
         <TabularCPD representing P(D:2 | C:2) at 0x7f76b11f9a90>]
        """
        parameters = []

        for node in sorted(self.model.nodes()):
            cpd = self.estimate_cpd(node, n_iter)
            parameters.append(cpd)

        return parameters

    def estimate_cpd(self, node, n_iter=1):
        """
        Method to estimate the CPD for a given variable.

        Parameters
        ----------
        node: int, string (any hashable python object)
            The name of the variable for which the CPD is to be estimated.

        n_iter: int, the number of iterations to perform the EM algorithm

        Returns
        -------
        CPD: TabularCPD

        Examples
        --------
        """

        # this will be very similar to MLE estimation, except we will add partial
        #   counts to the state_counts field

        # get the parents, state_names, etc for the node
        parents = sorted(self.model.get_parents(node))

        state_names = self.model.get_cpds(node).state_names

        parents_cardinalities = [len(state_names[parent]) for parent in parents]
        node_cardinality = len(state_names[node])

        # get counts of the cases where the node and all parents are observed
        complete_data = self.data.dropna(subset=[node] + parents)

        state_counts_data = complete_data.groupby([node] + parents).size()

        if not isinstance(state_counts_data.index, pd.MultiIndex):
            state_counts_data.index = pd.MultiIndex.from_arrays(
                [state_counts_data.index]
            )

        row_index = pd.MultiIndex.from_product(
            iterables=[v for v in state_names.values()],
            names=[k for k in state_names.keys()],
        )

        factor = state_counts_data.reindex(index=row_index).fillna(0)

        # count the partial cases based on ve inference for the cases where either the
        #   node or a parent or both are missing
        missing_data = self.data.drop(
            index=self.data.dropna(subset=[node] + parents).index
        )

        # we don't necessarily want to update the estimators model, lets leave that to
        #  the user to do once returning the cpds
        model = copy.deepcopy(self.model)

        # for n_iter iterations calculate the update
        for i_iter in range(n_iter):

            # we want to start with a new factor and Variable Elimination object for
            #   every iteration
            iter_factor = copy.deepcopy(factor)
            ve = VariableElimination(model)

            # for every sample that includes missing data for this cpd, calculate
            #   the factor with the known data
            for isamp, sample in missing_data.iterrows():
                # we need to separate the variables into evidence (observed) and
                #   variables (unobserved) to run VE
                variables = []
                evidence = dict(sample)

                for key in evidence.keys():
                    if evidence[key] != evidence[key]:
                        variables.append(key)
                for var in variables:
                    evidence.pop(var)

                # run VE on this sample
                q_factor = ve.query(
                    variables=variables, evidence=evidence, show_progress=False
                )

                # we need to marginalize over variables that are possibly unobserved
                #   but nevertheless not in [node] + parents
                marg_list = [v for v in q_factor.variables if v not in [node] + parents]

                if marg_list:
                    q_factor.marginalize(marg_list)

                # now we want to reindex this returned factor so that we can directly
                #   add it to factor and then marginalize
                idx_dict = q_factor.state_names
                idx_dict.update(evidence)

                iterables = [
                    idx_dict[v] if isinstance(idx_dict[v], list) else [idx_dict[v]]
                    for v in iter_factor.index.names
                ]
                names = list(iter_factor.index.names)

                qfs = pd.Series(
                    data=q_factor.values.flatten(),
                    index=pd.MultiIndex.from_product(iterables, names=names),
                )

                qfs = qfs.reindex(iter_factor.index).fillna(0)

                iter_factor = iter_factor + qfs

            # right now iter_factor is a pd.Series with all variables in the multiindex
            #   we want to unstack the parents to format it into a 2d factor to update
            #   the CPD
            iter_factor = iter_factor.unstack(parents)

            if not isinstance(iter_factor, pd.DataFrame):
                iter_factor = iter_factor.to_frame()

            iter_factor.loc[:, (iter_factor == 0).all()] = 1

            # reindex the factor so that for node the state_names are in the same order
            #   as state_names
            if not isinstance(iter_factor.index, pd.MultiIndex):
                iter_factor.index = pd.MultiIndex.from_arrays([iter_factor.index])


            iter_factor = iter_factor.reindex(
                pd.MultiIndex.from_arrays([state_names[node]], names=[node])
            )

            if parents:
                if not isinstance(iter_factor.columns, pd.MultiIndex):
                    iter_factor.columns = pd.MultiIndex.from_arrays(
                        [iter_factor.columns]
                    )

                column_index = pd.MultiIndex.from_product(
                    iterables=[state_names[p] for p in parents], names=parents
                )

                iter_factor = iter_factor.reindex(columns=column_index)

            cpd = TabularCPD(
                node,
                node_cardinality,
                np.array(iter_factor),
                evidence=parents,
                evidence_card=parents_cardinalities,
                state_names=state_names,
            )

            cpd.normalize()
            model.add_cpds(cpd)

        return cpd
