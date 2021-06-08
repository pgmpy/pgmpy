# coding:utf-8

import numpy as np
import pandas as pd

from pgmpy.estimators import ParameterEstimator
from pgmpy.factors.discrete import TabularCPD, DiscreteFactor
from pgmpy.models import BayesianModel

def moveaxis(in_array,in_labels, out_labels):
    """
    a helper function to move the axis of an array based on two lists of labels
     
    Parameters
    ----------
    in_array: the input np.ndarray

    in_labels: the list of labels that describe the axis of in_array

    out_labels: the reordering of the labels that is desired

    Returns
    -------
    out_array: in_array with its axes moved to correspond to out_labels
    """
    in_idx = range(len(in_labels))
    out_idx = [in_labels.index(label) for label in out_labels]
    return np.moveaxis(in_array,out_idx,in_idx)

class ExpectationMaximization(ParameterEstimator):
    def __init__(self, model, data, **kwargs):
        """
        Class used to compute parameters for a model using Expectation
        Maximization


        Parameters
        ----------
        model: A pgmpy.models.BayesianModel instance
                note that the prior CPDs must be specified in this model, this
                is because the prior CPDs are used along with a
                VariableElimination class to compute the Expectation Step for
                the missing data instances

        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names
            of the network.
            (If some values in the data are missing the data cells should be
                set to `numpy.NaN`. Note that pandas converts each column
                containing `numpy.NaN`s to dtype `float`.)

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

    def get_parameters(self, n_iter=1):
        """
        Method to estimate teh model parameters (CPDs) using Expectation
        Maximization.

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

        # to avoid circular imports lets import ve here
        from pgmpy.inference import VariableElimination

        # this will be very similar to MLE estimation, except we will add
        #  partial counts to the state_counts field

        # get the parents, state_names, etc for the node
        parents = sorted(self.model.get_parents(node))

        state_names = self.model.get_cpds(node).state_names
        state_names = {k: v for k, v in state_names.items() if k in [node] + parents}

        node_cardinality = len(state_names[node])

        parents_cardinalities = [len(state_names[parent]) for parent in parents]
        node_cardinality = len(state_names[node])

        variable_names = sorted(state_names.keys())

        # get counts of the cases where the node and all parents are observed
        complete_data = self.data.dropna(subset=variable_names)

        # count the partial cases based on ve inference for the cases where
        #  either the node or a parent or both are missing
        missing_data = self.data.drop(
            index=self.data.dropna(subset=variable_names).index
        )

        state_counts = complete_data.groupby(variable_names).size()

        # if there are no parents, the groupby index will not be a multi index,
        if not parents:
            tmp_idx = pd.MultiIndex.from_product(
                iterables=[state_counts.index.values], 
                names=state_counts.index.names
            )
            state_counts.index = tmp_idx

        factor_idx = pd.MultiIndex.from_product(
            iterables=[state_names[n] for n in variable_names],
            names=state_counts.index.names,
        )

        factor = state_counts.reindex(factor_idx).fillna(0)

        # we don't necessarily want to update the estimators model, lets leave
        #  that to the user to do once returning the cpds
        model = self.model.copy()

        # for n_iter iterations calculate the update
        for i_iter in range(n_iter):

            # we want to start with a new factor and Variable Elimination
            #  object for every iteration
            iter_factor = factor.copy()
            ve = VariableElimination(model)

            # for every sample that includes missing data for this cpd,
            #  calculate the factor with the known data
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
                    variables=variables,
                    evidence=evidence,
                    show_progress=False,
                )

                # we need to marginalize over variables that are possibly
                #  unobserved but nevertheless not in [node] + parents
                marg_list = [v for v in q_factor.variables if v not in variable_names]

                if marg_list:
                    q_factor.marginalize(marg_list)

                # find teh variables that are in variable_names (part of this
                #  factor) but also in evidence and hence observed
                evidence_factor_variables = [
                    v for v in variable_names if v in evidence.keys()
                ]

                if evidence_factor_variables:
                    # because the variables in evidence factor are all observed
                    #  their cardinality is 1
                    evidence_factor_cardinality = [1] * len(evidence_factor_variables)

                    # now "fake" the statenames by just grabbing them from the
                    #  evidence
                    evidence_factor_state_names = {
                        v: [evidence[v]] for v in evidence_factor_variables
                    }

                    # the values should just be [1] because all of the
                    #  cardinalities are 1 but lets do the multiplication just
                    #  for completeness
                    evidence_factor_values = [1] * np.prod(evidence_factor_cardinality)

                    # now we have a factor that only has unobserved variables
                    #  for this sample that are in [node] + parents but if
                    #  there was a node that was observed and is in [node] +
                    #  parents it wont be in this factor so now we need to
                    #  create an evidence factor
                    evidence_factor = DiscreteFactor(
                        variables=evidence_factor_variables,
                        cardinality=evidence_factor_cardinality,
                        state_names=evidence_factor_state_names,
                        values=evidence_factor_values,
                    )

                    # now get the product of q_factor and evidence_factor, this
                    #  is done inplace
                    q_factor.product(evidence_factor)

                # now we need to get this factor as a series with the same
                #  multiindex ordering as factor and sum the two
                q_factor_idx = pd.MultiIndex.from_product(
                    iterables=[q_factor.state_names[n] for n in variable_names],
                    names=variable_names,
                )

                # now get the data to create this index, we will have to swap
                #  axes before the flatten
                q_factor_series = pd.Series(
                    data=moveaxis(
                        q_factor.values,
                        q_factor.variables,
                        q_factor_idx.names,
                    ).flatten(),
                    index=q_factor_idx,
                )

                # now reindex to get the q_factor_series with the same index as
                #  iter_factor
                q_factor_series = q_factor_series.reindex(factor_idx).fillna(0)

                iter_factor = iter_factor + q_factor_series

            cpd_array = np.array(iter_factor)
            # alright enough messing around with this pandas bs, we have a 1d
            #  ndarray that we want to get into shape (card(node),
            #  prod(card(parents))) and, and, and, preserve the order of the
            #  variables

            # lets get this 1d array into a <num_variables>-d array
            cpd_array = cpd_array.reshape([len(state_names[n]) for n in variable_names])

            # now we want the node to be the 0th dimension, and the parents to
            #  be lexically sorted after that
            cpd_array = moveaxis(
                cpd_array,  # original array
                variable_names,  # sorted variable names
                [node] + sorted(parents),  # desired ordering of variable names
            )

            # now we are going to reshape it so the rowas are the states of the
            # node and the columns are the unique states of the parents
            cpd_array = cpd_array.reshape(
                [
                    len(state_names[node]),
                    int(np.prod([len(state_names[p]) for p in parents])),
                ]
            )

            # now we need to decide what to do with the columns that don't have
            # any counts, two options are
            #    1) set to uniform distribution
            #    2) set to prior
            # lets choose option 1) for now since it is easier to intert into
            #  numpy however, we are slinging around the priors so much that
            #  may be a good thing to use here as well

            cpd_array[:,np.sum(cpd_array, axis=0) == 0] = 1

            cpd = TabularCPD(
                node,
                node_cardinality,
                cpd_array,
                evidence=parents,
                evidence_card=parents_cardinalities,
                state_names=state_names,
            )

            old_cpd = model.get_cpds(node)
            model.remove_cpds(old_cpd)

            cpd.normalize()
            model.add_cpds(cpd)

        return cpd
