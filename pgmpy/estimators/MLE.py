# coding:utf-8

from pgmpy.estimators import BaseEstimator
from pgmpy.factors import TabularCPD
from pgmpy.models import BayesianModel
import numpy as np
import pandas as pd


class MaximumLikelihoodEstimator(BaseEstimator):
    """
    Class used to compute parameters for a model using Maximum Likelihood Estimation.

    Parameters
    ----------
    model: A pgmpy.models.BayesianModel instance

    data: pandas DataFrame object
        DataFrame object with column names identical to the variable names of the network

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.models import BayesianModel
    >>> from pgmpy.estimators import MaximumLikelihoodEstimator
    >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
    ...                       columns=['A', 'B', 'C', 'D', 'E'])
    >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
    >>> estimator = MaximumLikelihoodEstimator(model, data)
    """
    def __init__(self, model, data, node_values=None):
        if not isinstance(model, BayesianModel):
            raise NotImplementedError("Maximum Likelihood Estimate is only implemented for BayesianModel")

        super(MaximumLikelihoodEstimator, self).__init__(model, data, node_values)

    def get_parameters(self):
        """
        Method used to get parameters.

        Returns
        -------
        parameters: list
            List of TabularCPDs, one for each variable of the model

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
        ...                       columns=['A', 'B', 'C', 'D'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'))
        >>> estimator = MaximumLikelihoodEstimator(model, values)
        >>> estimator.get_parameters()
        [<TabularCPD representing P(C:2) at 0x7f7b534251d0>,
        <TabularCPD representing P(B:2 | C:2, A:2) at 0x7f7b4dfd4da0>,
        <TabularCPD representing P(A:2) at 0x7f7b4dfd4fd0>,
        <TabularCPD representing P(D:2 | C:2) at 0x7f7b4df822b0>]
        """
        parameters = []

        for node in self.model.nodes():
            cpd = self._estimate_cpd(node)
            parameters.append(cpd)

        return parameters

    def _estimate_cpd(self, node):
        """
        Method to estimate the CPD for a given variable.

        Parameters
        ----------
        node: int, string (any hashable python object)
            The name of the variable for which the CPD is to be estimated.

        Returns
        -------
        CPD: TabularCPD

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = BayesianModel([('A', 'C'), ('B', 'C')])
        >>> cpd_A = MaximumLikelihoodEstimator(model, data)._estimate_cpd('A')
        >>> print(str(cpd_A))
        ╒═════╤══════════╕
        │ A_0 │ 0.666667 │
        ├─────┼──────────┤
        │ A_1 │ 0.333333 │
        ╘═════╧══════════╛
        """

        parents = self.model.get_parents(node)
        if not parents:
            state_counts = self.data.ix[:, node].value_counts()
            state_counts = state_counts.reindex(sorted(state_counts.index))
            cpd = TabularCPD(node, len(self.node_values[node]),
                             state_counts.values[:, np.newaxis])
        else:
            parent_cardinalities = np.array([len(self.node_values[parent]) for parent in parents])
            node_cardinality = len(self.node_values[node])

            values = self.data.groupby([node] + parents).size().unstack(parents).fillna(0)
            if not len(values.columns) == np.prod(parent_cardinalities):
                # some columns are missing if for some states of the parents no data was observed.
                # reindex to add missing columns and fill in uniform (conditional) probabilities:
                full_index = pd.MultiIndex.from_product([range(card) for card in parent_cardinalities], names=parents)
                values = values.reindex(columns=full_index).fillna(1.0/node_cardinality)

            cpd = TabularCPD(node, node_cardinality, np.array(values),
                             evidence=parents,
                             evidence_card=parent_cardinalities.astype('int'))
        cpd.normalize()
        return cpd
