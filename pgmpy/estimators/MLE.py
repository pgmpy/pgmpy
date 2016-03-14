from pgmpy.estimators import BaseEstimator
from pgmpy.factors import TabularCPD
from pgmpy.models import BayesianModel
import numpy as np
import pandas as pd


class MaximumLikelihoodEstimator(BaseEstimator):
    """
    Class used to compute parameters for a model using Maximum Likelihood Estimate.

    Parameters
    ----------
    model: A pgmpy.models.BayesianModel instance

    data: pandas DataFrame object
        datafame object with column names same as the variable names of the network

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.models import BayesianModel
    >>> from pgmpy.estimators import MaximumLikelihoodEstimator
    >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
    ...                       columns=['A', 'B', 'C', 'D', 'E'])
    >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
    >>> estimator = MaximumLikelihoodEstimator(model, values)
    """
    def __init__(self, model, data):
        if not isinstance(model, BayesianModel):
            raise NotImplementedError("Maximum Likelihood Estimate is only implemented for BayesianModel")

        super(MaximumLikelihoodEstimator, self).__init__(model, data)

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
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
        ...                       columns=['A', 'B', 'C', 'D', 'E'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
        >>> estimator = MaximumLikelihoodEstimator(model, values)
        >>> estimator.get_parameters()
        """
        parameters = []

        for node in self.model.nodes():
            parents = self.model.get_parents(node)
            if not parents:
                state_counts = self.data.ix[:, node].value_counts()
                state_counts = state_counts.reindex(sorted(state_counts.index))
                cpd = TabularCPD(node, self.node_card[node],
                                 state_counts.values[:, np.newaxis])
                cpd.normalize()
                parameters.append(cpd)
            else:
                parent_card = np.array([self.node_card[parent] for parent in parents])
                var_card = self.node_card[node]

                values = self.data.groupby([node] + parents).size().unstack(parents).fillna(0)
                if not len(values.columns) == np.prod(parent_card):
                    # some columns are missing if for some states of the parents no data was observed.
                    # reindex to add missing columns and fill in uniform (conditional) probabilities:
                    full_index = pd.MultiIndex.from_product([range(card) for card in parent_card], names=parents)
                    values = values.reindex(columns=full_index).fillna(1.0/var_card)

                cpd = TabularCPD(node, var_card, np.array(values),
                                 evidence=parents,
                                 evidence_card=parent_card.astype('int'))
                cpd.normalize()
                parameters.append(cpd)

        return parameters
