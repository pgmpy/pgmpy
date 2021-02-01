# -*- coding: utf-8 -*-

from itertools import chain
from warnings import warn

import numpy as np
import numbers

from pgmpy.estimators import ParameterEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import BayesianModel


class BayesianEstimator(ParameterEstimator):
    def __init__(self, model, data, **kwargs):
        """
        Class used to compute parameters for a model using Bayesian Parameter Estimation.
        See `MaximumLikelihoodEstimator` for constructor parameters.
        """
        if not isinstance(model, BayesianModel):
            raise NotImplementedError(
                "Bayesian Parameter Estimation is only implemented for BayesianModel"
            )

        super(BayesianEstimator, self).__init__(model, data, **kwargs)

    def get_parameters(
        self, prior_type="BDeu", equivalent_sample_size=5, pseudo_counts=None
    ):
        """
        Method to estimate the model parameters (CPDs).

        Parameters
        ----------
        prior_type: 'dirichlet', 'BDeu', or 'K2'
            string indicting which type of prior to use for the model parameters.
            - If 'prior_type' is 'dirichlet', the following must be provided:
                'pseudo_counts' = dirichlet hyperparameters; a single number or a dict containing, for each
                 variable, a 2-D array of the shape (node_card, product of parents_card) with a "virtual"
                 count for each variable state in the CPD, that is added to the state counts.
                 (lexicographic ordering of states assumed)
            - If 'prior_type' is 'BDeu', then an 'equivalent_sample_size'
                must be specified instead of 'pseudo_counts'. This is equivalent to
                'prior_type=dirichlet' and using uniform 'pseudo_counts' of
                `equivalent_sample_size/(node_cardinality*np.prod(parents_cardinalities))` for each node.
                'equivalent_sample_size' can either be a numerical value or a dict that specifies
                the size for each variable separately.
            - A prior_type of 'K2' is a shorthand for 'dirichlet' + setting every pseudo_count to 1,
                regardless of the cardinality of the variable.

        Returns
        -------
        parameters: list
            List of TabularCPDs, one for each variable of the model

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import BayesianEstimator
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
        ...                       columns=['A', 'B', 'C', 'D'])
        >>> model = BayesianModel([('A', 'B'), ('C', 'B'), ('C', 'D')])
        >>> estimator = BayesianEstimator(model, values)
        >>> estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=5)
        [<TabularCPD representing P(C:2) at 0x7f7b534251d0>,
        <TabularCPD representing P(B:2 | C:2, A:2) at 0x7f7b4dfd4da0>,
        <TabularCPD representing P(A:2) at 0x7f7b4dfd4fd0>,
        <TabularCPD representing P(D:2 | C:2) at 0x7f7b4df822b0>]
        """
        parameters = []
        for node in self.model.nodes():
            _equivalent_sample_size = (
                equivalent_sample_size[node]
                if isinstance(equivalent_sample_size, dict)
                else equivalent_sample_size
            )
            if isinstance(pseudo_counts, numbers.Real):
                _pseudo_counts = pseudo_counts
            else:
                _pseudo_counts = pseudo_counts[node] if pseudo_counts else None

            cpd = self.estimate_cpd(
                node,
                prior_type=prior_type,
                equivalent_sample_size=_equivalent_sample_size,
                pseudo_counts=_pseudo_counts,
            )
            parameters.append(cpd)

        return parameters

    def estimate_cpd(
        self, node, prior_type="BDeu", pseudo_counts=[], equivalent_sample_size=5
    ):
        """
        Method to estimate the CPD for a given variable.

        Parameters
        ----------
        node: int, string (any hashable python object)
            The name of the variable for which the CPD is to be estimated.

        prior_type: 'dirichlet', 'BDeu', 'K2',
            string indicting which type of prior to use for the model parameters.
            - If 'prior_type' is 'dirichlet', the following must be provided:
                'pseudo_counts' = dirichlet hyperparameters; a single number or 2-D array
                 of shape (node_card, product of parents_card) with a "virtual" count for
                 each variable state in the CPD. The virtual counts are added to the
                 actual state counts found in the data. (if a list is provided, a
                 lexicographic ordering of states is assumed)
            - If 'prior_type' is 'BDeu', then an 'equivalent_sample_size'
                 must be specified instead of 'pseudo_counts'. This is equivalent to
                 'prior_type=dirichlet' and using uniform 'pseudo_counts' of
                 `equivalent_sample_size/(node_cardinality*np.prod(parents_cardinalities))`.
            - A prior_type of 'K2' is a shorthand for 'dirichlet' + setting every
              pseudo_count to 1, regardless of the cardinality of the variable.

        Returns
        -------
        CPD: TabularCPD

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import BayesianEstimator
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = BayesianModel([('A', 'C'), ('B', 'C')])
        >>> estimator = BayesianEstimator(model, data)
        >>> cpd_C = estimator.estimate_cpd('C', prior_type="dirichlet",
        ...                                pseudo_counts=[[1, 1, 1, 1],
        ...                                               [2, 2, 2, 2]])
        >>> print(cpd_C)
        ╒══════╤══════╤══════╤══════╤════════════════════╕
        │ A    │ A(0) │ A(0) │ A(1) │ A(1)               │
        ├──────┼──────┼──────┼──────┼────────────────────┤
        │ B    │ B(0) │ B(1) │ B(0) │ B(1)               │
        ├──────┼──────┼──────┼──────┼────────────────────┤
        │ C(0) │ 0.25 │ 0.25 │ 0.5  │ 0.3333333333333333 │
        ├──────┼──────┼──────┼──────┼────────────────────┤
        │ C(1) │ 0.75 │ 0.75 │ 0.5  │ 0.6666666666666666 │
        ╘══════╧══════╧══════╧══════╧════════════════════╛

        """
        node_cardinality = len(self.state_names[node])
        parents = sorted(self.model.get_parents(node))
        parents_cardinalities = [len(self.state_names[parent]) for parent in parents]
        cpd_shape = (node_cardinality, np.prod(parents_cardinalities, dtype=int))

        prior_type = prior_type.lower()

        # Throw a warning if pseudo_count is specified without prior_type=dirichlet
        if (pseudo_counts and pseudo_counts != []) and (prior_type != "dirichlet"):
            warn(
                f"pseudo count specified with {prior_type} prior. It will be ignored, use dirichlet prior for specifying pseudo_counts"
            )

        if prior_type == "k2":
            pseudo_counts = np.ones(cpd_shape, dtype=int)
        elif prior_type == "bdeu":
            alpha = float(equivalent_sample_size) / (
                node_cardinality * np.prod(parents_cardinalities)
            )
            pseudo_counts = np.ones(cpd_shape, dtype=float) * alpha
        elif prior_type == "dirichlet":
            if isinstance(pseudo_counts, numbers.Real):
                pseudo_counts = np.ones(cpd_shape, dtype=int) * pseudo_counts

            else:
                pseudo_counts = np.array(pseudo_counts)
                if pseudo_counts.shape != cpd_shape:
                    raise ValueError(
                        f"The shape of pseudo_counts for the node: {node} must be of shape: {str(cpd_shape)}"
                    )
        else:
            raise ValueError("'prior_type' not specified")

        state_counts = self.state_counts(node)
        bayesian_counts = state_counts + pseudo_counts

        cpd = TabularCPD(
            node,
            node_cardinality,
            np.array(bayesian_counts),
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names={var: self.state_names[var] for var in chain([node], parents)},
        )
        cpd.normalize()
        return cpd
