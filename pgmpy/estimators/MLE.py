# coding:utf-8

from itertools import chain
import warnings

import numpy as np
from joblib import Parallel, delayed

from pgmpy.base import DAG
from pgmpy.estimators import ParameterEstimator
from pgmpy.factors import FactorDict
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import DiscreteBayesianNetwork, JunctionTree


class MaximumLikelihoodEstimator(ParameterEstimator):
    """
    Class used to compute parameters for a model using Maximum Likelihood Estimation.

    Parameters
    ----------
    model: A pgmpy.models.DiscreteBayesianNetwork or pgmpy.models.JunctionTree instance

    data: pandas DataFrame object
        DataFrame object with column names identical to the variable names of the network.
        (If some values in the data are missing the data cells should be set to `numpy.nan`.
        Note that pandas converts each column containing `numpy.nan`s to dtype `float`.)

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states
        that the variable can take. If unspecified, the observed values
        in the data set are taken to be the only possible states.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from pgmpy.models import DiscreteBayesianNetwork
    >>> from pgmpy.estimators import MaximumLikelihoodEstimator
    >>> data = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 5)),
    ...                       columns=['A', 'B', 'C', 'D', 'E'])
    >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D'), ('B', 'E')])
    >>> estimator = MaximumLikelihoodEstimator(model, data)
    """

    def __init__(self, model, data, **kwargs):
        if not isinstance(model, (DiscreteBayesianNetwork, JunctionTree, DAG)):
            raise NotImplementedError(
                "Maximum Likelihood Estimate is only implemented for DiscreteBayesianNetwork, JunctionTree, and DAG"
            )

        if isinstance(model, (DAG, DiscreteBayesianNetwork)):
            if len(model.latents) > 0:
                raise ValueError(
                    f"Found latent variables: {model.latents}. "
                    "Maximum Likelihood doesn't support latent variables, please use ExpectationMaximization."
                )

            elif set(model.nodes()) > set(data.columns):
                raise ValueError(
                    f"Nodes detected in the model that are not present "
                    f"in the dataset: {set(model.nodes) - set(data.columns)}. "
                    "Refine the model so that all parameters can be estimated from the data."
                )

            if isinstance(model, JunctionTree):
                if len(set(data.columns) - set(chain(*model.nodes()))) != 0:
                    raise ValueError(
                        f"Nodes detected in the JunctionTree that are not present "
                        f"in the dataset: {set(data.columns) - set(chain(*model.nodes()))} "
                        "Refine the model to ensure all parameters can be estimated."
                    )

        super(MaximumLikelihoodEstimator, self).__init__(model, data, **kwargs)

    def get_parameters(self, n_jobs=1, weighted=False,smoothing=None,alpha=1.0):
        """
        Method to estimate the model parameters using Maximum Likelihood Estimation.

        Parameters
        ----------
        n_jobs: int (default: 1)
            Number of jobs to run in parallel. Default: 1 uses all the processors.
            Using n_jobs > 1 for small models might be slower.

        weighted: bool
            If weighted=True, the data must contain a `_weight` column specifying the
            weight of each datapoint (row). If False, assigns an equal weight to each
            datapoint.

        smoothing: str or None (default: None)
            Smoothing technique to handle zero counts. Options:
            - None: Use uniform smoothing (original behavior) with warning
            - 'laplace': Add alpha to all counts (Laplace smoothing)
            - 'dirichlet': Add alpha to all counts (equivalent to Laplace)
            - 'none': No smoothing, keep zero probabilities

         alpha: float (default: 1.0)
            Smoothing parameter when using Laplace/Dirichlet smoothing.

        Returns
        -------
        Estimated parameters: list or pgmpy.factors.FactorDict
            List of pgmpy.factors.discrete.TabularCPDs, one for each variable of the model
            Or a FactorDict representing potential values of a Junction Tree

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
        ...                       columns=['A', 'B', 'C', 'D'])
        >>> model = DiscreteBayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D')])
        >>> estimator = MaximumLikelihoodEstimator(model, values)
        >>> # Standard MLE with uniform smoothing (with warning)
        >>> estimator.get_parameters()
        [<TabularCPD representing P(C:2) at 0x7f7b534251d0>,
        <TabularCPD representing P(B:2 | C:2, A:2) at 0x7f7b4dfd4da0>,
        <TabularCPD representing P(A:2) at 0x7f7b4dfd4fd0>,
        <TabularCPD representing P(D:2 | C:2) at 0x7f7b4df822b0>]
        >>> # MLE with Laplace smoothing
        >>> estimator.get_parameters(smoothing='laplace',alpha=1.0)
        >>> # MLE with no smoothing (keeps zero probabilities)
        >>> estimator.get_parameters(smoothing='none')
        """

        if isinstance(self.model, JunctionTree):
            return self.estimate_potentials()

        parameters = Parallel(n_jobs=n_jobs)(
            delayed(self.estimate_cpd)(node, weighted) for node in self.model.nodes()
        )
        # TODO: A hacky solution to return correct value for the chosen backend. Ref #1675
        parameters = [p.copy() for p in parameters]

        return parameters

    def estimate_cpd(self, node, weighted=False,smoothing=None,alpha=1.0):
        """
        Method to estimate the CPD for a given variable.

        Parameters
        ----------
        node: int, string (any hashable python object)
            The name of the variable for which the CPD is to be estimated.

        weighted: bool
            If weighted=True, the data must contain a `_weight` column specifying the
            weight of each datapoint (row). If False, assigns an equal weight to each
            datapoint.

        smoothing: str or None (default: None)
            Smoothing technique to handle zero counts. Options:
            - None: Use uniform smoothing (original behavior) with warning
            - 'laplace': Add alpha to all counts (Laplace smoothing)
            - 'dirichlet': Add alpha to all counts (equivalent to Laplace)
            - 'none': No smoothing, keep zero probabilities
        
        alpha: float (default: 1.0)
            Smoothing parameter when using Laplace/Dirichlet smoothing.

        Returns
        -------
        Estimated CPD: pgmpy.factors.discrete.TabularCPD
            Estimated CPD for `node`.

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import DiscreteBayesianNetwork
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = DiscreteBayesianNetwork([('A', 'C'), ('B', 'C')])
        >>> cpd_A = MaximumLikelihoodEstimator(model, data).estimate_cpd('A')
        >>> print(cpd_A)
        ╒══════╤══════════╕
        │ A(0) │ 0.666667 │
        ├──────┼──────────┤
        │ A(1) │ 0.333333 │
        ╘══════╧══════════╛
        >>> cpd_C = MaximumLikelihoodEstimator(model, data).estimate_cpd('C')
        >>> print(cpd_C)
        ╒══════╤══════╤══════╤══════╤══════╕
        │ A    │ A(0) │ A(0) │ A(1) │ A(1) │
        ├──────┼──────┼──────┼──────┼──────┤
        │ B    │ B(0) │ B(1) │ B(0) │ B(1) │
        ├──────┼──────┼──────┼──────┼──────┤
        │ C(0) │ 0.0  │ 0.0  │ 1.0  │ 0.5  │
        ├──────┼──────┼──────┼──────┼──────┤
        │ C(1) │ 1.0  │ 1.0  │ 0.0  │ 0.5  │
        ╘══════╧══════╧══════╧══════╧══════╛
        """

        """
        >>> # Original behavior with warning
        >>> cpd_A = MaximumLikelihoodEstimator(model, data).estimate_cpd('A')
        >>> # With Laplace smoothing
        >>> cpd_A = MaximumLikelihoodEstimator(model, data).estimate_cpd('A', smoothing='laplace')
        >>> # With no smoothing
        >>> cpd_A = MaximumLikelihoodEstimator(model, data).estimate_cpd('A', smoothing='none')
        """

        state_counts = self.state_counts(node, weighted=weighted)

        #Handle zero counts based on smoothing strategy
        zero_columns = (state_counts.values == 0).all(axis=0)

        if zero_columns.any():
            if smoothing is None:
                # Original behavior with deprecation warning
                warnings.warn(
                    f"Zero counts detected for node '{node}'. Using uniform smoothing (filling with 1.0). "
                    "This behavior is deprecated. Please specify a smoothing method explicitly using the 'smoothing' parameter. "
                    "Options: 'laplace', 'dirichlet', or 'none'.",
                    DeprecationWarning,
                    stacklevel=2
                )
                state_counts.iloc[:, zero_columns] = 1.0
                
            elif smoothing.lower() in ['laplace', 'dirichlet']:
                # Add alpha to all counts (Laplace/Dirichlet smoothing)
                if alpha <= 0:
                    raise ValueError("Alpha must be positive for Laplace/Dirichlet smoothing")
                state_counts = state_counts + alpha
                
            elif smoothing.lower() == 'none':
                # Keep zero counts as they are - will result in zero probabilities
                pass
                
            else:
                raise ValueError(
                    f"Unknown smoothing method: {smoothing}. "
                    "Supported methods: 'laplace', 'dirichlet', 'none', or None for uniform smoothing."
                )

        parents = sorted(self.model.get_parents(node))
        parents_cardinalities = [len(self.state_names[parent]) for parent in parents]
        node_cardinality = len(self.state_names[node])

        # Get the state names for the CPD
        state_names = {node: list(state_counts.index)}
        if parents:
            state_names.update(
                {
                    state_counts.columns.names[i]: list(state_counts.columns.levels[i])
                    for i in range(len(parents))
                }
            )

        cpd = TabularCPD(
            node,
            node_cardinality,
            np.array(state_counts),
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names={var: self.state_names[var] for var in chain([node], parents)},
        )

        # Check for zero probabilities after normalization if smoothing is 'none'
        if smoothing == 'none' and zero_columns.any():
            warnings.warn(
                f"Zero probabilities detected in CPD for node '{node}' after normalization. "
                "This may cause issues during inference. Consider using smoothing.",
                UserWarning,
                stacklevel=2
            )

        cpd.normalize()
        return cpd

    def estimate_potentials(self):
        """
        Implements Iterative Proportional Fitting to estimate potentials specifically
        for a Decomposable Undirected Graphical Model. Decomposability is enforced
        by using a Junction Tree.

        Returns
        -------
        Estimated potentials: pgmpy.factors.FactorDict
            Estimated potentials for the entire graphical model.

        References
        ---------
        [1] Kevin P. Murphy, ML Machine Learning - A Probabilistic Perspective
            Algorithm 19.2 Iterative Proportional Fitting algorithm
              for tabular MRFs & Section 19.5.7.4 IPF for decomposable graphical models.
        [2] Eric P. Xing, Meng Song, Li Zhou, Probabilistic Graphical Models 10-708, Spring 2014.
            https://www.cs.cmu.edu/~epxing/Class/10708-14/scribe_notes/scribe_note_lecture8.pdf.

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import JunctionTree
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = JunctionTree()
        >>> model.add_edges_from([(("A", "C"), ("B", "C"))])
        >>> potentials = MaximumLikelihoodEstimator(model, data).estimate_potentials()
        >>> print(potentials[("A", "C")])
        +------+------+------------+
        | A    | C    |   phi(A,C) |
        +======+======+============+
        | A(0) | C(0) |     0.0000 |
        +------+------+------------+
        | A(0) | C(1) |     0.6667 |
        +------+------+------------+
        | A(1) | C(0) |     0.3333 |
        +------+------+------------+
        | A(1) | C(1) |     0.0000 |
        +------+------+------------+
        >>> print(potentials[("B", "C")])
        +------+------+------------+
        | B    | C    |   phi(B,C) |
        +======+======+============+
        | B(0) | C(0) |     1.0000 |
        +------+------+------------+
        | B(0) | C(1) |     0.5000 |
        +------+------+------------+
        | B(1) | C(0) |     0.0000 |
        +------+------+------------+
        | B(1) | C(1) |     0.5000 |
        +------+------+------------+
        """
        if not isinstance(self.model, JunctionTree):
            raise NotImplementedError(
                "Iterative Proportional Fitting is only implemented for Junction Trees."
            )

        if not hasattr(self.model, "clique_beliefs"):
            raise NotImplementedError(
                "A model containing clique beliefs is required to estimate parameters."
            )

        clique_beliefs = self.model.clique_beliefs

        if not isinstance(clique_beliefs, FactorDict):
            raise TypeError(
                "`UndirectedMaximumLikelihoodEstimator.model.clique_beliefs` must be a `FactorDict`."
            )

        # These are the variables as represented by the `JunctionTree`.
        cliques = list(clique_beliefs.keys())
        empirical_marginals = FactorDict.from_dataframe(df=self.data, marginals=cliques)
        potentials = FactorDict({})
        seen = set()

        # ML Machine Learning - A Probabilistic Perspective
        # Chapter 19, Algorithm 19.2, Page 682:
        # Update each clique by multiplying the potential value by
        # the ratio of the empirical counts over expected counts.
        # Since the potential values are equal to the expected counts
        # for a JunctionTree, we can simplify this to just the empirical counts.
        # This is also described in section 19.5.7.4.
        for clique in cliques:
            # Calculate the running sepset between the new clique and all of the
            # variables we have previously seen.
            variables = tuple(set(clique) - seen)
            seen.update(clique)
            potentials[clique] = empirical_marginals[clique]

            # Divide out the sepset.
            if variables:
                marginalized = empirical_marginals[clique].marginalize(
                    variables=variables, inplace=False
                )
                potentials[clique] = potentials[clique] / marginalized
        return potentials
