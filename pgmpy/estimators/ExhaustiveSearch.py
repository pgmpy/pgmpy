#!/usr/bin/env python
import numpy as np
import pandas as pd
import networkx as nx
from warnings import warn
from itertools import combinations
from pgmpy.estimators import StructureEstimator, BayesianScore
from pgmpy.utils.mathext import powerset
from pgmpy.models import BayesianModel


class ExhaustiveSearch(StructureEstimator):
    def __init__(self, data, scoring_method=None, **kwargs):
        """
        Search class for exhaustive searches over all BayesianModels with a given set of variables.
        Takes a `StructureScore`-Instance as parameter; `estimate` finds the model with maximal score.

        Parameters
        ----------
        model: pgmpy.models.BayesianModel or pgmpy.models.MarkovModel or pgmpy.models.NoisyOrModel
            model for which parameter estimation is to be done

        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        scoring_method: Instance of a `StructureScore`-subclass (`BayesianScore` is used if not set)
            An instance of either `BayesianScore` or `BicScore`.
            This score is optimized during structure estimation by the `estimate`-method.

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.

        Returns
        -------
        state_counts: pandas.DataFrame
            Table with state counts for 'variable'

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.estimators import ParameterEstimator
        >>> model = BayesianModel([('A', 'C'), ('B', 'C')])
        >>> data = pd.DataFrame(data={'A': ['a1', 'a1', 'a2'],
                                      'B': ['b1', 'b2', 'b1'],
                                      'C': ['c1', 'c1', 'c2']})
        >>> estimator = ParameterEstimator(model, data)
        >>> estimator.state_counts('A')
            A
        a1  2
        a2  1
        >>> estimator.state_counts('C')
        A  a1      a2
        B  b1  b2  b1  b2
        C
        c1  1   1   0   0
        c2  0   0   1   0
        """
        if scoring_method is not None:
            self.scoring_method = scoring_method
        else:
            from pgmpy.estimators import BayesianScore
            self.scoring_method = BayesianScore(data, **kwargs)

        super(ExhaustiveSearch, self).__init__(data, **kwargs)

    def all_dags(self, nodes=None):
        "Generates all possible DAGs with a given set of nodes; sparse ones first"
        if nodes is None:
            nodes = sorted(self.state_names.keys())
        if len(nodes) > 6:
            warn("Generating all DAGs of n nodes likely not feasible for n>6")

        edges = list(combinations(nodes, 2))  # n*(n-1) possible directed edges
        edges.extend([(y, x) for x, y in edges])
        all_graphs = powerset(edges)  # 2^(n*(n-1)) graphs

        for graph_edges in all_graphs:
            graph = nx.DiGraph()
            graph.add_nodes_from(nodes)
            graph.add_edges_from(graph_edges)
            if nx.is_directed_acyclic_graph(graph):
                yield graph

    def all_scores(self):
        "Computes an list of DAGs and their structure scores, ordered by score"

        scored_dags = sorted([(self.scoring_method.score(dag), dag) for dag in self.all_dags()],
                             key=lambda x: x[0])
        return scored_dags

    def estimate(self):
        """
        Estimates the `BayesianModel` structure that fits best to the given data set,
        according to the scoring method supplied in the constructor.
        Exhaustively searches through all models. Only estimates network structure, no parametrization.

        Returns
        -------
        model: `BayesianModel` instance
            A `BayesianModel` with maximal score.
        """

        best_dag = max(self.all_dags(), key=self.scoring_method.score)

        best_model = BayesianModel()
        best_model.add_nodes_from(sorted(best_dag.nodes()))
        best_model.add_edges_from(sorted(best_dag.edges()))
        return best_model
