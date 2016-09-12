#!/usr/bin/env python

from warnings import warn
from itertools import combinations

import networkx as nx

from pgmpy.estimators import StructureEstimator
from pgmpy.estimators import K2Score
from pgmpy.utils.mathext import powerset
from pgmpy.models import BayesianModel


class ExhaustiveSearch(StructureEstimator):
    def __init__(self, data, scoring_method=None, **kwargs):
        """
        Search class for exhaustive searches over all BayesianModels with a given set of variables.
        Takes a `StructureScore`-Instance as parameter; `estimate` finds the model with maximal score.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        scoring_method: Instance of a `StructureScore`-subclass (`K2Score` is used as default)
            An instance of `K2Score`, `BdeuScore`, or `BicScore`.
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
        """
        if scoring_method is not None:
            self.scoring_method = scoring_method
        else:
            self.scoring_method = K2Score(data, **kwargs)

        super(ExhaustiveSearch, self).__init__(data, **kwargs)

    def all_dags(self, nodes=None):
        """
        Computes all possible directed acyclic graphs with a given set of nodes,
        sparse ones first. `2**(n*(n-1))` graphs need to be searched, given `n` nodes,
        so this is likely not feasible for n>6. This is a generator.

        Parameters
        ----------
        nodes: list of nodes for the DAGs (optional)
            A list of the node names that the generated DAGs should have.
            If not provided, nodes are taken from data.

        Returns
        -------
        dags: Generator object for nx.DiGraphs
            Generator that yields all acyclic nx.DiGraphs, ordered by number of edges. Empty DAG first.

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.estimators import ExhaustiveSearch
        >>> s = ExhaustiveSearch(pd.DataFrame(data={'Temperature': [23, 19],
                                                    'Weather': ['sunny', 'cloudy'],
                                                    'Humidity': [65, 75]}))
        >>> list(s.all_dags())
        [<networkx.classes.digraph.DiGraph object at 0x7f6955216438>,
         <networkx.classes.digraph.DiGraph object at 0x7f6955216518>,
        ....
        >>> [dag.edges() for dag in s.all_dags()]
        [[], [('Humidity', 'Temperature')], [('Humidity', 'Weather')],
        [('Temperature', 'Weather')], [('Temperature', 'Humidity')],
        ....
        [('Weather', 'Humidity'), ('Weather', 'Temperature'), ('Temperature', 'Humidity')]]

        """
        if nodes is None:
            nodes = sorted(self.state_names.keys())
        if len(nodes) > 6:
            warn("Generating all DAGs of n nodes likely not feasible for n>6!")
            warn("Attempting to search through {0} graphs".format(2**(len(nodes)*(len(nodes)-1))))

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
        """
        Computes a list of DAGs and their structure scores, ordered by score.

        Returns
        -------
        list: a list of (score, dag) pairs
            A list of (score, dag)-tuples, where score is a float and model a acyclic nx.DiGraph.
            The list is ordered by score values.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ExhaustiveSearch, K2Score
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 2)), columns=list('AB'))
        >>> data['C'] = data['B']
        >>> searcher = ExhaustiveSearch(data, scoring_method=K2Score(data))
        >>> for score, model in searcher.all_scores():
        ...   print("{0}\t{1}".format(score, model.edges()))
        -24234.44977974726      [('A', 'B'), ('A', 'C')]
        -24234.449760691063     [('A', 'B'), ('C', 'A')]
        -24234.449760691063     [('A', 'C'), ('B', 'A')]
        -24203.700955937973     [('A', 'B')]
        -24203.700955937973     [('A', 'C')]
        -24203.700936881774     [('B', 'A')]
        -24203.700936881774     [('C', 'A')]
        -24203.700936881774     [('B', 'A'), ('C', 'A')]
        -24172.952132128685     []
        -16597.30920265254      [('A', 'B'), ('A', 'C'), ('B', 'C')]
        -16597.30920265254      [('A', 'B'), ('A', 'C'), ('C', 'B')]
        -16597.309183596342     [('A', 'B'), ('C', 'A'), ('C', 'B')]
        -16597.309183596342     [('A', 'C'), ('B', 'A'), ('B', 'C')]
        -16566.560378843253     [('A', 'B'), ('C', 'B')]
        -16566.560378843253     [('A', 'C'), ('B', 'C')]
        -16268.324549347722     [('A', 'B'), ('B', 'C')]
        -16268.324549347722     [('A', 'C'), ('C', 'B')]
        -16268.324530291524     [('B', 'A'), ('B', 'C')]
        -16268.324530291524     [('B', 'C'), ('C', 'A')]
        -16268.324530291524     [('B', 'A'), ('C', 'B')]
        -16268.324530291524     [('C', 'A'), ('C', 'B')]
        -16268.324530291524     [('B', 'A'), ('B', 'C'), ('C', 'A')]
        -16268.324530291524     [('B', 'A'), ('C', 'A'), ('C', 'B')]
        -16237.575725538434     [('B', 'C')]
        -16237.575725538434     [('C', 'B')]
        """

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

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ExhaustiveSearch
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(np.random.randint(0, 5, size=(5000, 2)), columns=list('AB'))
        >>> data['C'] = data['B']
        >>> est = ExhaustiveSearch(data)
        >>> best_model = est.estimate()
        >>> best_model
        <pgmpy.models.BayesianModel.BayesianModel object at 0x7f695c535470>
        >>> best_model.edges()
        [('B', 'C')]
        """

        best_dag = max(self.all_dags(), key=self.scoring_method.score)

        best_model = BayesianModel()
        best_model.add_nodes_from(sorted(best_dag.nodes()))
        best_model.add_edges_from(sorted(best_dag.edges()))
        return best_model
