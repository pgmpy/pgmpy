#!/usr/bin/env python

from itertools import combinations

import networkx as nx

from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.StructureScore import get_scoring_method
from pgmpy.global_vars import logger
from pgmpy.utils.mathext import powerset


class ExhaustiveSearch(StructureEstimator):
    """
    Search class for exhaustive searches over all DAGs with a given set of variables.
    Takes a `StructureScore`-Instance as parameter; `estimate` finds the model with maximal score.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.
        (If some values in the data are missing the data cells should be set to `numpy.NaN`.
        Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

    scoring_method: Instance of a `StructureScore`-subclass (`K2` is used as default)
        An instance of `K2`, `BDeu`, `BIC` or 'AIC'.
        This score is optimized during structure estimation by the `estimate`-method.

    state_names: dict (optional)
        A dict indicating, for each variable, the discrete set of states (or values)
        that the variable can take. If unspecified, the observed values in the data set
        are taken to be the only possible states.

    use_caching: boolean
        If True, uses caching of score for faster computation.
        Note: Caching only works for scoring methods which are decomposable. Can
        give wrong results in case of custom scoring methods.
    """

    def __init__(self, data, scoring_method=None, use_cache=True, **kwargs):
        super(ExhaustiveSearch, self).__init__(data, **kwargs)
        _, self.scoring_method = get_scoring_method(
            scoring_method, self.data, use_cache
        )

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
        >>> data = pd.DataFrame(
        ...     data={
        ...         "Temperature": [23, 19],
        ...         "Weather": ["sunny", "cloudy"],
        ...         "Humidity": [65, 75],
        ...     }
        ... )
        >>> est = ExhaustiveSearch(data)
        >>> list(est.all_dags())
        [<networkx.classes.digraph.DiGraph object at 0x...>, <networkx.classes.digraph.DiGraph object at 0x...>, ...]
        >>> [list(dag.edges()) for dag in est.all_dags()]
        [[], [('Humidity', 'Temperature')], [('Humidity', 'Weather')], [('Temperature', 'Weather')], ...

        """
        if nodes is None:
            nodes = sorted(self.state_names.keys())
        if len(nodes) > 6:
            logger.info("Generating all DAGs of n nodes likely not feasible for n>6!")
            logger.info(
                "Attempting to search through {n} graphs".format(
                    n=2 ** (len(nodes) * (len(nodes) - 1))
                )
            )

        edges = list(combinations(nodes, 2))  # n*(n-1) possible directed edges
        edges.extend([(y, x) for x, y in edges])
        all_graphs = powerset(edges)  # 2^(n*(n-1)) graphs

        for graph_edges in all_graphs:
            graph = nx.DiGraph(graph_edges)
            graph.add_nodes_from(nodes)
            if nx.is_directed_acyclic_graph(graph):
                yield graph

    def all_scores(self):
        """
        Computes a list of DAGs and their structure scores, ordered by score.

        Returns
        -------
        A list of (score, dag) pairs: list
            A list of (score, dag)-tuples, where score is a float and model a acyclic nx.DiGraph.
            The list is ordered by score values.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ExhaustiveSearch, K2
        >>> # Setting the random seed for reproducibility
        >>> np.random.seed(42)
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(
        ...     np.random.randint(low=0, high=5, size=(5000, 2)), columns=list("AB")
        ... )
        >>> data["C"] = data["B"]
        >>> searcher = ExhaustiveSearch(data, scoring_method=K2(data))
        >>> for score, model in searcher.all_scores():
        ...     print("{0}\t{1}".format(score, model.edges()))
        ...
        -24240.048463058432        [('A', 'B'), ('A', 'C')]
        -24240.03793877268        [('A', 'B'), ('C', 'A')]
        -24240.03793877268        [('A', 'C'), ('B', 'A')]
        -24207.21672011369        [('A', 'B')]
        -24207.21672011369        [('A', 'C')]
        -24207.20619582794        [('B', 'A')]
        -24207.20619582794        [('C', 'A')]
        -24174.38497716895        []
        -24143.64511922098        [('B', 'A'), ('C', 'A')]
        -16601.326068342074        [('A', 'B'), ('A', 'C'), ('C', 'B')]
        -16601.32606834207        [('A', 'B'), ('A', 'C'), ('B', 'C')]
        -16601.31554405632        [('A', 'B'), ('C', 'A'), ('C', 'B')]
        -16601.31554405632        [('A', 'C'), ('B', 'C'), ('B', 'A')]
        -16568.494325397332        [('A', 'B'), ('C', 'B')]
        -16568.494325397332        [('A', 'C'), ('B', 'C')]
        -16272.269477532585        [('A', 'B'), ('B', 'C')]
        -16272.269477532585        [('A', 'C'), ('C', 'B')]
        -16272.258953246836        [('B', 'C'), ('B', 'A')]
        -16272.258953246836        [('B', 'C'), ('C', 'A')]
        -16272.258953246836        [('B', 'A'), ('C', 'B')]
        -16272.258953246836        [('C', 'A'), ('C', 'B')]
        -16239.437734587846        [('B', 'C')]
        -16239.437734587846        [('C', 'B')]
        -16208.697876639875        [('B', 'C'), ('B', 'A'), ('C', 'A')]
        -16208.697876639875        [('B', 'A'), ('C', 'A'), ('C', 'B')]
        """

        scored_dags = sorted(
            [(self.scoring_method.score(dag), dag) for dag in self.all_dags()],
            key=lambda x: x[0],
        )
        return scored_dags

    def estimate(self):
        """
        Estimates the `DAG` structure that fits best to the given data set,
        according to the scoring method supplied in the constructor.
        Exhaustively searches through all models. Only estimates network structure, no parametrization.

        Returns
        -------
        Estimated Model: pgmpy.base.DAG
            A `DAG` with maximal score.

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ExhaustiveSearch, K2
        >>> # create random data sample with 3 variables, where B and C are identical:
        >>> data = pd.DataFrame(
        ...     np.random.randint(low=0, high=5, size=(5000, 2)), columns=list("AB")
        ... )
        >>> data["C"] = data["B"]
        >>> est = ExhaustiveSearch(data, scoring_method=K2(data))
        >>> best_model = est.estimate()
        >>> best_model
        <pgmpy.base.DAG.DAG object at 0x...>
        >>> best_model.edges()
        OutEdgeView([('B', 'A'), ('B', 'C'), ('C', 'A')])
        """

        best_dag = max(self.all_dags(), key=self.scoring_method.score)

        best_model = DAG()
        best_model.add_nodes_from(sorted(best_dag.nodes()))
        best_model.add_edges_from(sorted(best_dag.edges()))
        return best_model
