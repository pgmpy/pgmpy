#!/usr/bin/env python
import networkx as nx
import pandas as pd

from pgmpy.base import UndirectedGraph
from pgmpy.causal_discovery._base import _BaseCausalDiscovery
from pgmpy.causal_discovery.HillClimbSearch import HillClimbSearch as HillClimbSearchCD
from pgmpy.estimators import BDeu, ExpertKnowledge
from pgmpy.estimators.CITests import chi_square
from pgmpy.utils.mathext import powerset


class MMHC(_BaseCausalDiscovery):
    """
    sklearn compatible Implements the MMHC hybrid structure estimation procedure for
    learning BayesianNetworks from discrete data.

    Parameters
    ----------
    significance_level: float, default: 0.01
        The significance level to use for conditional independence tests in the data set. See `mmpc`-method.

    scoring_method: instance of a Scoring method (default: BDeu)
        The method to use for scoring during Hill Climb Search. Can be an instance of any of the
            scoring methods implemented in pgmpy.

    tabu_length: int
        If provided, the last `tabu_length` graph modifications cannot be reversed
        during the search procedure. This serves to enforce a wider exploration
        of the search space. Default value: 100.

    Returns
    -------
    Estimated model: pgmpy.base.DAG
        The estimated model without the parameterization.

    References
    ----------
    Tsamardinos et al., The max-min hill-climbing Bayesian network structure learning algorithm (2005),
    Algorithm 3
    http://www.dsl-lab.org/supplements/mmhc_paper/paper_online.pdf

    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from pgmpy.estimators import MmhcEstimator
    >>> data = pd.DataFrame(
    ...     np.random.randint(0, 2, size=(2500, 4)), columns=list("XYZW")
    ... )
    >>> data["sum"] = data.sum(axis=1)
    >>> est = MmhcEstimator(data)
    >>> model = est.estimate()
    >>> print(model.edges())
    [('Z', 'sum'), ('X', 'sum'), ('W', 'sum'), ('Y', 'sum')]

    """

    def __init__(self, scoring_method=None, tabu_length=10, significance_level=0.01):
        self.scoring_method = scoring_method
        self.tabu_length = tabu_length
        self.significance_level = significance_level

    def _fit(self, X: pd.DataFrame):
        """
        Fits the MMHC algorithm on the given data. First estimates a graph
        skeleton using MMPC, then orients edges using score-based Hill Climbing.

        Parameters
        ----------
        data: pd.DataFrame
            DataFrame where each column represents one variable. Missing values
            should be set to numpy.nan.

        Returns
        -------
        self: MMHC
            Returns self. The fitted model is stored in self.model_

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.causal_discovery import MMHC
        >>> data = pd.DataFrame(
        ...     np.random.randint(0, 2, size=(2500, 4)), columns=list("XYZW")
        ... )
        >>> data["sum"] = data.sum(axis=1)
        >>> est = MMHC(significance_level=0.01, tabu_length=10)
        >>> est.fit(data)
        >>> print(est.causal_graph_.edges())
        [('Z', 'sum'), ('X', 'sum'), ('W', 'sum'), ('Y', 'sum')]
        """
        self.X_ = X
        self.state_names_ = {col: X[col].unique().tolist() for col in X.columns}
        scoring_method = self.scoring_method

        if scoring_method is None:
            scoring_method = BDeu(X, equivalent_sample_size=10)

        skel = self.mmpc(X)
        possible_edges = nx.complete_graph(n=self.state_names_.keys(), create_using=nx.Graph).edges()

        expert_knowledge = ExpertKnowledge(forbidden_edges=possible_edges - skel.to_directed().edges())
        hc = HillClimbSearchCD(
            scoring_method=scoring_method,
            expert_knowledge=expert_knowledge,
            tabu_length=self.tabu_length,
        )
        hc.fit(X)
        self.causal_graph_ = hc.causal_graph_
        self.adjacency_matrix_ = nx.to_pandas_adjacency(self.causal_graph_, weight=1, dtype="int")

        return self

    def mmpc(self, data):
        """Estimates a graph skeleton (UndirectedGraph) for the data set, using then
        MMPC (max-min parents-and-children) algorithm.

        Parameters
        ----------
        data:pd.DataFrame
            DataFrame where each column is variable.

        Returns
        -------
        skeleton: pgmpy.base.UndirectedGraph
            An estimate for the undirected graph skeleton of the BN underlying the data.

        References
        ----------
        Tsamardinos et al., The max-min hill-climbing Bayesian network structure
        learning algorithm (2005), Algorithm 1 & 2
        http://www.dsl-lab.org/supplements/mmhc_paper/paper_online.pdf

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import MmhcEstimator
        >>> data = pd.DataFrame(
        ...     np.random.randint(0, 2, size=(5000, 5)), columns=list("ABCDE")
        ... )
        >>> data["F"] = data["A"] + data["B"] + data["C"]
        >>> est = PC(data)
        >>> skel, sep_sets = est.estimate_skeleton()
        >>> skel.edges()
        [('A', 'F'), ('B', 'F'), ('C', 'F')]
        >>> # all independencies are unconditional:
        >>> sep_sets
        {('D', 'A'): (), ('C', 'A'): (), ('C', 'E'): (), ('E', 'F'): (), ('B', 'D'): (),
         ('B', 'E'): (), ('D', 'F'): (), ('D', 'E'): (), ('A', 'E'): (), ('B', 'A'): (),
         ('B', 'C'): (), ('C', 'D'): ()}
        >>> data = pd.DataFrame(
        ...     np.random.randint(0, 2, size=(5000, 3)), columns=list("XYZ")
        ... )
        >>> data["X"] += data["Z"]
        >>> data["Y"] += data["Z"]
        >>> est = PC(data)
        >>> skel, sep_sets = est.estimate_skeleton()
        >>> skel.edges()
        [('X', 'Z'), ('Y', 'Z')]
        >>> # X, Y dependent, but conditionally independent given Z:
        >>> sep_sets
        {('X', 'Y'): ('Z',)}
        """

        nodes = self.state_names_.keys()

        def max_min_heuristic(X, Zs):
            "Finds variable that maximizes min_assoc with `node` relative to `neighbors`."
            max_min_assoc = 0
            best_Y = None
            for Y in set(nodes) - set(Zs + [X]):
                min_assoc_val = min(
                    1 - chi_square(X, Y, Zs_subset, data, boolean=False)[1] for Zs_subset in powerset(Zs)
                )
                if min_assoc_val >= max_min_assoc:
                    best_Y = Y
                    max_min_assoc = min_assoc_val
            return (best_Y, max_min_assoc)

        # Find parents and children for each node
        neighbors = dict()
        for node in nodes:
            neighbors[node] = []

            # Forward Phase
            while True:
                new_neighbor, new_neighbor_min_assoc = max_min_heuristic(node, neighbors[node])
                if new_neighbor_min_assoc > 0:
                    neighbors[node].append(new_neighbor)
                else:
                    break

            # Backward Phase
            for neigh in neighbors[node]:
                other_neighbors = [n for n in neighbors[node] if n != neigh]
                for sep_set in powerset(other_neighbors):
                    if chi_square(
                        X=node,
                        Y=neigh,
                        Z=sep_set,
                        data=data,
                        significance_level=self.significance_level,
                    ):
                        neighbors[node].remove(neigh)
                        break

        # correct for false positives
        for node in nodes:
            for neigh in neighbors[node]:
                if node not in neighbors[neigh]:
                    neighbors[node].remove(neigh)

        skel = UndirectedGraph()
        skel.add_nodes_from(nodes)
        for node in nodes:
            skel.add_edges_from([(node, neigh) for neigh in neighbors[node]])

        return skel
