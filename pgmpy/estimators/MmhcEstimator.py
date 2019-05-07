#!/usr/bin/env python
from pgmpy.utils.mathext import powerset
from pgmpy.base import UndirectedGraph
from pgmpy.models import BayesianModel
from pgmpy.estimators import StructureEstimator, HillClimbSearch, BdeuScore
from pgmpy.independencies import Independencies, IndependenceAssertion


class MmhcEstimator(StructureEstimator):
    def __init__(self, data, **kwargs):
        """
        Implements the MMHC hybrid structure estimation procedure for
        learning BayesianModels from discrete data.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)

        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states (or values)
            that the variable can take. If unspecified, the observed values in the data set
            are taken to be the only possible states.

        complete_samples_only: bool (optional, default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
            This sets the behavior of the `state_count`-method.

        Reference
        ---------
        Tsamardinos et al., The max-min hill-climbing Bayesian network structure learning algorithm (2005)
        http://www.dsl-lab.org/supplements/mmhc_paper/paper_online.pdf
        """
        super(MmhcEstimator, self).__init__(data, **kwargs)

    def estimate(self, significance_level=0.01):
        """
        Estimates a BayesianModel for the data set, using MMHC. First estimates a
        graph skeleton using MMPC and then orients the edges using score-based local
        search (hill climbing).

        Parameters
        ----------
        significance_level: float, default: 0.01
            The significance level to use for conditional independence tests in the data set. See `mmpc`-method.

        Returns
        -------
        model: BayesianModel()-instance, not yet parametrized.

        Reference
        ---------
        Tsamardinos et al., The max-min hill-climbing Bayesian network structure learning algorithm (2005),
        Algorithm 3
        http://www.dsl-lab.org/supplements/mmhc_paper/paper_online.pdf

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> data = pd.DataFrame(np.random.randint(0, 2, size=(2500, 4)), columns=list('XYZW'))
        >>> data['sum'] = data.sum(axis=1)
        >>> est = MmhcEstimator(data)
        >>> model = est.estimate()
        >>> print(model.edges())
        [('Z', 'sum'), ('X', 'sum'), ('W', 'sum'), ('Y', 'sum')]
        """

        skel = self.mmpc(significance_level)

        hc = HillClimbSearch(self.data, scoring_method=BdeuScore(self.data, equivalent_sample_size=10))
        model = hc.estimate(white_list=skel.to_directed().edges(), tabu_length=10)

        return model

    def mmpc(self, significance_level=0.01):
        """Estimates a graph skeleton (UndirectedGraph) for the data set, using then
        MMPC (max-min parents-and-children) algorithm.

        Parameters
        ----------
        significance_level: float, default=0.01
            The significance level to use for conditional independence tests in the data set.

            `significance_level` is the desired Type 1 error probability of
            falsely rejecting the null hypothesis that variables are independent,
            given that they are. The lower `significance_level`, the less likely
            we are to accept dependencies, resulting in a sparser graph.

        Returns
        -------
        skeleton: UndirectedGraph
            An estimate for the undirected graph skeleton of the BN underlying the data.
        seperating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            seperating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation)


        Reference
        ---------
        Tsamardinos et al., The max-min hill-climbing Bayesian network structure learning algorithm (2005),
        Algorithm 1 & 2
        http://www.dsl-lab.org/supplements/mmhc_paper/paper_online.pdf

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>>
        >>> data = pd.DataFrame(np.random.randint(0, 2, size=(5000, 5)), columns=list('ABCDE'))
        >>> data['F'] = data['A'] + data['B'] + data ['C']
        >>> est = ConstraintBasedEstimator(data)
        >>> skel, sep_sets = est.estimate_skeleton()
        >>> skel.edges()
        [('A', 'F'), ('B', 'F'), ('C', 'F')]
        >>> # all independencies are unconditional:
        >>> sep_sets
        {('D', 'A'): (), ('C', 'A'): (), ('C', 'E'): (), ('E', 'F'): (), ('B', 'D'): (),
         ('B', 'E'): (), ('D', 'F'): (), ('D', 'E'): (), ('A', 'E'): (), ('B', 'A'): (),
         ('B', 'C'): (), ('C', 'D'): ()}
        >>>
        >>> data = pd.DataFrame(np.random.randint(0, 2, size=(5000, 3)), columns=list('XYZ'))
        >>> data['X'] += data['Z']
        >>> data['Y'] += data['Z']
        >>> est = ConstraintBasedEstimator(data)
        >>> skel, sep_sets = est.estimate_skeleton()
        >>> skel.edges()
        [('X', 'Z'), ('Y', 'Z')]
        >>> # X, Y dependent, but conditionally independent given Z:
        >>> sep_sets
        {('X', 'Y'): ('Z',)}
        """

        nodes = self.state_names.keys()

        def is_independent(X, Y, Zs):
            """Returns result of hypothesis test for the null hypothesis that
            X _|_ Y | Zs, using a chi2 statistic and threshold `significance_level`.
            """
            chi2, p_value, sufficient_data = self.test_conditional_independence(X, Y, Zs)
            return p_value >= significance_level and sufficient_data

        def assoc(X, Y, Zs):
            """Measure for (conditional) association between variables. Use negative
            p-value of independence test.
            """
            return 1 - self.test_conditional_independence(X, Y, Zs)[1]

        def min_assoc(X, Y, Zs):
            "Minimal association of X, Y given any subset of Zs."
            return min(assoc(X, Y, Zs_subset) for Zs_subset in powerset(Zs))

        def max_min_heuristic(X, Zs):
            "Finds variable that maximizes min_assoc with `node` relative to `neighbors`."
            max_min_assoc = 0
            best_Y = None

            for Y in set(nodes) - set(Zs + [X]):
                min_assoc_val = min_assoc(X, Y, Zs)
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
                    if is_independent(node, neigh, sep_set):
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
