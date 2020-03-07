#!/usr/bin/env python

from warnings import warn
from itertools import combinations
import networkx as nx

from pgmpy.base import UndirectedGraph
from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator
from pgmpy.independencies import Independencies, IndependenceAssertion
from pgmpy.estimators.CITests import ChiSquare, Pearsonr, IndependenceMatching


class PC(StructureEstimator):
    def __init__(self, data=None, ci_test="chi_square", **kwargs):
        """
        Class for constraint-based estimation of DAGs from a given
        data set. Identifies (conditional) dependencies in data set using
        chi_square dependency test and uses the PC algorithm to estimate a DAG
        pattern that satisfies the identified dependencies. The DAG pattern can
        then be completed to a faithful DAG, if possible.

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

        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques,
            2009, Section 18.2
        [2] Neapolitan, Learning Bayesian Networks, Section 10.1.2 for the PC algorithm (page 550),
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        """
        if ci_test == "chi_square":
            self.ci_test = ChiSquare(data=data, kwargs=kwargs)
        elif ci_test == "pearsonr":
            self.ci_test = Pearsonr(data=data, kwargs=kwargs)
        elif ci_test == "independence_match":
            self.ci_test = IndependenceMatching(**kwargs)

        super(PC, self).__init__(data=data, kwargs=kwargs)

    def estimate(self, significance_level=0.01):
        """
        Estimates a DAG for the data set, using the PC constraint-based
        structure learning algorithm. Independencies are identified from the
        data set using a chi-squared statistic with the acceptance threshold of
        `significance_level`. PC identifies a partially directed acyclic graph (PDAG), given
        that the tested independencies admit a faithful Bayesian network representation.
        This method returns a DAG that is a completion of this PDAG.

        Parameters
        ----------
        significance_level: float, default: 0.01
            The significance level to use for conditional independence tests in the data set.

            `significance_level` is the desired Type 1 error probability of
            falsely rejecting the null hypothesis that variables are independent,
            given that they are. The lower `significance_level`, the less likely
            we are to accept dependencies, resulting in a sparser graph.

        Returns
        -------
        model: DAG()-instance
            An estimate for the DAG for the data set (not yet parametrized).

        References
        ----------
        Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> data = pd.DataFrame(np.random.randint(0, 5, size=(2500, 3)), columns=list('XYZ'))
        >>> data['sum'] = data.sum(axis=1)
        >>> print(data)
              X  Y  Z  sum
        0     3  0  1    4
        1     1  4  3    8
        2     0  0  3    3
        3     0  2  3    5
        4     2  1  1    4
        ...  .. .. ..  ...
        2495  2  3  0    5
        2496  1  1  2    4
        2497  0  4  2    6
        2498  0  0  0    0
        2499  2  4  0    6

        [2500 rows x 4 columns]
        >>> c = ConstraintBasedEstimator(data)
        >>> model = c.estimate()
        >>> print(model.edges())
        [('Z', 'sum'), ('X', 'sum'), ('Y', 'sum')]
        """
        skel, separating_sets = self.build_skeleton()
        pdag = self.skeleton_to_pdag(skel, separating_sets)
        return pdag.to_dag()

    def build_skeleton(self):
        """Estimates a graph skeleton (UndirectedGraph) from a set of independencies
        using (the first part of) the PC algorithm. The independencies can either be
        provided as an instance of the `Independencies`-class or by passing a
        decision function that decides any conditional independency assertion.
        Returns a tuple `(skeleton, separating_sets)`.

        If an Independencies-instance is passed, the contained IndependenceAssertions
        have to admit a faithful BN representation. This is the case if
        they are obtained as a set of d-seperations of some Bayesian network or
        if the independence assertions are closed under the semi-graphoid axioms.
        Otherwise the procedure may fail to identify the correct structure.

        Parameters
        ----------
        nodes: list, array-like
            A list of node/variable names of the network skeleton.

        independencies: Independencies-instance or function.
            The source of independency information from which to build the skeleton.
            The provided Independencies should admit a faithful representation.
            Can either be provided as an Independencies()-instance or by passing a
            function `f(X, Y, Zs)` that returns `True` when X _|_ Y | Zs,
            otherwise `False`. (X, Y being individual nodes and Zs a list of nodes).

        Returns
        -------
        skeleton: UndirectedGraph
            An estimate for the undirected graph skeleton of the BN underlying the data.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation procedures)

        References
        ----------
        [1] Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [2] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
            Section 3.4.2.1 (page 85), Algorithm 3.3

        Examples
        --------
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> from pgmpy.models import DAG
        >>> from pgmpy.independencies import Independencies

        >>> # build skeleton from list of independencies:
        ... ind = Independencies(['B', 'C'], ['A', ['B', 'C'], 'D'])
        >>> # we need to compute closure, otherwise this set of independencies doesn't
        ... # admit a faithful representation:
        ... ind = ind.closure()
        >>> skel, sep_sets = ConstraintBasedEstimator.build_skeleton("ABCD", ind)
        >>> print(skel.edges())
        [('A', 'D'), ('B', 'D'), ('C', 'D')]

        >>> # build skeleton from d-seperations of DAG:
        ... model = DAG([('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')])
        >>> skel, sep_sets = ConstraintBasedEstimator.build_skeleton(model.nodes(), model.get_independencies())
        >>> print(skel.edges())
        [('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')]
        """
        graph = UndirectedGraph(combinations(self.nodes, 2))
        lim_neighbors = 0
        separating_sets = dict()
        while not all(
            [len(list(graph.neighbors(node))) < lim_neighbors for node in self.nodes]
        ):
            for node in self.nodes:
                for neighbor in list(graph.neighbors(node)):
                    # search if there is a set of neighbors (of size lim_neighbors)
                    # that makes X and Y independent:
                    for separating_set in combinations(
                        set(graph.neighbors(node)) - set([neighbor]), lim_neighbors
                    ):
                        if self.ci_test.test_independence(
                            node, neighbor, separating_set
                        ):
                            separating_sets[
                                frozenset((node, neighbor))
                            ] = separating_set
                            graph.remove_edge(node, neighbor)
                            break
            lim_neighbors += 1

        return graph, separating_sets

    @staticmethod
    def skeleton_to_pdag(skel, separating_sets):
        """Orients the edges of a graph skeleton based on information from
        `separating_sets` to form a DAG pattern (DAG).

        Parameters
        ----------
        skel: UndirectedGraph
            An undirected graph skeleton as e.g. produced by the
            estimate_skeleton method.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation)

        Returns
        -------
        pdag: DAG
            An estimate for the DAG pattern of the BN underlying the data. The
            graph might contain some nodes with both-way edges (X->Y and Y->X).
            Any completion by (removing one of the both-way edges for each such
            pair) results in a I-equivalent Bayesian network DAG.

        References
        ----------
        Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf


        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> data = pd.DataFrame(np.random.randint(0, 4, size=(5000, 3)), columns=list('ABD'))
        >>> data['C'] = data['A'] - data['B']
        >>> data['D'] += data['A']
        >>> c = ConstraintBasedEstimator(data)
        >>> pdag = c.skeleton_to_pdag(*c.estimate_skeleton())
        >>> pdag.edges() # edges: A->C, B->C, A--D (not directed)
        [('B', 'C'), ('A', 'C'), ('A', 'D'), ('D', 'A')]
        """

        pdag = skel.to_directed()
        node_pairs = combinations(pdag.nodes(), 2)

        # 1) for each X-Z-Y, if Z not in the separating set of X,Y, then orient edges as X->Z<-Y
        # (Algorithm 3.4 in Koller & Friedman PGM, page 86)
        for X, Y in node_pairs:
            if not skel.has_edge(X, Y):
                for Z in set(skel.neighbors(X)) & set(skel.neighbors(Y)):
                    if Z not in separating_sets[frozenset((X, Y))]:
                        pdag.remove_edges_from([(Z, X), (Z, Y)])

        progress = True
        while progress:  # as long as edges can be oriented (removed)
            num_edges = pdag.number_of_edges()

            # 2) for each X->Z-Y, orient edges to Z->Y
            for X, Y in node_pairs:
                for Z in (set(pdag.successors(X)) - set(pdag.predecessors(X))) & (
                    set(pdag.successors(Y)) & set(pdag.predecessors(Y))
                ):
                    pdag.remove(Y, Z)

            # 3) for each X-Y with a directed path from X to Y, orient edges to X->Y
            for X, Y in node_pairs:
                for path in nx.all_simple_paths(pdag, X, Y):
                    is_directed = True
                    for src, dst in path:
                        if pdag.has_edge(dst, src):
                            is_directed = False
                    if is_directed:
                        pdag.remove(Y, X)
                        break

            # 4) for each X-Z-Y with X->W, Y->W, and Z-W, orient edges to Z->W
            for X, Y in node_pairs:
                for Z in (
                    set(pdag.successors(X))
                    & set(pdag.predecessors(X))
                    & set(pdag.successors(Y))
                    & set(pdag.predecessors(Y))
                ):
                    for W in (
                        (set(pdag.successors(X)) - set(pdag.predecessors(X)))
                        & (set(pdag.successors(Y)) - set(pdag.predecessors(Y)))
                        & (set(pdag.successors(Z)) & set(pdag.predecessors(Z)))
                    ):
                        pdag.remove(W, Z)

            progress = num_edges > pdag.number_of_edges()

        return pdag
