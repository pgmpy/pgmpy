#!/usr/bin/env python

from warnings import warn
from itertools import combinations
import networkx as nx

from pgmpy.base import UndirectedGraph
from pgmpy.base import DAG
from pgmpy.estimators import StructureEstimator
from pgmpy.independencies import Independencies, IndependenceAssertion
from pgmpy.estimators.CITests import chi_square, pearsonr, independence_match


class PC(StructureEstimator):
    def __init__(self, data=None, **kwargs):
        """
        Class for constraint-based estimation of DAGs using the PC algorithm
        from a given data set.  Identifies (conditional) dependencies in data
        set using chi_square dependency test and uses the PC algorithm to
        estimate a DAG pattern that satisfies the identified dependencies. The
        DAG pattern can then be completed to a faithful DAG, if possible.

        Parameters
        ----------
        data: pandas DataFrame object
            datafame object where each column represents one variable.  (If some
            values in the data are missing the data cells should be set to
            `numpy.NaN`.  Note that pandas converts each column containing
            `numpy.NaN`s to dtype `float`.)

        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques,
            2009, Section 18.2
        [2] Neapolitan, Learning Bayesian Networks, Section 10.1.2 for the PC algorithm (page 550), http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        """
        super(PC, self).__init__(data=data, **kwargs)

    def estimate(
        self,
        variant="stable",
        ci_test="chi_square",
        max_cond_vars=5,
        return_type="dag",
        significance_level=0.01,
        n_jobs=-1,
        **kwargs,
    ):
        """
        Estimates a DAG/PDAG from the given dataset using the PC algorithm which
        is a constraint-based structure learning algorithm[1]. The independencies
        in the dataset are identified by doing statistical independece test. This
        method returns a DAG/PDAG structure which is faithful to the independencies
        implied by the dataset

        Parameters
        ----------
        variant: str (one of "orig", "stable", "parallel")
            The variant of PC algorithm to run.
                "orig": The original PC algorithm. Might not give the same
                        results in different runs but does less independence
                        tests compared to stable.
                "stable": Gives the same result in every run but does needs to
                        do more statistical independence tests.
                "parallel": Parallel version of PC Stable. Can run on multiple
                        cores with the same result on each run.

        ci_test: str or fun
            The statistical test to use for testing conditional independence in
            the dataset. If `str` values should be one of:
                "independence_match": If using this option, an additional parameter
                        `independencies` must be specified.
                "chi_square": Uses the Chi-Square independence test. This works
                        only for discrete datasets.
                "pearsonr": Uses the pertial correlation based on pearson
                        correlation coefficient to test independence. This works
                        only for continuous datasets.

        max_cond_vars: int
            The maximum number of conditional variables allowed to do the statistical
            test with.

        return_type: str (one of "dag", "cpdag", "pdag")
            The type of structure to return. If `return_type=pdag` or `return_type=cpdag`,
            a partially directed structure is returned. If `return_type=dag`, a
            fully directed structure is returned if it is possible to orient all
            the edges.

        significance_level: float (default: 0.01)
            The statistical tests use this value to compare with the p-value of
            the test to decide whether the tested variables are independent or
            not. Different tests can treat this parameter differently:
                1. Chi-Square: If p-value > significance_level, it assumes that the
                    independence condition satisfied in the data.
                2. pearsonr: If p-value > significance_level, it assumes that the
                    independence condition satisfied in the data.

        Returns
        -------
        model: DAG-instance or PDAG-instance
                The estimated model structure, can be a partially directed graph (PDAG)
                or a fully directed graph (DAG) depending on the value of `return_type`
                argument.

        References
        ----------
        [1] Original PC: P. Spirtes, C. Glymour, and R. Scheines, Causation,
                    Prediction, and Search, 2nd ed. Cambridge, MA: MIT Press, 2000.
        [2] Stable PC:  D. Colombo and M. H. Maathuis, “A modification of the PC algorithm
                    yielding order-independent skeletons,” ArXiv e-prints, Nov. 2012.
        [3] Parallel PC: Le, Thuc, et al. "A fast PC algorithm for high dimensional causal
                    discovery with multi-core PCs." IEEE/ACM transactions on computational
                    biology and bioinformatics (2016).

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
        >>> c = PC(data)
        >>> model = c.estimate()
        >>> print(model.edges())
        [('Z', 'sum'), ('X', 'sum'), ('Y', 'sum')]
        """
        if variant not in ("orig", "stable", "parallel"):
            raise ValueError(
                f"variant must be one of: orig, stable, or parallel. Got: {variant}"
            )

        # Step 1: Run the PC algorithm to build the skeleton and get the separating sets.
        skel, separating_sets = self.build_skeleton(ci_test, max_cond_vars,
                                                    significance_level,
                                                    variant, kwargs)

        # Step 2: Orient the edges based on build the PDAG/CPDAG.
        pdag = self.skeleton_to_pdag(skel, separating_sets)

        # Step 3: Either return the CPDAG or fully orient the edges to build a DAG.
        if return_type.lower() in ("pdag", "cpdag"):
            return pdag
        elif return_type.lower() == "dag":
            return pdag.to_dag()
        else:
            raise ValueError(
                f"return_type must be one of: dag, pdag, or cpdag. Got: {return_type}"
            )

    def build_skeleton(
        self, ci_test="chi_square", max_cond_vars=5, significance_level=0.01, variant='stable', **kwargs
    ):
        """
        Estimates a graph skeleton (UndirectedGraph) from a set of independencies
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

        # Initialize initial values and structures.
        lim_neighbors = 0
        separating_sets = dict()
        if ci_test == "chi_square":
            ci_test = chi_square
        elif ci_test == "pearsonr":
            ci_test = pearsonr
        elif callable(ci_test):
            ci_test = ci_test
        else:
            raise ValueError("CI test must be this this")

        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=self.variables, create_using=nx.Graph)

        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while (not all([len(list(graph.neighbors(var))) < lim_neighbors for var in self.variables])):

            # Step 2: Iterate over the edges and find a conditioning set of 
            # size `lim_neighbors` which makes u and v independent.
            if variant == 'orig':
                for (u, v) in graph.edges():
                    for separating_set in combinations(
                        set(graph.neighbors(u)) - set([v]), lim_neighbors
                    ):
                        # If a conditioning set exists remove the edge, store the separating set
                        # and move on to finding conditioning set for next edge.
                        if ci_test(u, v, separating_set):
                            separating_sets[(u, v)] = separating_set
                            graph.remove_edge(u, v)
                            break

            elif variant == 'stable':
                # In case of stable, precompute neighbors as this is the stable algorithm.
                neighbors = {node: set(graph[node]) for node in graph.nodes()}
                for (u, v) in graph.edges():
                    for separating_set in combinations(
                        neighbors[u] - set([v]), lim_neighbors
                    ):
                        # If a conditioning set exists remove the edge, store the 
                        # separating set and move on to finding conditioning set for next edge.
                        if ci_test(u, v, separating_set):
                            separating_sets[(u, v)] = separating_set
                            graph.remove_edge(u, v)
                            break

            elif variant == 'parallel':
                neighbors = {node: set(graph[node]) for node in graph.nodes()}
                for (u, v) in graph.edges():
                    # In case of parallel, precompute neighbors as this is the stable algorithm.
                    for separating_set in combinations(
                        neighbors[u] - set([v]), lim_neighbors
                    ):
                        # If a conditioning set exists remove the edge, store the separating set
                        # and move on to finding conditioning set for next edge.
                        if ci_test(u, v, separating_set):
                            separating_sets[(u, v)] = separating_set
                            graph.remove_edge(u, v)
                            break

            else:
                raise ValueError(f"variant must be one of (orig, stable, parallel). Got: {variant}")

            # Step 3: After iterating over all the edges, expand the search space by increasing the size
            #         of conditioning set by 1.
            if lim_neighbors >= max_cond_vars:
                warn("Reached maximum number of allowed conditional variables. Exiting")
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
