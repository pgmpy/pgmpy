#!/usr/bin/env python

from itertools import chain, combinations, permutations

import networkx as nx
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import PDAG
from pgmpy.estimators import StructureEstimator
from pgmpy.estimators.CITests import *
from pgmpy.global_vars import logger

CI_TESTS = {
    "chi_square": chi_square,
    "independence_match": independence_match,
    "pearsonr": pearsonr,
    "pillai": ci_pillai,
    "g_sq": g_sq,
    "log_likelihood": log_likelihood,
    "freeman_tuckey": freeman_tuckey,
    "modified_log_likelihood": modified_log_likelihood,
    "neyman": neyman,
    "cressie_read": cressie_read,
    "power_divergence": power_divergence,
}


class PC(StructureEstimator):
    """
    Class for constraint-based estimation of DAGs using the PC algorithm
    from a given data set.  Identifies (conditional) dependencies in data
    set using statistical independence tests and estimates a DAG pattern
    that satisfies the identified dependencies. The DAG pattern can then be
    completed to a faithful DAG, if possible.

    Parameters
    ----------
    data: pandas DataFrame object
        dataframe object where each column represents one variable.  (If some
        values in the data are missing the data cells should be set to
        `numpy.nan`.  Note that pandas converts each column containing
        `numpy.nan`s to dtype `float`.)

    References
    ----------
    [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques,
        2009, Section 18.2
    [2] Neapolitan, Learning Bayesian Networks, Section 10.1.2 for the PC algorithm (page 550), http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
    """

    def __init__(self, data=None, independencies=None, **kwargs):
        super(PC, self).__init__(data=data, independencies=independencies, **kwargs)

    def estimate(
        self,
        variant="stable",
        ci_test="chi_square",
        max_cond_vars=5,
        return_type="dag",
        significance_level=0.01,
        n_jobs=-1,
        show_progress=True,
        **kwargs,
    ):
        """
        Estimates a DAG/PDAG from the given dataset using the PC algorithm which
        is a constraint-based structure learning algorithm[1]. The independencies
        in the dataset are identified by doing statistical independece test. This
        method returns a DAG/PDAG structure which is faithful to the independencies
        implied by the dataset.

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
                "g_sq": G-test. Works only for discrete datasets.
                "log_likelihood": Log-likelihood test. Works only for discrete dataset.
                "freeman_tuckey": Freeman Tuckey test. Works only for discrete dataset.
                "modified_log_likelihood": Modified Log Likelihood test. Works only for discrete variables.
                "neyman": Neyman test. Works only for discrete variables.
                "cressie_read": Cressie Read test. Works only for discrete variables.

        max_cond_vars: int
            The maximum number of conditional variables allowed to do the statistical
            test with.

        return_type: str (one of "dag", "cpdag", "pdag", "skeleton")
            The type of structure to return.

            If `return_type=pdag` or `return_type=cpdag`: a partially directed structure
                is returned.
            If `return_type=dag`, a fully directed structure is returned if it
                is possible to orient all the edges.
            If `return_type="skeleton", returns an undirected graph along
                with the separating sets.

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
        Estimated model: pgmpy.base.DAG, pgmpy.base.PDAG, or tuple(networkx.UndirectedGraph, dict)
                The estimated model structure, can be a partially directed graph (PDAG)
                or a fully directed graph (DAG), or (Undirected Graph, separating sets)
                depending on the value of `return_type` argument.

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
        >>> from pgmpy.utils import get_example_model
        >>> from pgmpy.estimators import PC
        >>> model = get_example_model('alarm')
        >>> data = model.simulate(n_samples=1000)
        >>> est = PC(data)
        >>> model_chi = est.estimate(ci_test='chi_square')
        >>> print(len(model_chi.edges()))
        28
        >>> model_gsq, _ = est.estimate(ci_test='g_sq', return_type='skeleton')
        >>> print(len(model_gsq.edges()))
        33
        """
        # Step 0: Do checks that the specified parameters are correct, else throw meaningful error.
        if variant not in ("orig", "stable", "parallel"):
            raise ValueError(
                f"variant must be one of: orig, stable, or parallel. Got: {variant}"
            )
        elif (not callable(ci_test)) and (
            ci_test.lower() not in (list(CI_TESTS.keys()) + ["independence_match"])
        ):
            raise ValueError(
                "ci_test must be a callable or one of the tests defined in CITests.py"
            )

        if (ci_test == "independence_match") and (self.independencies is None):
            raise ValueError(
                "For using independence_match, independencies argument must be specified"
            )
        elif (ci_test in set(CI_TESTS.keys()) - set(["independence_match"])) and (
            self.data is None
        ):
            raise ValueError(
                "For using Chi Square or Pearsonr, data argument must be specified"
            )

        # Step 1: Run the PC algorithm to build the skeleton and get the separating sets.
        skel, separating_sets = self.build_skeleton(
            ci_test=ci_test,
            max_cond_vars=max_cond_vars,
            significance_level=significance_level,
            variant=variant,
            n_jobs=n_jobs,
            show_progress=show_progress,
            **kwargs,
        )

        if return_type.lower() == "skeleton":
            return skel, separating_sets

        # Step 2: Orient the edges based on build the PDAG/CPDAG.
        pdag = self.skeleton_to_pdag(skel, separating_sets)

        # Step 3: Either return the CPDAG or fully orient the edges to build a DAG.
        if self.data is not None:
            pdag.add_nodes_from(set(self.data.columns) - set(pdag.nodes()))

        if return_type.lower() in ("pdag", "cpdag"):
            return pdag
        elif return_type.lower() == "dag":
            return pdag.to_dag()
        else:
            raise ValueError(
                f"return_type must be one of: dag, pdag, cpdag, or skeleton. Got: {return_type}"
            )

    def build_skeleton(
        self,
        ci_test="chi_square",
        max_cond_vars=5,
        significance_level=0.01,
        variant="stable",
        n_jobs=-1,
        show_progress=True,
        **kwargs,
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
        Otherwise, the procedure may fail to identify the correct structure.

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
        """
        # Initialize initial values and structures.
        lim_neighbors = 0
        separating_sets = dict()
        if not callable(ci_test):
            try:
                ci_test = CI_TESTS[ci_test]
            except KeyError:
                raise ValueError(
                    f"ci_test must either be one of {list(CI_TESTS.keys())}, or a function. Got: {ci_test}"
                )

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working for n conditional variables: 0")

        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=self.variables, create_using=nx.Graph)

        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in self.variables]
        ):
            # Step 2: Iterate over the edges and find a conditioning set of
            # size `lim_neighbors` which makes u and v independent.
            if variant == "orig":
                for u, v in graph.edges():
                    for separating_set in chain(
                        combinations(set(graph.neighbors(u)) - set([v]), lim_neighbors),
                        combinations(set(graph.neighbors(v)) - set([u]), lim_neighbors),
                    ):
                        # If a conditioning set exists remove the edge, store the separating set
                        # and move on to finding conditioning set for next edge.
                        if ci_test(
                            u,
                            v,
                            separating_set,
                            data=self.data,
                            independencies=self.independencies,
                            significance_level=significance_level,
                            **kwargs,
                        ):
                            separating_sets[frozenset((u, v))] = separating_set
                            graph.remove_edge(u, v)
                            break

            elif variant == "stable":
                # In case of stable, precompute neighbors as this is the stable algorithm.
                neighbors = {node: set(graph[node]) for node in graph.nodes()}
                for u, v in graph.edges():
                    for separating_set in chain(
                        combinations(set(neighbors[u]) - set([v]), lim_neighbors),
                        combinations(set(neighbors[v]) - set([u]), lim_neighbors),
                    ):
                        # If a conditioning set exists remove the edge, store the
                        # separating set and move on to finding conditioning set for next edge.
                        if ci_test(
                            u,
                            v,
                            separating_set,
                            data=self.data,
                            independencies=self.independencies,
                            significance_level=significance_level,
                            **kwargs,
                        ):
                            separating_sets[frozenset((u, v))] = separating_set
                            graph.remove_edge(u, v)
                            break

            elif variant == "parallel":
                neighbors = {node: set(graph[node]) for node in graph.nodes()}

                def _parallel_fun(u, v):
                    for separating_set in chain(
                        combinations(set(graph.neighbors(u)) - set([v]), lim_neighbors),
                        combinations(set(graph.neighbors(v)) - set([u]), lim_neighbors),
                    ):
                        if ci_test(
                            u,
                            v,
                            separating_set,
                            data=self.data,
                            independencies=self.independencies,
                            significance_level=significance_level,
                            **kwargs,
                        ):
                            return (u, v), separating_set

                results = Parallel(n_jobs=n_jobs)(
                    delayed(_parallel_fun)(u, v) for (u, v) in graph.edges()
                )
                for result in results:
                    if result is not None:
                        (u, v), sep_set = result
                        graph.remove_edge(u, v)
                        separating_sets[frozenset((u, v))] = sep_set

            else:
                raise ValueError(
                    f"variant must be one of (orig, stable, parallel). Got: {variant}"
                )

            # Step 3: After iterating over all the edges, expand the search space by increasing the size
            #         of conditioning set by 1.
            if lim_neighbors >= max_cond_vars:
                logger.info(
                    "Reached maximum number of allowed conditional variables. Exiting"
                )
                break
            lim_neighbors += 1

            if show_progress and config.SHOW_PROGRESS:
                pbar.update(1)
                pbar.set_description(
                    f"Working for n conditional variables: {lim_neighbors}"
                )

        if show_progress and config.SHOW_PROGRESS:
            pbar.close()
        return graph, separating_sets

    @staticmethod
    def skeleton_to_pdag(skeleton, separating_sets):
        """Orients the edges of a graph skeleton based on information from
        `separating_sets` to form a DAG pattern (DAG).

        Parameters
        ----------
        skeleton: UndirectedGraph
            An undirected graph skeleton as e.g. produced by the
            estimate_skeleton method.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation)

        Returns
        -------
        Model after edge orientation: pgmpy.base.DAG
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
        >>> from pgmpy.estimators import PC
        >>> data = pd.DataFrame(np.random.randint(0, 4, size=(5000, 3)), columns=list('ABD'))
        >>> data['C'] = data['A'] - data['B']
        >>> data['D'] += data['A']
        >>> c = PC(data)
        >>> pdag = c.skeleton_to_pdag(*c.build_skeleton())
        >>> pdag.edges() # edges: A->C, B->C, A--D (not directed)
        [('B', 'C'), ('A', 'C'), ('A', 'D'), ('D', 'A')]
        """

        pdag = skeleton.to_directed()
        node_pairs = list(permutations(pdag.nodes(), 2))

        # 1) for each X-Z-Y, if Z not in the separating set of X,Y, then orient edges as X->Z<-Y
        # (Algorithm 3.4 in Koller & Friedman PGM, page 86)
        for pair in node_pairs:
            X, Y = pair
            if not skeleton.has_edge(X, Y):
                for Z in set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y)):
                    if Z not in separating_sets[frozenset((X, Y))]:
                        pdag.remove_edges_from([(Z, X), (Z, Y)])

        progress = True
        while progress:  # as long as edges can be oriented (removed)
            num_edges = pdag.number_of_edges()

            # 2) for each X->Z-Y, orient edges to Z->Y
            # (Explanation in Koller & Friedman PGM, page 88)
            for pair in node_pairs:
                X, Y = pair
                if not pdag.has_edge(X, Y):
                    for Z in (set(pdag.successors(X)) - set(pdag.predecessors(X))) & (
                        set(pdag.successors(Y)) & set(pdag.predecessors(Y))
                    ):
                        pdag.remove_edge(Y, Z)

            # 3) for each X-Y with a directed path from X to Y, orient edges to X->Y
            for pair in node_pairs:
                X, Y = pair
                if pdag.has_edge(Y, X) and pdag.has_edge(X, Y):
                    for path in nx.all_simple_paths(pdag, X, Y):
                        is_directed = True
                        for src, dst in list(zip(path, path[1:])):
                            if pdag.has_edge(dst, src):
                                is_directed = False
                        if is_directed:
                            pdag.remove_edge(Y, X)
                            break

            # 4) for each X-Z-Y with X->W, Y->W, and Z-W, orient edges to Z->W
            for pair in node_pairs:
                X, Y = pair
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
                        pdag.remove_edge(W, Z)

            progress = num_edges > pdag.number_of_edges()

        # TODO: This is temp fix to get a PDAG object.
        edges = set(pdag.edges())
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))
        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
