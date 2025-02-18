#!/usr/bin/env python

from itertools import chain, combinations, permutations

import networkx as nx
from joblib import Parallel, delayed
from tqdm.auto import tqdm

from pgmpy import config
from pgmpy.base import PDAG
from pgmpy.estimators import ExpertKnowledge, StructureEstimator
from pgmpy.estimators.CITests import get_ci_test
from pgmpy.global_vars import logger


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
        variant="parallel",
        ci_test="chi_square",
        return_type="pdag",
        significance_level=0.01,
        max_cond_vars=5,
        expert_knowledge=None,
        enforce_expert_knowledge=False,
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

        expert_knowledge: pgmpy.estimators.ExpertKnowledge instance
            Expert knowledge to be used with the algorithm. Expert knowledge
            includes required/forbidden edges in the final graph, temporal
            information about the variables etc. Please refer
            pgmpy.estimators.ExpertKnowledge class for more details.

        enforce_expert_knowledge: boolean (default: False)
            If True, the algorithm modifies the search space according to the
            edges specified in expert knowledge object. This implies the following:
                1. For every edge (u, v) specified in `forbidden_edges`, there will
                    be no edge between u and v.
                2. For every edge (u, v) specified in `required_edges`, one of the
                    following would be present in the final model: u -> v, u <-
                    v, or u - v (if CPDAG is returned).

            If False, the algorithm attempts to make the edge orientations as
            specified by expert knowledge after learning the skeleton. This
            implies the following:
                1. For every edge (u, v) specified in `forbidden_edges`, the final
                    graph would have either v <- u or no edge except if u -> v is part
                    of a collider structure in the learned skeleton.
                2. For every edge (u, v) specified in `required_edges`, the final graph
                    would either have u -> v or no edge except if v <- u is part of a
                    collider structure in the learned skeleton.

        Returns
        -------
        Estimated model: pgmpy.base.DAG, pgmpy.base.PDAG, or tuple(networkx.UndirectedGraph, dict)
            The estimated model structure:
                1. Partially Directed Graph (PDAG) if `return_type='pdag'` or `return_type='cpdag'`.
                2. Directed Acyclic Graph (DAG) if `return_type='dag'`.
                3. (nx.Graph, separating sets) if `return_type='skeleton'`.

        References
        ----------
        [1] Original PC: P. Spirtes, C. Glymour, and R. Scheines, Causation,
                    Prediction, and Search, 2nd ed. Cambridge, MA: MIT Press, 2000.
        [2] Stable PC:  D. Colombo and M. H. Maathuis, “A modification of the PC algorithm
                    yielding order-independent skeletons,” ArXiv e-prints, Nov. 2012.
        [3] Parallel PC: Le, Thuc, et al. "A fast PC algorithm for high dimensional causal
                    discovery with multi-core PCs." IEEE/ACM transactions on computational
                    biology and bioinformatics (2016).
        [4] Expert Knowledge: Meek, Christopher. "Causal inference and causal
                explanation with background knowledge." arXiv preprint arXiv:1302.4972
                (2013).

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

        ci_test = get_ci_test(
            ci_test, full=True, data=self.data, independencies=self.independencies
        )

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        # Step 1: Run the PC algorithm to build the skeleton and get the separating sets.
        skel, separating_sets = self.build_skeleton(
            ci_test=ci_test,
            significance_level=significance_level,
            variant=variant,
            n_jobs=n_jobs,
            expert_knowledge=expert_knowledge,
            enforce_expert_knowledge=enforce_expert_knowledge,
            show_progress=show_progress,
            **kwargs,
        )

        if return_type.lower() == "skeleton":
            return skel, separating_sets

        # Step 2: Orient the edges based on collider structures.
        pdag = self.orient_colliders(
            skel, separating_sets, expert_knowledge.temporal_ordering
        )

        # Step 3: Either return the CPDAG, integrate expert knowledge or fully orient the edges to build a DAG.
        if expert_knowledge.temporal_order != [[]]:
            pdag = expert_knowledge.apply_expert_knowledge(pdag)
            pdag = self.apply_orientation_rules(pdag, apply_r4=True)

        elif not enforce_expert_knowledge:
            pdag = self.apply_orientation_rules(pdag)
            pdag = expert_knowledge.apply_expert_knowledge(pdag)
            pdag = self.apply_orientation_rules(pdag, apply_r4=True)

        else:
            pdag = self.apply_orientation_rules(pdag)

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
        variant="stable",
        ci_test="chi_square",
        significance_level=0.01,
        max_cond_vars=5,
        expert_knowledge=None,
        enforce_expert_knowledge=False,
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
        ci_test = get_ci_test(ci_test, full=True, data=None)

        if expert_knowledge is None:
            expert_knowledge = ExpertKnowledge()

        if show_progress and config.SHOW_PROGRESS:
            pbar = tqdm(total=max_cond_vars)
            pbar.set_description("Working for n conditional variables: 0")

        # Step 1: Initialize a fully connected undirected graph
        graph = nx.complete_graph(n=self.variables, create_using=nx.Graph)
        temporal_ordering = expert_knowledge.temporal_ordering
        if enforce_expert_knowledge:
            graph.remove_edges_from(expert_knowledge.forbidden_edges)

        # Exit condition: 1. If all the nodes in graph has less than `lim_neighbors` neighbors.
        #             or  2. `lim_neighbors` is greater than `max_conditional_variables`.
        while not all(
            [len(list(graph.neighbors(var))) < lim_neighbors for var in self.variables]
        ):
            # Step 2: Iterate over the edges and find a conditioning set of
            # size `lim_neighbors` which makes u and v independent.
            if variant == "orig":
                for u, v in graph.edges():
                    if (enforce_expert_knowledge is False) or (
                        (u, v) not in expert_knowledge.required_edges
                    ):
                        for separating_set in PC._get_potential_sepsets(
                            u, v, temporal_ordering, graph, lim_neighbors
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
                    if (enforce_expert_knowledge is False) or (
                        (u, v) not in expert_knowledge.required_edges
                    ):
                        for separating_set in PC._get_potential_sepsets(
                            u, v, temporal_ordering, graph, lim_neighbors
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
                    for separating_set in PC._get_potential_sepsets(
                        u, v, temporal_ordering, graph, lim_neighbors
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
                    delayed(_parallel_fun)(u, v)
                    for (u, v) in graph.edges()
                    if (enforce_expert_knowledge is False)
                    or ((u, v) not in expert_knowledge.required_edges)
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
    def _get_potential_sepsets(u, v, temporal_ordering, graph, lim_neighbors):
        """
        Return the temporally consistent superset of separating set of u, v.

        The temporal order (if specified) of the superset can only be smaller
        ("earlier") than the particular node. The neighbors of 'u' satisfying
        this condition are returned.

        Parameters
        ----------
        u: variable
            The node whose neighbors are being considered for separating set.

        v: variable
            The node along with u whose separating set is being calculated.

        temporal_ordering: dict
            The temporal ordering of variables according to prior knowledgee.

        graph: UndirectedGraph
            The graph where separating sets are being calculated for the edges.

        lim_neighbors: int
            The maximum number of neighbours (conditioning variables) for u, v.

        Returns
        --------
        separating_set: set
            Set containing the superset of separating set of u, v.
        """
        separating_set_u = set(graph.neighbors(u))
        separating_set_v = set(graph.neighbors(v))
        separating_set_u.discard(v)
        separating_set_v.discard(u)

        if temporal_ordering != dict():
            max_order = min(temporal_ordering[u], temporal_ordering[u])
            for neigh in list(separating_set_u):
                if temporal_ordering[neigh] > max_order:
                    separating_set_u.discard(neigh)

            for neigh in list(separating_set_v):
                if temporal_ordering[neigh] > max_order:
                    separating_set_v.discard(neigh)

        return chain(
            combinations(separating_set_u, lim_neighbors),
            combinations(separating_set_v, lim_neighbors),
        )

    @staticmethod
    def _check_incoming_edges(pdag, u, v):
        "Used for checking whether a new v-structure is getting formed"
        for predecessor in pdag.predecessors(v):
            if (
                not pdag.has_edge(
                    v, predecessor
                )  # this ignores bidirected edges of 'v'
                and not pdag.has_edge(
                    predecessor, u
                )  # prevent the case (by returning true) when a new unshielded
                and not pdag.has_edge(
                    u, predecessor
                )  # collider may form at 'v'  i.e. predecessor--> v <--u
            ):
                return True
        return False

    @staticmethod
    def orient_colliders(skeleton, separating_sets, temporal_ordering=dict()):
        """
        Orients the edges that form v-structures in a graph skeleton
        based on information from `separating_sets` to form a DAG pattern (PDAG).

        Parameters
        ----------
        skeleton: nx.Graph
            An undirected graph skeleton as e.g. produced by the
            estimate_skeleton method.

        separating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            separating set ("witnessing set") of variables that makes them
            conditionally independent.

        Returns
        -------
        Model after edge orientation: pgmpy.base.PDAG
            An estimate for the DAG pattern of the BN underlying the data. The
            graph might contain some nodes with both-way edges (X->Y and Y->X).
            Any completion by (removing one of the both-way edges for each such
            pair) results in a I-equivalent Bayesian network DAG.

        References
        ----------
        [1] Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm
                10.2 (page 550)
        [2] http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import PC
        >>> data = pd.DataFrame(np.random.randint(0, 4, size=(5000, 3)), columns=list('ABD'))
        >>> data['C'] = data['A'] - data['B']
        >>> data['D'] += data['A']
        >>> c = PC(data)
        >>> pdag = c.orient_colliders(*c.build_skeleton())
        >>> pdag.edges() # edges: A->C, B->C, A--D (not directed)
        OutEdgeView([('B', 'C'), ('A', 'C'), ('A', 'D'), ('D', 'A')])
        """

        pdag = skeleton.to_directed()
        node_pairs = list(permutations(sorted(pdag.nodes()), 2))

        # 1) for each X-Z-Y, if Z not in the separating set of X,Y, then orient edges
        # as X->Z<-Y (Algorithm 3.4 in Koller & Friedman PGM, page 86)
        for pair in node_pairs:
            X, Y = pair
            if not skeleton.has_edge(X, Y):
                for Z in set(skeleton.neighbors(X)) & set(skeleton.neighbors(Y)):
                    if Z not in separating_sets[frozenset((X, Y))]:
                        if (temporal_ordering == dict()) or (
                            (temporal_ordering[Z] >= temporal_ordering[X])
                            and (temporal_ordering[Z] >= temporal_ordering[Y])
                        ):
                            pdag.remove_edges_from([(Z, X), (Z, Y)])

        edges = set(pdag.edges())
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))

        pdag_oriented = PDAG(
            directed_ebunch=directed_edges, undirected_ebunch=undirected_edges
        )
        pdag_oriented.add_nodes_from(pdag.nodes())

        return pdag_oriented

    @staticmethod
    def apply_orientation_rules(pdag, apply_r4=False):
        """Orients the edges of a graph skeleton based on information from
        `separating_sets` to form a DAG pattern (CPDAG/MPDAG).

        Parameters
        ----------
        pdag: pgmpy.base.PDAG
            A  partial DAG produced by orienting v-structures in
            the skeleton.

        apply_r4: boolean
            If true, use Rule 4 of Meek's rules to integrate background knowledge into
            the phase of orienting edges. Defaults to False.

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
        >>> data = pd.DataFrame(np.random.randint(0, 4, size=(5000, 4)), columns=list('ABDE'))
        >>> data['C'] = data['A'] - data['B']
        >>> data['D'] += data['A']
        >>> data['E'] += data['C']
        >>> c = PC(data)
        >>> pdag = c.orient_colliders(*c.build_skeleton())
        >>> pdag.edges() # edges: A->C, B->C, A--D (not directed), C--E (not directed)
        OutEdgeView([('B', 'C'), ('C', 'E'), ('A', 'C'), ('A', 'D'), ('E', 'C'), ('D', 'A')])
        >>> pdag = c.apply_orientation_rules(pdag)
        >>> pdag.edges()
        OutEdgeView([('C', 'E'), ('B', 'C'), ('A', 'C'), ('A', 'D'), ('D', 'A')])
        """

        node_pairs = list(permutations(sorted(pdag.nodes()), 2))

        progress = True
        while progress:  # as long as edges can be oriented (removed)
            num_edges = pdag.number_of_edges()

            # 1) for each X->Z-Y, orient edges to Z->Y
            # (Explanation in Koller & Friedman PGM, page 88)
            for pair in node_pairs:
                X, Y = pair
                if not pdag.has_edge(X, Y) and not pdag.has_edge(Y, X):
                    for Z in (set(pdag.successors(X)) - set(pdag.predecessors(X))) & (
                        set(pdag.successors(Y)) & set(pdag.predecessors(Y))
                    ):
                        if not PC._check_incoming_edges(pdag, Z, Y):
                            any_directed = False
                            for path in nx.all_simple_paths(pdag, Y, Z):
                                is_directed = True
                                for src, dst in list(zip(path, path[1:])):
                                    if pdag.has_edge(dst, src):
                                        is_directed = False
                                if is_directed:
                                    any_directed = True
                            if not any_directed:
                                pdag.remove_edge(Y, Z)

            # 2) for each X-Y with a directed path from X to Y, orient edges to X->Y
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

            # 3) for each X-Z-Y with X->W, Y->W, and Z-W, orient edges to Z->W
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
                        # if not PC.check_incoming_edges(pdag, W):
                        pdag.remove_edge(W, Z)

            # This rule (rule 4 in Meek's rules) is only used in the case of a
            #   knowledge base of required and forbidden edges.
            # For a comprehensive explanation, check out Meek's original paper
            # - https://doi.org/10.48550/arXiv.1302.4972
            if apply_r4 is not False:
                # 4) for each X-Z-Y with Z-Y->W and Z...W->X, orient edges to Z->X
                #    the dotted line above represents the possibility of either a
                #    directed or an undirected edge
                for pair in node_pairs:
                    X, Y = pair
                    for Z in (
                        set(pdag.successors(X))
                        & set(pdag.predecessors(X))
                        & set(pdag.predecessors(Y))
                        & set(pdag.successors(Y))
                    ):
                        for W in (
                            (set(pdag.successors(Y)) - set(pdag.predecessors(Y)))
                            & (set(pdag.predecessors(Z)) | set(pdag.successors(Z)))
                            & set(pdag.predecessors(X))
                        ):
                            pdag.remove_edge(X, Z)

            progress = num_edges > pdag.number_of_edges()

        edges = set(pdag.edges())
        undirected_edges = []
        directed_edges = []
        for u, v in edges:
            if (v, u) in edges:
                undirected_edges.append((u, v))
            else:
                directed_edges.append((u, v))

        return PDAG(directed_ebunch=directed_edges, undirected_ebunch=undirected_edges)
