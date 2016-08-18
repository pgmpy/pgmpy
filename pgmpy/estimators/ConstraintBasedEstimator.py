#!/usr/bin/env python
import numpy as np
import pandas as pd
from warnings import warn
from itertools import combinations
from scipy.stats import chisquare

from pgmpy.base import UndirectedGraph
from pgmpy.models import BayesianModel
from pgmpy.estimators import StructureEstimator
from pgmpy.independencies import Independencies, IndependenceAssertion


class ConstraintBasedEstimator(StructureEstimator):
    def __init__(self, data, **kwargs):
        """
        Class for constraint-based estimation of BayesianModels from a given
        data set. Identifies (conditional) dependencies in data set using
        chi_square dependency test and uses the PC algorithm to estimate a DAG
        pattern that satisfies the identified dependencies. The DAG pattern can
        then be completed to a faithful BayesianModel, if possible.

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
        super(ConstraintBasedEstimator, self).__init__(data, **kwargs)

    def estimate(self, p_value=0.05):
        """
        Estimates a BayesianModel for the data set, using the PC contraint-based
        structure learning algorithm. Independencies are identified from the
        data set using a chi-squared statistic with the acceptance threshold of
        `p_value`. PC identifies a partially directed acyclic graph (PDAG), given
        that the tested independencies admit a faithful Bayesian network representation.
        This method returns a BayesianModel that is a completion of this PDAG.

        Parameters
        ----------
        p_value: float, default: 0.05
            A significance level to use for conditional independence tests in the data set.
            The p_value is the threshold probability of falsely rejecting the hypothesis
            that variables are conditionally dependent.

            The lower `p_value`, the more likely we are to reject dependencies,
            resulting in a sparser graph.

        Returns
        -------
        model: BayesianModel()-instance
            An estimate for the BayesianModel for the data set (not yet parametrized).

        Reference
        ---------
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

        skel, seperating_sets = self.estimate_skeleton(p_value)
        pdag = self.skeleton_to_pdag(skel, seperating_sets)
        model = self.pdag_to_dag(pdag)
        return model

    def estimate_skeleton(self, p_value=0.05):
        """Estimates a graph skeleton (UndirectedGraph) for the data set.
        Uses the build_skeleton method (PC algorithm); independencies are
        determined using a chisquare statistic with the acceptance threshold
        of `p_value`. Returns a tuple `(skeleton, seperating_sets).

        Parameters
        ----------
        p_value: float, default: 0.05
            A significance level to use for conditional independence tests in
            the data set. The p_value is the threshold probability of falsely
            rejecting the hypothesis that variables are conditionally dependent.

            The lower `p_value`, the more likely we are to reject dependencies,
            resulting in a sparser graph.

        Returns
        -------
        skeleton: UndirectedGraph
            An estimate for the undirected graph skeleton of the BN underlying the data.
        seperating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            seperating set of variables that makes then conditionally independent.
            (needed for edge orientation procedures)

        Reference
        ---------
        [1] Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [2] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence

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
        >>>
        """

        nodes = self.state_names.keys()

        def is_independent(X, Y, Zs):
            """Computes p-value of chi2 independence test. If it is larger
            than or equal to the provided threshold `p_value`, return `True`
            (thereby rejecting the hypothesis that variables X and Y are
            dependent given a list of variables Zs). Otherwise return `False`.
            """
            return self.test_conditional_independence(X, Y, Zs) >= p_value

        return self.build_skeleton(nodes, is_independent)

    @staticmethod
    def estimate_from_independencies(nodes, independencies):
        """Estimates a BayesianModel from an Independencies()-object or a
        decision function for conditional independencies. This requires that
        the set of independencies admits a faithful representation (e.g. is a
        set of d-seperation for some BN or is closed under the semi-graphoid
        axioms). See `build_skeleton`, `skeleton_to_pdag`, `pdag_to_dag` for
        details.

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
        model: BayesianModel()-instance

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.independencies import Independencies

        >>> ind = Independencies(['B', 'C'], ['A', ['B', 'C'], 'D'])
        >>> ind = ind.closure()
        >>> skel = ConstraintBasedEstimator.estimate_from_independencies("ABCD", ind)
        >>> print(skel.edges())
        [('B', 'D'), ('A', 'D'), ('C', 'D')]

        >>> model = BayesianModel([('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')])
        >>> skel = ConstraintBasedEstimator.estimate_from_independencies(model.nodes(), model.get_independencies())
        >>> print(skel.edges())
        [('B', 'C'), ('A', 'C'), ('C', 'E'), ('D', 'B')]
        >>> # note that ('D', 'B') is flipped compared to the original network;
        >>> # Both networks belong to the same PDAG/are I-equivalent
        """

        skel, seperating_sets = ConstraintBasedEstimator.build_skeleton(nodes, independencies)
        pdag = ConstraintBasedEstimator.skeleton_to_pdag(skel, seperating_sets)
        dag = ConstraintBasedEstimator.pdag_to_dag(pdag)

        return dag

    @staticmethod
    def pdag_to_dag(pdag):
        """Completes a PDAG to a DAG, without adding v-structures, if such a
        completion exists. If no faithful extension is possible, some fully
        oriented DAG that corresponds to the PDAG is returned and a warning is
        generated. This is a static method.

        Parameters
        ----------
        pdag: DirectedGraph
            A directed acyclic graph pattern, consisting in (acyclic) directed edges
            as well as "undirected" edges, represented as both-way edges between
            nodes.

        Returns
        -------
        dag: BayesianModel
            A faithful orientation of pdag, if one exists. Otherwise any
            fully orientated DAG/BayesianModel with the structure of pdag.

        References
        ----------
        [1] Chickering, Learning Equivalence Classes of Bayesian-Network Structures,
            2002; See page 454 (last paragraph) for the algorithm pdag_to_dag
            http://www.jmlr.org/papers/volume2/chickering02a/chickering02a.pdf
        [2] Dor & Tarsi, A simple algorithm to construct a consistent extension
            of a partially oriented graph, 1992,
            http://ftp.cs.ucla.edu/pub/stat_ser/r185-dor-tarsi.pdf

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.base import DirectedGraph
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> data = pd.DataFrame(np.random.randint(0, 4, size=(5000, 3)), columns=list('ABD'))
        >>> data['C'] = data['A'] - data['B']
        >>> data['D'] += data['A']
        >>> c = ConstraintBasedEstimator(data)
        >>> pdag = c.skeleton_to_pdag(c.estimate_skeleton())
        >>> pdag.edges()
        [('B', 'C'), ('D', 'A'), ('A', 'D'), ('A', 'C')]
        >>> c.pdag_to_dag(pdag).edges()
        [('B', 'C'), ('A', 'D'), ('A', 'C')]

        >>> # pdag_to_dag is static:
        ... pdag1 = DirectedGraph([('A', 'B'), ('C', 'B'), ('C', 'D'), ('D', 'C'), ('D', 'A'), ('A', 'D')])
        >>> ConstraintBasedEstimator.pdag_to_dag(pdag1).edges()
        [('D', 'C'), ('C', 'B'), ('A', 'B'), ('A', 'D')]

        >>> # example of a pdag with no faithful extension:
        ... pdag2 = DirectedGraph([('A', 'B'), ('A', 'C'), ('B', 'C'), ('C', 'B')])
        >>> ConstraintBasedEstimator.pdag_to_dag(pdag2).edges()
        UserWarning: PDAG has no faithful extension (= no oriented DAG with the same v-structures as PDAG).
        Remaining undirected PDAG edges oriented arbitrarily.
        [('B', 'C'), ('A', 'B'), ('A', 'C')]
        """

        dag = BayesianModel()
        dag.add_nodes_from(pdag.nodes())

        # add already directed edges of pdag to dag
        for X, Y in pdag.edges():
            if not pdag.has_edge(Y, X):
                dag.add_edge(X, Y)

        while pdag.number_of_nodes() > 0:
            # find node with (1) no directed outgoing edges and
            #                (2) the set of undirected neighbors is either empty or
            #                    undirected neighbors + parents of X are a clique
            found = False
            for X in pdag.nodes():
                directed_outgoing_edges = set(pdag.successors(X)) - set(pdag.predecessors(X))
                undirected_neighbors = set(pdag.successors(X)) & set(pdag.predecessors(X))
                neighbors_are_clique = all((pdag.has_edge(Y, Z)
                                            for Z in pdag.predecessors(X)
                                            for Y in undirected_neighbors if not Y == Z))

                if not directed_outgoing_edges and \
                        (not undirected_neighbors or neighbors_are_clique):
                    found = True
                    # add all edges of X as outgoing edges to dag
                    for Y in pdag.predecessors(X):
                        dag.add_edge(Y, X)

                    pdag.remove_node(X)
                    break

            if not found:
                warn("PDAG has no faithful extension (= no oriented DAG with the " +
                     "same v-structures as PDAG). Remaining undirected PDAG edges " +
                     "oriented arbitrarily.")
                for X, Y in pdag.edges():
                    if not dag.has_edge(Y, X):
                        try:
                            dag.add_edge(X, Y)
                        except ValueError:
                            pass
                break

        return dag

    @staticmethod
    def model_to_pdag(model):
        """Construct the DAG pattern (representing the I-equivalence class) for
        a given BayesianModel. This is the "inverse" to pdag_to_dag.
        """

        if not isinstance(pdag, BayesianModel):
            TypeError("'model' must be a BayesianModel()-instance.")

        skel, seperating_sets = ConstraintBasedEstimator.build_skeleton(
                                    model.nodes(),
                                    model.get_independencies())
        pdag = ConstraintBasedEstimator.skeleton_to_pdag(skel, seperating_sets)

        return pdag

    @staticmethod
    def skeleton_to_pdag(skel, seperating_sets):
        """Orients the edges of a graph skeleton based on information from
        `seperating_sets` to form a DAG pattern (DirectedGraph).

        Parameters
        ----------
        skel: UndirectedGraph
            An undirected graph skeleton as e.g. produced by the
            estimate_skeleton()-method.
        seperating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            seperating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation)

        Returns
        -------
        pdag: DirectedGraph
            An estimate for the DAG pattern of the BN underlying the data. The
            graph might contain some nodes with both-way edges (X->Y and Y->X).
            Any completion by (removing one of the both-way edges for each such
            pair) results in a I-equivalent Bayesian network DAG.

        Reference
        ---------
        Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
        http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf


        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.base import DirectedGraph
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> data = pd.DataFrame(np.random.randint(0, 4, size=(5000, 3)), columns=list('ABD'))
        >>> data['C'] = data['A'] - data['B']
        >>> data['D'] += data['A']
        >>> c = ConstraintBasedEstimator(data)
        >>> pdag = c.skeleton_to_pdag(c.estimate_skeleton())
        >>> pdag.edges() # edges: A->C, B->C, A--D (not directed)
        [('B', 'C'), ('A', 'C'), ('A', 'D'), ('D', 'A')]
        """

        pdag = skel.to_directed()
        node_pairs = combinations(pdag.nodes(), 2)

        # 1) for each X-Z-Y, if Z not in the seperating set of X,Y, then orient edges as X->Z<-Y
        # (Algorithm 3.4 in Koller & Friedman PGM, page 86)
        for X, Y in node_pairs:
            if not skel.has_edge(X, Y):
                for Z in set(skel.neighbors(X)) & set(skel.neighbors(Y)):
                    if Z not in seperating_sets[frozenset((X, Y))]:
                        pdag.remove_edges_from([(Z, X), (Z, Y)])

        progress = True
        while progress:  # as long as edges can be oriented (removed)
            num_edges = pdag.number_of_edges()

            # 2) for each X->Z-Y, orient edges to Z->Y
            for X, Y in node_pairs:
                for Z in ((set(pdag.successors(X)) - set(pdag.predecessors(X))) &
                          (set(pdag.successors(Y)) & set(pdag.predecessors(Y)))):
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
                for Z in (set(pdag.successors(X)) & set(pdag.predecessors(X)) &
                          set(pdag.successors(Y)) & set(pdag.predecessors(Y))):
                    for W in ((set(pdag.successors(X)) - set(pdag.predecessors(X))) &
                              (set(pdag.successors(Y)) - set(pdag.predecessors(Y))) &
                              (set(pdag.successors(Z)) & set(pdag.predecessors(Z)))):
                        pdag.remove(W, Z)

            progress = num_edges > pdag.number_of_edges()

        return pdag

    @staticmethod
    def build_skeleton(nodes, independencies):
        """Estimates a graph skeleton (UndirectedGraph) from a set of independencies
        using (the first part of) the PC algorithm. The independencies can either be
        provided as an instance of the `Independencies`-class or by passing a
        decision function that decides any conditional independency assertion.
        Returns a tuple `(skeleton, seperating_sets)`.

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
        seperating_sets: dict
            A dict containing for each pair of not directly connected nodes a
            seperating set ("witnessing set") of variables that makes then
            conditionally independent. (needed for edge orientation procedures)

        Reference
        ---------
        [1] Neapolitan, Learning Bayesian Networks, Section 10.1.2, Algorithm 10.2 (page 550)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [2] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
            Section 3.4.2.1 (page 85), Algorithm 3.3

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> from pgmpy.models import BayesianModel
        >>> from pgmpy.independencies import Independencies

        >>> # build skeleton from list of independencies:
        ... ind = Independencies(['B', 'C'], ['A', ['B', 'C'], 'D'])
        >>> # we need to compute closure, otherwise this set of independencies doesn't
        ... # admit a faithful representation:
        ... ind = ind.closure()
        >>> skel, sep_sets = ConstraintBasedEstimator.build_skeleton("ABCD", ind)
        >>> print(skel.edges())
        [('A', 'D'), ('B', 'D'), ('C', 'D')]

        >>> # build skeleton from d-seperations of BayesianModel:
        ... model = BayesianModel([('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')])
        >>> skel, sep_sets = ConstraintBasedEstimator.build_skeleton(model.nodes(), model.get_independencies())
        >>> print(skel.edges())
        [('A', 'C'), ('B', 'C'), ('B', 'D'), ('C', 'E')]
        """

        nodes = list(nodes)

        if isinstance(independencies, Independencies):
            def is_independent(X, Y, Zs):
                return IndependenceAssertion(X, Y, Zs) in independencies
        elif callable(independencies):
            is_independent = independencies
        else:
            raise ValueError("'independencies' must be either Independencies-instance " +
                             "or a ternary function that decides independencies.")

        graph = UndirectedGraph(combinations(nodes, 2))
        lim_neighbors = 0
        seperating_sets = dict()
        while not all([len(graph.neighbors(node)) < lim_neighbors for node in nodes]):
            for node in nodes:
                for neighbor in graph.neighbors(node):
                    # search if there is a set of neighbors (of size lim_neighbors)
                    # that makes X and Y independent:
                    for seperating_set in combinations(set(graph.neighbors(node)) - set([neighbor]), lim_neighbors):
                        if is_independent(node, neighbor, seperating_set):
                            seperating_sets[frozenset((node, neighbor))] = seperating_set
                            graph.remove_edge(node, neighbor)
                            break
            lim_neighbors += 1

        return graph, seperating_sets

    def test_conditional_independence(self, X, Y, Zs=[]):
        """Chi-square conditional independence test.
        Tests if X is independent from Y given Zs in the data.

        This is done by comparing the observed frequencies with the expected
        frequencies if X,Y were conditionally independent, using a chisquare
        deviance statistic. The expected frequencies given independence are
        `P(X,Y,Zs) = P(X|Zs)*P(Y|Zs)*P(Zs)`. The latter term can be computed
        as `P(X,Zs)*P(Y,Zs)/P(Zs).

        Parameters
        ----------
        X: int, string, hashable object
            A variable name contained in the data set
        Y: int, string, hashable object
            A variable name contained in the data set, different from X
        Zs: list of variable names
            A list of variable names contained in the data set, different from X and Y.
            This is the seperating set that (potentially) makes X and Y independent.
            Default: []

        Returns
        -------
        p_value: float
            A significance level for the hypothesis that X and Y are dependent
            given Zs. The p_value is the probability of falsely rejecting the
            hypothesis that the variables are conditionally dependent. A low
            p_value (e.g. below 0.05 or 0.01) indicates dependence. (The lower
            the threshold for the p_value, the more likely we are to reject
            dependency, resulting in a sparser graph.)

        References
        ----------
        [1] Koller & Friedman, Probabilistic Graphical Models - Principles and Techniques, 2009
        Section 18.2.2.3 (page 789)
        [2] Neapolitan, Learning Bayesian Networks, Section 10.3 (page 600ff)
            http://www.cs.technion.ac.il/~dang/books/Learning%20Bayesian%20Networks(Neapolitan,%20Richard).pdf
        [3] Chi-square test https://en.wikipedia.org/wiki/Pearson%27s_chi-squared_test#Test_of_independence

        Examples
        --------
        >>> import pandas as pd
        >>> import numpy as np
        >>> from pgmpy.estimators import ConstraintBasedEstimator
        >>> data = pd.DataFrame(np.random.randint(0, 2, size=(50000, 4)), columns=list('ABCD'))
        >>> data['E'] = data['A'] + data['B'] + data['C']
        >>> c = ConstraintBasedEstimator(data)
        >>> print(c.test_conditional_independence('A', 'C'))  # independent
        0.9848481578
        >>> print(c.test_conditional_independence('A', 'B', 'D'))  # independent
        0.962206185665
        >>> print(c.test_conditional_independence('A', 'B', ['D', 'E']))  # dependent
        0.0
        """

        if isinstance(Zs, (frozenset, list, set, tuple,)):
            Zs = list(Zs)
        else:
            Zs = [Zs]

        # Check is sample size is sufficient. Require at least 5 samples per parameter (on average)
        # (As suggested in Spirtes et al., Causation, Prediction and Search, 2000, and also used in
        # Tsamardinos et al., The max-min hill-climbing BN structure learning algorithm, 2005, Section 4)
        num_params = ((len(self.state_names[X])-1) *
                      (len(self.state_names[Y])-1) *
                      np.prod([len(self.state_names[Z]) for Z in Zs]))
        if len(self.data) < num_params:
            warn("Insufficient data for testing {0} _|_ {1} | {2}. ".format(X, Y, Zs) +
                 "At least {0} samples recommended, {1} present.".format(num_params, len(self.data)))

        # compute actual frequency/state_count table:
        # = P(X,Y,Zs)
        XYZ_state_counts = pd.crosstab(index=self.data[X],
                                       columns=[self.data[Y]] + [self.data[Z] for Z in Zs])
        # reindex to add missing rows & columns (if some values don't appear in data)
        row_index = self.state_names[X]
        column_index = pd.MultiIndex.from_product(
                            [self.state_names[Y]] + [self.state_names[Z] for Z in Zs], names=[Y]+Zs)
        XYZ_state_counts = XYZ_state_counts.reindex(index=row_index,    columns=column_index).fillna(0)

        # compute the expected frequency/state_count table if X _|_ Y | Zs:
        # = P(X|Zs)*P(Y|Zs)*P(Zs) = P(X,Zs)*P(Y,Zs)/P(Zs)
        if Zs:
            XZ_state_counts = XYZ_state_counts.sum(axis=1, level=Zs)  # marginalize out Y
            YZ_state_counts = XYZ_state_counts.sum().unstack(Zs)      # marginalize out X
        else:
            XZ_state_counts = XYZ_state_counts.sum(axis=1)
            YZ_state_counts = XYZ_state_counts.sum()
        Z_state_counts = YZ_state_counts.sum()  # marginalize out both

        XYZ_expected = pd.DataFrame(index=XYZ_state_counts.index, columns=XYZ_state_counts.columns)
        for X_val in XYZ_expected.index:
            if Zs:
                for Y_val in XYZ_expected.columns.levels[0]:
                    XYZ_expected.loc[X_val, Y_val] = (XZ_state_counts.loc[X_val] *
                                                      YZ_state_counts.loc[Y_val] /
                                                      Z_state_counts).values
            else:
                for Y_val in XYZ_expected.columns:
                    XYZ_expected.loc[X_val, Y_val] = (XZ_state_counts.loc[X_val] *
                                                      YZ_state_counts.loc[Y_val] /
                                                      Z_state_counts)

        observed = XYZ_state_counts.values.flatten()
        expected = XYZ_expected.values.flatten()
        # remove elements where the expected value is 0;
        # this also corrects the degrees of freedom for chisquare
        observed, expected = zip(*((o, e) for o, e in zip(observed, expected) if not e == 0))

        chi2, p_value = chisquare(observed, expected)

        return p_value
