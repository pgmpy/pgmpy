#!/usr/bin/env python3
import itertools
from collections import defaultdict

import networkx as nx
import numpy as np

from pgmpy.base import UndirectedGraph
from pgmpy.factors.discrete import factor_product, DiscreteFactor
from pgmpy.independencies import Independencies
from pgmpy.extern.six.moves import map, range, zip


class MarkovModel(UndirectedGraph):
    """
    Base class for markov model.

    A MarkovModel stores nodes and edges with potentials

    MarkovModel holds undirected edges.

    Parameters
    ----------
    data : input graph
        Data to initialize graph.  If data=None (default) an empty
        graph is created.  The data can be an edge list, or any
        NetworkX graph object.

    Examples
    --------
    Create an empty Markov Model with no nodes and no edges.

    >>> from pgmpy.models import MarkovModel
    >>> G = MarkovModel()

    G can be grown in several ways.

    **Nodes:**

    Add one node at a time:

    >>> G.add_node('a')

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(['a', 'b'])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge('a', 'b')

    a list of edges,

    >>> G.add_edges_from([('a', 'b'), ('b', 'c')])

    If some edges connect nodes not yet in the model, the nodes
    are added automatically.  There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> 'a' in G     # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3

    Public Methods
    --------------
    add_node('node1')
    add_nodes_from(['node1', 'node2', ...])
    add_edge('node1', 'node2')
    add_edges_from([('node1', 'node2'),('node3', 'node4')])
    """

    def __init__(self, ebunch=None):
        super(MarkovModel, self).__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.factors = []

    def add_edge(self, u, v, **kwargs):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph

        Parameters
        ----------
        u,v : nodes
            Nodes can be any hashable Python object.

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> G = MarkovModel()
        >>> G.add_nodes_from(['Alice', 'Bob', 'Charles'])
        >>> G.add_edge('Alice', 'Bob')
        """
        # check that there is no self loop.
        if u != v:
            super(MarkovModel, self).add_edge(u, v, **kwargs)
        else:
            raise ValueError('Self loops are not allowed')

    def add_factors(self, *factors):
        """
        Associate a factor to the graph.
        See factors class for the order of potential values

        Parameters
        ----------
        *factor: pgmpy.factors.factors object
            A factor object on any subset of the variables of the model which
            is to be associated with the model.

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles'),
        ...                        ('Charles', 'Debbie'), ('Debbie', 'Alice')])
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[3, 2],
        ...                 values=np.random.rand(6))
        >>> student.add_factors(factor)
        """
        for factor in factors:
            if set(factor.variables) - set(factor.variables).intersection(
                    set(self.nodes())):
                raise ValueError("Factors defined on variable not in the model",
                                 factor)

            self.factors.append(factor)

    def get_factors(self):
        """
        Returns the factors that have been added till now to the graph

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                 values=np.random.rand(4))
        >>> student.add_factors(factor)
        >>> student.get_factors()
        """
        return self.factors

    def remove_factors(self, *factors):
        """
        Removes the given factors from the added factors.

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                 values=np.random.rand(4))
        >>> student.add_factors(factor)
        >>> student.remove_factors(factor)
        """
        for factor in factors:
            self.factors.remove(factor)

    def get_cardinality(self, check_cardinality=False):
        """
        Returns a dictionary with the given factors as keys and their respective
        cardinality as values.

        Parameters
        ----------
        check_cardinality: boolean, optional
            If, check_cardinality=True it checks if cardinality information
            for all the variables is availble or not. If not it raises an error.

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> factor = DiscreteFactor(['Alice', 'Bob'], cardinality=[2, 2],
        ...                 values=np.random.rand(4))
        >>> student.add_factors(factor)
        >>> student.get_cardinality()
        defaultdict(<class 'int'>, {'Bob': 2, 'Alice': 2})
        """
        cardinalities = defaultdict(int)
        for factor in self.factors:
            for variable, cardinality in zip(factor.scope(), factor.cardinality):
                cardinalities[variable] = cardinality
        if check_cardinality and len(self.nodes()) != len(cardinalities):
            raise ValueError('Factors for all the variables not defined')
        return cardinalities

    def check_model(self):
        """
        Check the model for various errors. This method checks for the following
        errors -

        * Checks if the cardinalities of all the variables are consistent across all the factors.
        * Factors are defined for all the random variables.

        Returns
        -------
        check: boolean
            True if all the checks are passed
        """
        cardinalities = self.get_cardinality()
        for factor in self.factors:
            for variable, cardinality in zip(factor.scope(), factor.cardinality):
                if cardinalities[variable] != cardinality:
                    raise ValueError(
                        'Cardinality of variable {var} not matching among factors'.format(var=variable))
            for var1, var2 in itertools.combinations(factor.variables, 2):
                if var2 not in self.neighbors(var1):
                    raise ValueError("DiscreteFactor inconsistent with the model.")
        return True

    def to_factor_graph(self):
        """
        Converts the markov model into factor graph.

        A factor graph contains two types of nodes. One type corresponds to
        random variables whereas the second type corresponds to factors over
        these variables. The graph only contains edges between variables and
        factor nodes. Each factor node is associated with one factor whose
        scope is the set of variables that are its neighbors.

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> student = MarkovModel([('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> factor1 = DiscreteFactor(['Alice', 'Bob'], [3, 2], np.random.rand(6))
        >>> factor2 = DiscreteFactor(['Bob', 'Charles'], [2, 2], np.random.rand(4))
        >>> student.add_factors(factor1, factor2)
        >>> factor_graph = student.to_factor_graph()
        """
        from pgmpy.models import FactorGraph
        factor_graph = FactorGraph()

        if not self.factors:
            raise ValueError('Factors not associated with the random variables.')

        factor_graph.add_nodes_from(self.nodes())
        for factor in self.factors:
            scope = factor.scope()
            factor_node = 'phi_' + '_'.join(scope)
            factor_graph.add_edges_from(itertools.product(scope, [factor_node]))
            factor_graph.add_factors(factor)

        return factor_graph

    def triangulate(self, heuristic='H6', order=None, inplace=False):
        """
        Triangulate the graph.

        If order of deletion is given heuristic algorithm will not be used.

        Parameters
        ----------
        heuristic: H1 | H2 | H3 | H4 | H5 | H6
            The heuristic algorithm to use to decide the deletion order of
            the variables to compute the triangulated graph.
            Let X be the set of variables and X(i) denotes the i-th variable.

            * S(i) - The size of the clique created by deleting the variable.
            * E(i) - Cardinality of variable X(i).
            * M(i) - Maximum size of cliques given by X(i) and its adjacent nodes.
            * C(i) - Sum of size of cliques given by X(i) and its adjacent nodes.

            The heuristic algorithm decide the deletion order if this way:

            * H1 - Delete the variable with minimal S(i).
            * H2 - Delete the variable with minimal S(i)/E(i).
            * H3 - Delete the variable with minimal S(i) - M(i).
            * H4 - Delete the variable with minimal S(i) - C(i).
            * H5 - Delete the variable with minimal S(i)/M(i).
            * H6 - Delete the variable with minimal S(i)/C(i).

        order: list, tuple (array-like)
            The order of deletion of the variables to compute the triagulated
            graph. If order is given heuristic algorithm will not be used.

        inplace: True | False
            if inplace is true then adds the edges to the object from
            which it is called else returns a new object.

        Reference
        ---------
        http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.56.3607

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = MarkovModel()
        >>> G.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        >>> G.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
        ...                   ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
        ...                   ('x4', 'x7'), ('x5', 'x7')])
        >>> phi = [DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in G.edges()]
        >>> G.add_factors(*phi)
        >>> G_chordal = G.triangulate()
        """
        self.check_model()

        if self.is_triangulated():
            if inplace:
                return
            else:
                return self

        graph_copy = nx.Graph(self.edges())
        edge_set = set()

        def _find_common_cliques(cliques_list):
            """
            Finds the common cliques among the given set of cliques for
            corresponding node.
            """
            common = set([tuple(x) for x in cliques_list[0]])
            for i in range(1, len(cliques_list)):
                common = common & set([tuple(x) for x in cliques_list[i]])
            return list(common)

        def _find_size_of_clique(clique, cardinalities):
            """
            Computes the size of a clique.

            Size of a clique is defined as product of cardinalities of all the
            nodes present in the clique.
            """
            return list(map(lambda x: np.prod([cardinalities[node] for node in x]),
                            clique))

        def _get_cliques_dict(node):
            """
            Returns a dictionary in the form of {node: cliques_formed} of the
            node along with its neighboring nodes.

            clique_dict_removed would be containing the cliques created
            after deletion of the node
            clique_dict_node would be containing the cliques created before
            deletion of the node
            """
            graph_working_copy = nx.Graph(graph_copy.edges())
            neighbors = graph_working_copy.neighbors(node)
            graph_working_copy.add_edges_from(itertools.combinations(neighbors, 2))
            clique_dict = nx.cliques_containing_node(graph_working_copy,
                                                     nodes=([node] + neighbors))
            graph_working_copy.remove_node(node)
            clique_dict_removed = nx.cliques_containing_node(graph_working_copy,
                                                             nodes=neighbors)
            return clique_dict, clique_dict_removed

        if not order:
            order = []

            cardinalities = self.get_cardinality()
            for index in range(self.number_of_nodes()):
                # S represents the size of clique created by deleting the
                # node from the graph
                S = {}
                # M represents the size of maximum size of cliques given by
                # the node and its adjacent node
                M = {}
                # C represents the sum of size of the cliques created by the
                # node and its adjacent node
                C = {}
                for node in set(graph_copy.nodes()) - set(order):
                    clique_dict, clique_dict_removed = _get_cliques_dict(node)
                    S[node] = _find_size_of_clique(
                        _find_common_cliques(list(clique_dict_removed.values())),
                        cardinalities
                    )[0]
                    common_clique_size = _find_size_of_clique(
                        _find_common_cliques(list(clique_dict.values())),
                        cardinalities
                    )
                    M[node] = np.max(common_clique_size)
                    C[node] = np.sum(common_clique_size)

                if heuristic == 'H1':
                    node_to_delete = min(S, key=S.get)

                elif heuristic == 'H2':
                    S_by_E = {key: S[key] / cardinalities[key] for key in S}
                    node_to_delete = min(S_by_E, key=S_by_E.get)

                elif heuristic == 'H3':
                    S_minus_M = {key: S[key] - M[key] for key in S}
                    node_to_delete = min(S_minus_M, key=S_minus_M.get)

                elif heuristic == 'H4':
                    S_minus_C = {key: S[key] - C[key] for key in S}
                    node_to_delete = min(S_minus_C, key=S_minus_C.get)

                elif heuristic == 'H5':
                    S_by_M = {key: S[key] / M[key] for key in S}
                    node_to_delete = min(S_by_M, key=S_by_M.get)

                else:
                    S_by_C = {key: S[key] / C[key] for key in S}
                    node_to_delete = min(S_by_C, key=S_by_C.get)

                order.append(node_to_delete)

        graph_copy = nx.Graph(self.edges())
        for node in order:
            for edge in itertools.combinations(graph_copy.neighbors(node), 2):
                graph_copy.add_edge(edge[0], edge[1])
                edge_set.add(edge)
            graph_copy.remove_node(node)

        if inplace:
            for edge in edge_set:
                self.add_edge(edge[0], edge[1])
            return self

        else:
            graph_copy = MarkovModel(self.edges())
            for edge in edge_set:
                graph_copy.add_edge(edge[0], edge[1])
            return graph_copy

    def to_junction_tree(self):
        """
        Creates a junction tree (or clique tree) for a given markov model.

        For a given markov model (H) a junction tree (G) is a graph
        1. where each node in G corresponds to a maximal clique in H
        2. each sepset in G separates the variables strictly on one side of the
        edge to other.

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> mm = MarkovModel()
        >>> mm.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        >>> mm.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
        ...                    ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
        ...                    ('x4', 'x7'), ('x5', 'x7')])
        >>> phi = [DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in mm.edges()]
        >>> mm.add_factors(*phi)
        >>> junction_tree = mm.to_junction_tree()
        """
        from pgmpy.models import JunctionTree

        # Check whether the model is valid or not
        self.check_model()

        # Triangulate the graph to make it chordal
        triangulated_graph = self.triangulate()

        # Find maximal cliques in the chordal graph
        cliques = list(map(tuple, nx.find_cliques(triangulated_graph)))

        # If there is only 1 clique, then the junction tree formed is just a
        # clique tree with that single clique as the node
        if len(cliques) == 1:
            clique_trees = JunctionTree()
            clique_trees.add_node(cliques[0])

        # Else if the number of cliques is more than 1 then create a complete
        # graph with all the cliques as nodes and weight of the edges being
        # the length of sepset between two cliques
        elif len(cliques) >= 2:
            complete_graph = UndirectedGraph()
            edges = list(itertools.combinations(cliques, 2))
            weights = list(map(lambda x: len(set(x[0]).intersection(set(x[1]))),
                           edges))
            for edge, weight in zip(edges, weights):
                complete_graph.add_edge(*edge, weight=-weight)

            # Create clique trees by minimum (or maximum) spanning tree method
            clique_trees = JunctionTree(nx.minimum_spanning_tree(complete_graph).edges())

        # Check whether the factors are defined for all the random variables or not
        all_vars = itertools.chain(*[factor.scope() for factor in self.factors])
        if set(all_vars) != set(self.nodes()):
            ValueError('DiscreteFactor for all the random variables not specified')

        # Dictionary stating whether the factor is used to create clique
        # potential or not
        # If false, then it is not used to create any clique potential
        is_used = {factor: False for factor in self.factors}

        for node in clique_trees.nodes():
            clique_factors = []
            for factor in self.factors:
                # If the factor is not used in creating any clique potential as
                # well as has any variable of the given clique in its scope,
                # then use it in creating clique potential
                if not is_used[factor] and set(factor.scope()).issubset(node):
                    clique_factors.append(factor)
                    is_used[factor] = True

            # To compute clique potential, initially set it as unity factor
            var_card = [self.get_cardinality()[x] for x in node]
            clique_potential = DiscreteFactor(node, var_card, np.ones(np.product(var_card)))
            # multiply it with the factors associated with the variables present
            # in the clique (or node)
            clique_potential *= factor_product(*clique_factors)
            clique_trees.add_factors(clique_potential)

        if not all(is_used.values()):
            raise ValueError('All the factors were not used to create Junction Tree.'
                             'Extra factors are defined.')

        return clique_trees

    def markov_blanket(self, node):
        """
        Returns a markov blanket for a random variable.

        Markov blanket is the neighboring nodes of the given node.

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> mm = MarkovModel()
        >>> mm.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        >>> mm.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
        ...                    ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
        ...                    ('x4', 'x7'), ('x5', 'x7')])
        >>> mm.markov_blanket('x1')
        """
        return self.neighbors(node)

    def get_local_independencies(self, latex=False):
        """
        Returns all the local independencies present in the markov model.

        Local independencies are the independence assertion in the form of
        .. math:: {X \perp W - {X} - MB(X) | MB(X)}
        where MB is the markov blanket of all the random variables in X

        Parameters
        ----------
        latex: boolean
            If latex=True then latex string of the indepedence assertion would
            be created

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> mm = MarkovModel()
        >>> mm.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        >>> mm.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
        ...                    ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
        ...                    ('x4', 'x7'), ('x5', 'x7')])
        >>> mm.get_local_independecies()
        """
        local_independencies = Independencies()

        all_vars = set(self.nodes())
        for node in self.nodes():
            markov_blanket = set(self.markov_blanket(node))
            rest = all_vars - set([node]) - markov_blanket
            try:
                local_independencies.add_assertions([node, list(rest), list(markov_blanket)])
            except ValueError:
                pass

        local_independencies.reduce()

        if latex:
            return local_independencies.latex_string()
        else:
            return local_independencies

    def to_bayesian_model(self):
        """
        Creates a Bayesian Model which is a minimum I-Map for this markov model.

        The ordering of parents may not remain constant. It would depend on the
        ordering of variable in the junction tree (which is not constant) all the
        time.

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> mm = MarkovModel()
        >>> mm.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        >>> mm.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
        ...                    ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
        ...                    ('x4', 'x7'), ('x5', 'x7')])
        >>> phi = [DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in mm.edges()]
        >>> mm.add_factors(*phi)
        >>> bm = mm.to_bayesian_model()
        """
        from pgmpy.models import BayesianModel

        bm = BayesianModel()
        var_clique_dict = defaultdict(tuple)
        var_order = []

        # Create a junction tree from the markov model.
        # Creation of clique tree involves triangulation, finding maximal cliques
        # and creating a tree from these cliques
        junction_tree = self.to_junction_tree()

        # create an ordering of the nodes based on the ordering of the clique
        # in which it appeared first
        root_node = junction_tree.nodes()[0]
        bfs_edges = nx.bfs_edges(junction_tree, root_node)
        for node in root_node:
            var_clique_dict[node] = root_node
            var_order.append(node)
        for edge in bfs_edges:
            clique_node = edge[1]
            for node in clique_node:
                if not var_clique_dict[node]:
                    var_clique_dict[node] = clique_node
                    var_order.append(node)

        # create a bayesian model by adding edges from parent of node to node as
        # par(x_i) = (var(c_k) - x_i) \cap {x_1, ..., x_{i-1}}
        for node_index in range(len(var_order)):
            node = var_order[node_index]
            node_parents = (set(var_clique_dict[node]) - set([node])).intersection(
                set(var_order[:node_index]))
            bm.add_edges_from([(parent, node) for parent in node_parents])
            # TODO : Convert factor into CPDs
        return bm

    def get_partition_function(self):
        """
        Returns the partition function for a given undirected graph.

        A partition function is defined as

        .. math:: \sum_{X}(\prod_{i=1}^{m} \phi_i)

        where m is the number of factors present in the graph
        and X are all the random variables present.

        Examples
        --------
        >>> from pgmpy.models import MarkovModel
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> G = MarkovModel()
        >>> G.add_nodes_from(['x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7'])
        >>> G.add_edges_from([('x1', 'x3'), ('x1', 'x4'), ('x2', 'x4'),
        ...                   ('x2', 'x5'), ('x3', 'x6'), ('x4', 'x6'),
        ...                   ('x4', 'x7'), ('x5', 'x7')])
        >>> phi = [DiscreteFactor(edge, [2, 2], np.random.rand(4)) for edge in G.edges()]
        >>> G.add_factors(*phi)
        >>> G.get_partition_function()
        """
        self.check_model()

        factor = self.factors[0]
        factor = factor_product(factor, *[self.factors[i] for i in
                                          range(1, len(self.factors))])
        if set(factor.scope()) != set(self.nodes()):
            raise ValueError('DiscreteFactor for all the random variables not defined.')

        return np.sum(factor.values)

    def copy(self):
        """
        Returns a copy of this Markov Model.

        Returns
        -------
        MarkovModel: Copy of this Markov model.

        Examples
        -------
        >>> from pgmpy.factors.discrete import DiscreteFactor
        >>> from pgmpy.models import MarkovModel
        >>> G = MarkovModel()
        >>> G.add_nodes_from([('a', 'b'), ('b', 'c')])
        >>> G.add_edge(('a', 'b'), ('b', 'c'))
        >>> G_copy = G.copy()
        >>> G_copy.edges()
        [(('a', 'b'), ('b', 'c'))]
        >>> G_copy.nodes()
        [('a', 'b'), ('b', 'c')]
        >>> factor = DiscreteFactor([('a', 'b')], cardinality=[3],
        ...                 values=np.random.rand(3))
        >>> G.add_factors(factor)
        >>> G.get_factors()
        [<DiscreteFactor representing phi(('a', 'b'):3) at 0x...>]
        >>> G_copy.get_factors()
        []
        """
        clone_graph = MarkovModel(self.edges())
        clone_graph.add_nodes_from(self.nodes())

        if self.factors:
            factors_copy = [factor.copy() for factor in self.factors]
            clone_graph.add_factors(*factors_copy)

        return clone_graph
