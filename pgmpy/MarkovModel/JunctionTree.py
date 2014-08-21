import networkx as nx
from pgmpy.MarkovModel.MarkovModel import UndirectedGraph
from pgmpy.Factor.Factor import Factor


class JunctionTree(UndirectedGraph):
    """
    This class is meant to represent junction trees (Called as clique trees popularly).
    It will contain a lot of functionalities to work on junction trees and to run
    inference algorithms on JunctionTrees,
    """

    def __init__(self):
        super(UndirectedGraph, self).__init__()
        self._pull_status = False
        self._push_status = False

    def add_jt_edges(self):
        """
        This adds appropriate edges to the junction tree graph. Given a junction tree
        with all the nodes containing cliques of the MarkovModel, running the function
        on the junction tree will add all edges to the JT and then remove edges as
        necessary to make it a Junction Tree (is a tree and satisfies Running Intersection
        Property)

        Parameters
        ----------
        None

        See Also
        --------
        _junction_tree1, _jt_from_chordal_graph in UndirectedGraph.py

        Examples
        --------
        It is used in make_jt in UndirectedGraph and I don't expect
        the user to use it directly. Does this need an example?
        """
        nodes = self.nodes()
        num_nodes = self.number_of_nodes()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                set1 = set(self.node[nodes[i]]["clique_nodes"])
                set2 = set(self.node[nodes[j]]["clique_nodes"])
                set3 = set1.intersection(set2)
                self.add_edge(nodes[i], nodes[j], weight=-len(set3))
        new_edges = list(nx.minimum_spanning_edges(self))
        new_edges = [(edge[0], edge[1]) for edge in new_edges]
        self.remove_edges_from(list(set(self.edges()) -
                                    set(new_edges)))
        return self

    def insert_factors(self, factors):
        """
        Given a junction tree, this adds all the factors to appropriate nodes
        in the junction tree

        Parameters
        ----------
        jt : The completely made junction tree ready to be attached to factors

        Examples
        --------
        It is used in make_jt in UndirectedGraph and I don't expect
        the user to use it directly. Does this need an example?
        """
        for node in self.nodes():
            self.node[node]["factor"] = None
        for factor in factors:
            fact_vars = set(factor.get_variables())
            #print("Vars for the factor "+str(vars))
            flag = False
            for node in self.nodes():
                maxclique_nodes = set(self.node[node]["clique_nodes"])
                if len(fact_vars.difference(maxclique_nodes)) == 0:
                    if self.node[node]["factor"] is None:
                        self.node[node]["factor"] = factor
                    else:
                        f_prev = self.node[node]["factor"]
                        assert isinstance(f_prev, Factor)
                        f_res = f_prev.product(factor)
                        self.node[node]["factor"] = f_res
                    flag = True
                    break
            if not flag:
                raise ValueError("The factor " + str(factor)
                                 + " doesn't correspond to any maxclique")

    def _unvisited_neighbors(self, curr_node):
        """
        Returns the unvisited neighbors of a node when the visited attribute
        has been set for all the nodes
        """
        try:
            ret = [node for node in self.neighbors(curr_node)
                   if not self.node[node]["visited"]]
            return ret
        except KeyError:
            raise KeyError("Visited not defined for the nodes")

    def _pull(self, func):
        """
        Pull phase of the message passing

        See Also
        --------
        _pull_h
        """
        root_node = self.nodes()[0]
        if self._pull_status:
            return self.node[root_node]["pull_factor"]
        for node in self.nodes():
            self.node[node]["visited"] = False
        factor = self._pull_h(root_node, func)
        assert isinstance(factor, Factor)
        self._pull_status = True
        return factor

    def _pull_h(self, node, func):
        """
        Helps in the recursion for _pull

        See Also
        --------
        _pull
        """
        self.node[node]["visited"] = True
        factor = self.node[node]["factor"]
        self_vars = factor.get_variables()
        nbrs_to_pull_from = self._unvisited_neighbors(node)
        for nbr in nbrs_to_pull_from:
            f = self._pull_h(nbr, func)
            assert isinstance(f, Factor)
            f = func(f, self_vars)
            assert isinstance(f, Factor)
            factor = factor.product(f)
        self.node[node]["pull_factor"] = factor
        return factor

    def _push(self, func):
        """
        Push phase of the message passing

        See Also
        --------
        _push_h
        """
        for node in self.nodes():
            self.node[node]["visited"] = False
        empty_factor = None
        root_node = self.nodes()[0]
        self._push_h(root_node, empty_factor, func)
        self._push_status = True

    def _push_h(self, node, factor, func):
        """
        Helps in the recursion for _push

        See Also
        --------
        _push
        """
        self.node[node]["visited"] = True
        self_factor = self.node[node]["factor"]
        if factor is None:
            full_prod = self.node[node]["pull_factor"]
        else:
            factor = func(factor, self_factor.get_variables())
            self.node[node]["push_factor"] = factor
            full_prod = factor.product(self.node[node]["pull_factor"])
        assert isinstance(full_prod, Factor)
        self.node[node]["prod_factor"] = full_prod
        rel_nbrs = self._unvisited_neighbors(node)
        for nbr in rel_nbrs:
            nbr_factor = self.node[nbr]["pull_factor"]
            assert isinstance(nbr_factor, Factor)
            nbr_factor = func(nbr_factor, (self_factor.get_variables()))
            fact_push = full_prod.divide(nbr_factor)
            self._push_h(nbr, fact_push, func)

    def normalization_constant(self):
        """
        Finds the normalization constant using Junction Trees

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> graph = mm.MarkovModel([('d', 'g'), ('i', 'g')])
        >>> graph.add_states(
        ...    {'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart']})
        >>> graph.add_factor(['d', 'g'], [1, 2, 3, 4, 5, 6])
        >>> graph.add_factor(['i', 'g'], [1, 2, 3, 4, 5, 6])
        >>> jt = graph.make_jt(2)
        >>> jt.normalization_constant()
        163.0
        """
        factor = self._pull(Factor.marginalize_except)
        assert isinstance(factor, Factor)
        norm = factor.marginalize_except([])
        return norm.values[0]

    def marginal_prob(self, var):
        """
        Uses junction tree to find the marginal probability of any variable

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> graph = mm.MarkovModel([('d', 'g'), ('i', 'g')])
        >>> graph.add_states(
        ...    {'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart']})
        >>> graph.add_factor(['d', 'g'], [1, 2, 3, 4, 5, 6])
        >>> graph.add_factor(['i', 'g'], [1, 2, 3, 4, 5, 6])
        >>> jt = graph.make_jt(2)
        >>> jt.marginal_prob('d')
        d	phi(d)
        d_0	46.0
        d_1	109.0
        """
        self._pull(Factor.marginalize_except)
        #print("pulled factor" + str(factor))
        self._push(Factor.marginalize_except)
        for node in self.nodes():
            prod_factor = self.node[node]["prod_factor"]
            assert isinstance(prod_factor, Factor)
            if var in prod_factor.get_variables():
                #print("product factor " + str(prod_factor))
                rel_fact = prod_factor.marginalize_except(var)
                return rel_fact
        raise Exception("Should never reach here! If here, then trouble!")

    def map(self):
        """
        Uses junction tree to find the marginal probability of any variable

        Example
        -------
        >>> from pgmpy import MarkovModel as mm
        >>> graph = mm.MarkovModel([('d', 'g'), ('i', 'g')])
        >>> graph.add_states(
        ...    {'d': ['easy', 'hard'], 'g': ['A', 'B', 'C'], 'i': ['dumb', 'smart']})
        >>> f = graph.add_factor(['d', 'g'], [1, 2, 3, 4, 5, 6])
        >>> f = graph.add_factor(['i', 'g'], [1, 2, 3, 4, 5, 6])
        >>> jt = graph.make_jt(2)
        >>> jt.map()
        [('i', 1), ('d', 1), ('g', 2)]
        """
        factor = self._pull(Factor.maximize_except)
        factor = factor.maximize_except([])
        assert isinstance(factor, Factor)
        return factor.data[0]