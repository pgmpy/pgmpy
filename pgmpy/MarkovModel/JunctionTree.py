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

    def _pull(self):
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
        factor = self._pull_h(root_node)
        self._pull_status = True
        return factor

    def _pull_h(self, node):
        """
        Helps in the recursion for _pull

        See Also
        --------
        _pull
        """
        self.node[node]["visited"] = True
        factor = self.node[node]["factor"]
        #print(str(node)+ "self factor "+str(factor))
        self_vars = factor.get_variables()
        nbrs_to_pull_from = self._unvisited_neighbors(node)
        for nbr in nbrs_to_pull_from:
            #print(nbr)
            f = self._pull_h(nbr)
            assert isinstance(f, Factor)
            #print("marginalize at " + str(node) + " using " + str(self_vars))
            f = f.marginalize_except(self_vars)
            factor = factor.product(f)
        self.node[node]["pull_factor"] = factor
        return factor

    def _push(self):
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
        self._push_h(root_node, empty_factor)
        self._push_status = True

    def _push_h(self, node, factor):
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
            factor = factor.marginalize_except(self_factor.get_variables())
            self.node[node]["push_factor"] = factor
            full_prod = factor.product(self.node[node]["pull_factor"])
        assert isinstance(full_prod, Factor)
        self.node[node]["prod_factor"] = full_prod
        rel_nbrs = self._unvisited_neighbors(node)
        for nbr in rel_nbrs:
            nbr_factor = self.node[nbr]["pull_factor"]
            assert isinstance(nbr_factor, Factor)
            nbr_factor = nbr_factor.marginalize_except(self_factor.get_variables())
            fact_push = full_prod.divide(nbr_factor)
            self._push_h(nbr, fact_push)

    def normalization_constant(self):
        """
        Finds the normalization constant using Junction Trees

        Example
        -------

        """
        factor = self._pull()
        assert isinstance(factor, Factor)
        norm = factor.sum_values()
        return norm

    def marginal_prob(self, var):
        self._pull()
        #print("pulled factor" + str(factor))
        self._push()
        for node in self.nodes():
            prod_factor = self.node[node]["prod_factor"]
            assert isinstance(prod_factor, Factor)
            if var in prod_factor.get_variables():
                #print("product factor " + str(prod_factor))
                rel_fact = prod_factor.marginalize_except(var)
                return rel_fact
        raise Exception("Should never reach here! If here, then trouble!")