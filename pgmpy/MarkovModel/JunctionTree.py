import networkx as nx
from pgmpy.MarkovModel.MarkovModel import UndirectedGraph
from pgmpy.Factor.Factor import Factor

debug=False

class JunctionTree(UndirectedGraph):
    """
    This class is meant to represent junction trees (Called as clique trees popularly).
    It will contain a lot of functionalities to work on junction trees and to run
    inference algorithms on JunctionTrees,
    """
    def __init__(self):
        self._pull_status = False
        self._push_status = False

    def _add_jt_edges(self):
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
                #print(self.node[nodes[i]]["clique_nodes"])
                set1 = set(self.node[nodes[i]]["clique_nodes"])
                set2 = set(self.node[nodes[j]]["clique_nodes"])
                set3 = set1.intersection(set2)
                self.add_edge(nodes[i], nodes[j], weight=-len(set3))
        if debug:
            self.print_graph("before the MST, after adding all edges")
        new_edges = list(nx.minimum_spanning_edges(self))
        new_edges = [(edge[0],edge[1]) for edge in new_edges]
        #print("new edges "+str(new_edges))
        #print("old edges "+ str(self.edges()))
        self.remove_edges_from(list(set(self.edges())-
                                    set(new_edges)))
        #self.print_graph("")
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
            vars = set(factor.get_variables())
            #print("Vars for the factor "+str(vars))
            flag=False
            for node in self.nodes():
                maxcliqueNodes = set(self.node[node]["clique_nodes"])
                if len(vars.difference(maxcliqueNodes)) == 0:
                    if self.node[node]["factor"] is None:
                        self.node[node]["factor"] = factor
                    else:
                        f_prev = self.node[node]["factor"]
                        assert isinstance(f_prev, Factor)
                        f_res = f_prev.product(factor)
                        self.node[node]["factor"] = f_res
                    flag=True
                    break
            if not flag:
                raise ValueError("The factor " + str(factor)
                                 + " doesn't correspond to any maxclique")

    def _unvisited_neighbors(self, curr_node):
        try:
            ret =  [node for node in self.neighbors(curr_node)
                   if not self.node[node]["visited"]]
        except KeyError:
            raise KeyError("Visited not defined for the nodes")

    def _pull(self):
        root_node = self.nodes()[0]
        if self._pull_status:
            return self.node[root_node]["pull_factor"]
        for node in self.nodes():
            self.node[node]["visited"]=False
        factor = self._pull_h(root_node)
        self._pull_status = True
        return factor

    def _pull_h(self, node):
        self.node[node]["visited"]=True
        factor = self.node[node]["factor"]
        #print(str(node)+ "self factor "+str(factor))
        self_vars = factor.get_variables()
        nbrs_to_pull_from = self._unvisited_neighbors(node)
        for nbr in nbrs_to_pull_from:
            #print(nbr)
            f = self._pull_h(nbr)
            assert isinstance(f, Factor)
            print("marginalize at "+str(node)+" using "+str(self_vars))
            #f = f.marginalize_except(self_vars)
            factor = factor.product(f)
        self.node[node]["pull_factor"] = factor
        return factor

    def _push(self):
        for node in self.nodes():
            self.node[node]["visited"]=False
        empty_factor = Factor([],[],[])
        assert isinstance(empty_factor, Factor)
        root_node = self.nodes()[0]
        self._push_h(root_node, empty_factor)
        self._push_status = True

    def _push_h(self, node, factor):
        self.node[node]["visited"]=True
        self_factor = self.node[node]["factor"]
        factor = factor.marginalize_except(self_factor.get_variables())
        self.node[node]["push_factor"] = factor
        assert isinstance(factor, Factor)
        full_prod = factor.product(self.node[node]["pull_factor"])
        assert isinstance(full_prod, Factor)
        self.node[node]["prod_factor"] = full_prod
        rel_nbrs = self._unvisited_neighbors(node)
        for nbr in rel_nbrs:
            fact_push = full_prod.divide(self.node[nbr]["pull_factor"])
            self._push_h(nbr, fact_push)

    def normalization_constant(self):
        factor = self._pull()
        assert isinstance(factor,Factor)
        norm = factor.sum_values()
        return norm
