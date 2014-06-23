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
