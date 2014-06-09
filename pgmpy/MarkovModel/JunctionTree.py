import networkx as nx
from pgmpy.MarkovModel.UndirectedGraph import UndirectedGraph


class JunctionTree(UndirectedGraph):
    def add_jt_edges(self):
        nodes = self.nodes()
        num_nodes = self.number_of_nodes()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                print(self.node[nodes[i]]["clique_nodes"])
                set1 = set(self.node[nodes[i]]["clique_nodes"])
                set2 = set(self.node[nodes[j]]["clique_nodes"])
                set3 = set1.intersection(set2)
                self.add_edge(nodes[i], nodes[j], weight=-len(set3))
        self.print_graph("before the MST, after adding all edges")
        new_edges = nx.minimum_spanning_edges(self)
        self.remove_edges_from(self.edges())
        self.add_edges_from(new_edges)
        return self