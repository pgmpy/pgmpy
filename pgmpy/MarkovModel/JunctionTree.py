__author__ = 'navin'


from pgmpy.MarkovModel.UndirectedGraph import UndirectedGraph


class JunctionTree(UndirectedGraph):

    def addJTEdges(self):
        nodes = self.nodes()
        num_nodes = self.number_of_nodes()
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                print(self.node[nodes[i]]["clique_nodes"])
                set1 = set(self.node[nodes[i]]["clique_nodes"])
                set2 = set(self.node[nodes[j]]["clique_nodes"])
                set3 = set1.intersection(set2)
                self.add_edge(nodes[i], nodes[j], {'wt': len(set3)})
        self.print_graph("before the MST, after adding all edges")
        self.maximumSpanningTree()
        return self