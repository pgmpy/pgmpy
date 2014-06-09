from heapq import heappush, heappop, heapify
import networkx as nx

debug = False


def p(x):
    if debug:
        print(x)


class UndirectedGraph(nx.Graph):
    def equal_graphs(self, graph2):
        """
        Just to check if the current graph is same as G2 in terms of nodes and edges

        Parameters
        ----------
        G2:
            The other graph

        """
        assert isinstance(graph2, UndirectedGraph)
        node_set1 = set(self.nodes())
        node_set2 = set(graph2.nodes())
        if node_set1 != node_set2:
            return False
        edge_set1 = set(self.edges())
        edge_set2 = set(graph2.edges())
        if edge_set1 != edge_set2:
            return False
        return True

    def is_triangulated(self):
        """
        Just checks if the graph is triangulated

        Parameters
        ----------

        See Also
        --------
        maxCardinalitySearch

        Example
        --------
        >>> from pgmpy import MarkovModel
        >>> G = MarkovModel.UndirectedGraph()
        >>> G.read_simple_format("graph")
        >>> G.jt_techniques(2,False,True)
        4
        >>> G.is_triangulated()
        True
        """
        ret = nx.is_chordal(self)
        return ret

    def print_graph(self, s):
        """
        Prints the graph in a particular fashion.
        Useful for debugging

        Parameters
        ----------
        nodes  :  List of nodes
                The list of nodes which are to be checked for clique property

        See Also
        --------
        make_clique

        Example
        -------
        >>> from pgmpy import MarkovModel
        >>> G = MarkovModel.UndirectedGraph()
        >>> G.read_simple_format("graph")
        >>> G.print_graph("Test Printing")
        Printing the graph Test Printing<<<
        10( {} ) : {'8': {}, '5': {}, '7': {}}
        1( {} ) : {'0': {}, '2': {}, '4': {}, '8': {}}
        0( {} ) : {'1': {}, '8': {}, '3': {}}
        3( {} ) : {'9': {}, '0': {}, '8': {}}
        2( {} ) : {'1': {}, '4': {}, '7': {}, '6': {}}
        5( {} ) : {'9': {}, '8': {}, '10': {}}
        4( {} ) : {'1': {}, '8': {}, '2': {}, '7': {}}
        7( {} ) : {'10': {}, '2': {}, '4': {}, '6': {}}
        6( {} ) : {'2': {}, '7': {}}
        9( {} ) : {'3': {}, '5': {}}
        8( {} ) : {'10': {}, '1': {}, '0': {}, '3': {}, '5': {}, '4': {}}
        >>>

        """

        print("Printing the graph " + s + "<<<")
        for node in self.nodes():
            print(str(node) + "( " + str(self.node[node]) + " ) : " + str(self[node]))
        print(">>>")

    def check_clique(self, nodes):
        """
        Check if a given set of nodes form a clique

        Parameters
        ----------
        nodes  :  List of nodes
                The list of nodes which are to be checked for clique property

        See Also
        --------
        make_clique

        Example
        -------
        >>> from pgmpy import MarkovModel
        >>> G = MarkovModel.UndirectedGraph()
        >>> G.read_simple_format("graph")
        >>> G.check_clique([1,2,3])
        False

        """
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if not self.has_edge(nodes[i], nodes[j]):
                    return False
        return True

    def make_clique(self, clique_nodes):
        new_edges = []
        for i in range(len(clique_nodes)):
            a = clique_nodes[i]
            for j in range(i + 1, len(clique_nodes)):
                b = clique_nodes[j]
                if not self.has_edge(a, b):
                    p("Edge added b/w " + a + " and " + b)
                    self.add_edge(a, b)
                    new_edges.append((a, b))
        return new_edges

    def _junction_tree1(self, return_junction_tree=True, triangulate_graph=False, f=None):
        """
        Applies the basic junction creation algorithms.
        Refer to Algorithm 9.4 in PGM (Koller)

        Parameters
        ----------
        return_junction_tree  :  boolean
                returns the junction tree if yes, size of max clique if no

        triangulateGraph :
                Removes the edges added while triangulation if False
        f:
                The function that will be used to pick up the node at each stage

        See Also
        --------

        Example
        -------
        private function
        """
        from pgmpy.MarkovModel.JunctionTree import JunctionTree

        jt = JunctionTree()
        jtnode = 0
        nodes = [(f(self, node), node) for node in self.nodes()]
        heapify(nodes)
        p(str(nodes))
        max_clique_size = 0
        triangulated_nodes = set()
        new_edges = []
        while len(nodes) != 0:
            min_el = heappop(nodes)
            curr_node = min_el[1]
            p("Popped " + str(curr_node))
            if curr_node in triangulated_nodes:
                p("continued")
                continue

            triangulated_nodes.add(curr_node)
            flag = False
            set2 = set(self.neighbors(curr_node))
            for nbr in self.neighbors(curr_node):
                if nbr in triangulated_nodes:
                    set1 = set(self.neighbors(nbr))
                    if set2 - set1 is None:
                        flag = True
                        break
            if flag:
                break
            #actual triangulation begins here
            #Take all the neighbours and connect them. Done
            jtnode += 1
            jt.add_node(jtnode)
            clique_nodes = []
            for node in self.neighbors(curr_node):
                if node not in triangulated_nodes:
                    clique_nodes.append(node)
            jt.node[jtnode]["clique_nodes"] = clique_nodes
            p(str(clique_nodes))
            max_clique_size = max(max_clique_size, len(clique_nodes))
            p("Working with " + str(curr_node))
            new_edges_temp = self.make_clique(clique_nodes)
            new_edges.extend(new_edges_temp)
            for node in clique_nodes:
                heappush(nodes, (f(self, node), node))
        if not triangulate_graph:
            for edge in new_edges:
                self.remove_edge(edge[0], edge[1])
        if not return_junction_tree:
            return max_clique_size
        jt.add_jt_edges()
        return jt

    def jt_from_chordal_graph(self, return_junction_tree):
        """

        """
        from pgmpy import MarkovModel

        if return_junction_tree:
            cliques = nx.chordal_graph_cliques(self)
            jt = MarkovModel.JunctionTree()
            jtnode = 0
            for max_clique in cliques:
                jtnode += 1
                jt.add_node(jtnode)
                jt.node[jtnode]["clique_nodes"] = max_clique
            jt.add_jt_edges()
            return jt
        else:
            return nx.chordal_graph_treewidth(self) + 1

    def jt_optimal(self, return_junction_tree, triangulate_graph):
        """
        Exponential strategy to find the optimal junction tree creation strategy

        Parameters
        ----------
        return_junction_tree  :  boolean
                returns the junction tree if yes, size of max clique if no

        triangulateGraph :
                Removes the edges added while triangulation if False

        Example
        -------

        """
        return None

    @staticmethod
    def min_fill_heuristic(graph, node):
        """
        Minimum-fill heuristic
        """
        nbrs = graph.neighbors(node)
        num_edges = 0
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if graph.has_edge(nbrs[i], nbrs[j]):
                    num_edges += 1
        val = ((len(nbrs) * (len(nbrs) - 1)) / 2) - num_edges
        return val

    def best_triangulation_heuristic(self):
        """
        Tries every triangulation heuristic defined in jt_techniques and finds
        the best one

        Parameters
        ----------
        None

        See Also
        --------
        jt_techniques

        Example
        -------
        >>> from pgmpy import MarkovModel
        >>> G = MarkovModel.UndirectedGraph()
        >>> G.read_simple_format("graph")
        >>> G.best_triangulation_heuristic()
        2
        """
        i = 2
        min_clique_size = int("inf")
        min_clique_technique = -1
        while True:
            size = self.jt_tree_width(i)
            if not size:
                break
            if size < min_clique_size:
                min_clique_technique = i
                min_clique_size = size
        return min_clique_technique

    def jt_techniques(self, triangulation_technique, return_junction_tree, triangulate_graph):
        """
        Returns the junction tree or the max clique size depending on a triangulation technique
        Takes option about whether to retain the edges added during triangulation

        Parameters
        ----------
        triangulation_technique  :  integer
                The index of the triangulation technique to use
                0 : THe graph is already triangulated. Just make the junction tree
                1 : Optimal Triangulation. Exponential. Use if you have ample time
                2 : Min-neighbour
                3 : Min =fill
                4 : Max-neighbour

        return_junction_tree : boolean
                Returns the junction tree if true, returns the triangulated graph otherwise

        triangulate_graph : boolean
                Retains the edges added during triangulation, if yes. Deletes the edges
                otherwise

        See Also
        --------
        junctionTree1
        jt_optimal
        jt_from_chordal_graph

        Example
        -------
        >>> from pgmpy import MarkovModel
        >>> G = MarkovModel.UndirectedGraph()
        >>> G.read_simple_format("graph")
        >>> G.jt_tree_width(2)
        4
        """
        ret=None
        if triangulation_technique == 0:
            ret = self.jt_from_chordal_graph(return_junction_tree)
            if not ret:
                raise Exception(" Graph Is Not Triangulated ")
        if triangulation_technique == 1:
            ret = self.jt_optimal(return_junction_tree, triangulate_graph)
        if triangulation_technique == 2:
            f = (lambda graph, node: len(graph.neighbors(node)))
            ret = self._junction_tree1(return_junction_tree, triangulate_graph, f)
        elif triangulation_technique == 3:
            f = (lambda graph, node: UndirectedGraph.min_fill_heuristic(graph, node))
            ret = self._junction_tree1(return_junction_tree, triangulate_graph, f)
        elif triangulation_technique == 4:
            f = (lambda graph, node: -len(graph.neighbors(node)))
            ret = self._junction_tree1(return_junction_tree, triangulate_graph, f)
        elif triangulation_technique == 5:
            f = (lambda graph, node: 5)
            ret = self._junction_tree1(return_junction_tree, triangulate_graph, f)
        else:
            ret = False
        return ret

    def jt_tree_width(self, triangulation_technique):
        """
        Returns the max-clique size that a triangulation technique will return
        It removes all the edges it added while triangulating it and hence doesn't affect the graph

        Parameters
        ----------
        triangulation_technique  :  integer
                The index of the triangulation technique to use
                See the documentation of jt_techniques function to see the
                index corresponding to each heuristic

        See Also
        --------
        jt_techniques

        Example
        -------
        >>> from pgmpy import MarkovModel
        >>> G = MarkovModel.UndirectedGraph()
        >>> G.read_simple_format("graph")
        >>> G.jt_tree_width(2)
        4
        """

        return self.jt_techniques(triangulation_technique, False, False)

    def make_jt(self, triangulation_technique):
        """
        Return the junction Tree after triangulating it using the triangulation_technique.
        It removes all the edges it added while triangulating it and hence doesn't affect the graph

        Parameters
        ----------
        triangulation_technique  :  integer
                The index of the triangulation technique to use
                See the documentation of jt_techniques function to see the
                index corresponding to each heuristic

        See Also
        --------
        jt_techniques

        Example
        -------
        >>> from pgmpy import MarkovModel
        >>> G = MarkovModel.UndirectedGraph()
        >>> G.read_simple_format("graph")
        >>> jt=G.make_jt(2)
        >>> jt
        """

        return self.jt_techniques(triangulation_technique, True, False)

    def read_simple_format(self, filename):
        """
        Read the graph from a file assuming a very simple graph reading format

        Parameters
        ----------
        filename  :  String
                The file which has the graph data

        Example
        -------
        >>> from pgmpy import MarkovModel
        >>> G = MarkovModel.UndirectedGraph()
        >>> G.read_simple_format("graph")
        """
        file = open(filename, "r")
        num_nodes = int(file.readline())
        for i in range(num_nodes):
            self.add_node(str(i))
        #print("nodes"+str(num_nodes))
        file.readline()
        while True:
            edge = file.readline()
            if not edge:
                break
            nodes = edge.split()
            self.add_edge(nodes[0], nodes[1])