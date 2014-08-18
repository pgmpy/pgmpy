import networkx as nx


class DirectedGraph(nx.DiGraph):
    """
    Base class for implementing functions for Directed Graphs
    Flow_algorithms have been implemented currently
    """

    def get_node_name_with_suffix(self, node):
        """
        When we need to add a node, then we always run into risk of already having
        a node with that name. So this tries to suggest an alternative name by adding
        suffixes

        Parameter
        ---------
        node: the string to which suffixes are added

        Example
        --------
        >>> import MarkovModel as mm
        >>> g = mm.DirectedGraph()
        >>> g.add_node("a")
        >>> g.get_node_name_with_suffix("a")
        'a0'
        """
        if node not in self.nodes():
            return node
        i = 0
        while True:
            if node + str(i) not in self.nodes():
                return node + str(i)
            i += 1

    def add_to_flow_edge_capacity(self, node1, node2, cpcty):
        """
        Increases the capacity of the edge between node1 and node2

        Parameter
        ----------
        node1 : the source node
        node2 : the dest node
        cpcty : the amount by which the capacity needs to be increased

        Example
        ---------
        >>> import MarkovModel as mm
        >>> g = mm.DirectedGraph()
        >>> g.add_nodes_from(['a','b'])
        >>> g.add_all_flow_edges()
        >>> g.add_to_flow_edge_capacity('a','b',5)
        >>> g.get_flow_capacity('a','b')
        5
        """
        if self.has_edge(node1, node2):
            self[node1][node2]['capacity'] += cpcty
        else:
            self.add_edge(node1, node2, capacity=cpcty, flow=0)
            self.add_edge(node2, node1, capacity=0, flow=0)

    def get_flow_capacity(self, var1, var2):
        """
        Retrieves the flow capacity between two nodes

        Parameter
        ----------
        var1 : the source node
        var2 : the dest node

        Example
        ---------
        >>> import MarkovModel as mm
        >>> g = mm.DirectedGraph()
        >>> g.add_nodes_from(['a','b'])
        >>> g.add_all_flow_edges()
        >>> g.add_to_flow_edge_capacity('a','b',5)
        >>> g.get_flow_capacity('a','b')
        5
        """
        if self.has_edge(var1, var2):
            return self[var1][var2]['capacity']
        else:
            return 0.0

    def add_all_flow_edges(self):
        """
        Effectively adds edges between all the nodes in the graph with 0 capacity
        (Note : However this function doesn't do anything now because addition of edges
        is being handled in add_to_flow_edge_capacity )

        Parameter
        ----------

        Example
        ---------
        >>> import MarkovModel as mm
        >>> g = mm.DirectedGraph()
        >>> g.add_nodes_from(['a','b'])
        >>> g.add_all_flow_edges()
        """
        pass

    def __str__(self):
        return self.print_graph("")

    def print_graph(self, s=""):
        """
        Prints the graph

        Parameter
        ----------
        s: Debug message

        Example
        ---------
        >>> import MarkovModel as mm
        >>> g = mm.DirectedGraph()
        >>> g.add_nodes_from(['a','b'])
        >>> g.add_all_flow_edges()
        >>> g.add_to_flow_edge_capacity('a','b',5)
        >>> g.print_graph()
        ==================
        Printing the graph
        a 	(b,{'capacity': 5, 'flow': 0})
        <BLANKLINE>
        b 	(a,{'capacity': 0, 'flow': 0})
        <BLANKLINE>
        ==================
        """
        print("==================")
        print("Printing the graph " + s)
        for node in self.nodes():
            str_node = str(node)
            if self.node[node]:
                str_node += "( " + str(self.node[node]) + " ) : "
            for nbr in self.neighbors(node):
                str_node += " \t(" + str(nbr) + "," + str(self[node][nbr]) + ") \n"
            print(str_node)
        print("==================")

    def _mf_ff_find_path(self, source, sink, dfs_set, path):
        """
        Finds the path by dfs on the residual graph
        Helper function to ford-fulkerson-max-flow
        """
        if source == sink:
            return path
        dfs_set.add(source)
        for nbr in self.neighbors(source):
            edge = (source, nbr)
            edge_dict = self[source][nbr]
            residual = edge_dict['capacity'] - edge_dict['flow']
            if residual > 0 and nbr not in dfs_set:
                result = self._mf_ff_find_path(nbr, sink, dfs_set, path + [edge])
                if result != None:
                    return result

    def max_flow_ford_fulkerson(self, source, sink):
        """
        Performs max_flow algorithm on the directed graph and returns the max_flow
        Also retains the flow values in the edges

        Parameter
        --------
        source : source node for maxflow
        sink  : sink node for maxflow

        Returns
        -------
        Value of the max_flow

        Example
        ---------
        >>> import MarkovModel as mm
        >>> graph = mm.DirectedGraph()
        >>> for node in "sopqrt":
        ...     graph.add_node(node)
        >>> graph.add_to_flow_edge_capacity('s','o',3)
        >>> graph.add_to_flow_edge_capacity('s','p',3)
        >>> graph.add_to_flow_edge_capacity('o','p',2)
        >>> graph.add_to_flow_edge_capacity('o','q',3)
        >>> graph.add_to_flow_edge_capacity('p','r',2)
        >>> graph.add_to_flow_edge_capacity('r','t',3)
        >>> graph.add_to_flow_edge_capacity('q','r',4)
        >>> graph.add_to_flow_edge_capacity('q','t',2)
        >>> graph.max_flow_ford_fulkerson('s','t')
        5
        """
        dfs_set = set()
        path = self._mf_ff_find_path(source, sink, dfs_set, [])
        while path != None:
            residuals = [self[edge[0]][edge[1]]['capacity'] - self[edge[0]][edge[1]]['flow']
                         for edge in path]
            flow = min(residuals)
            for edge in path:
                self[edge[0]][edge[1]]['flow'] += flow
                self[edge[1]][edge[0]]['flow'] -= flow
            dfs_set = set()
            path = self._mf_ff_find_path(source, sink, dfs_set, [])
        return sum(self[source][nbr]['flow'] for nbr in self.neighbors(source))


    def flow_dfs(self, node, dfs_set):
        """
        Performs dfs on a flow graph

        Parameter
        ----------
        node: start point for dfs
        dfs_set : the set used for tracking the nodes visited by dfs

        Example
        -------
        >>> import MarkovModel as mm
        >>> graph = mm.DirectedGraph()
        >>> for node in "sopqrt":
        ...     graph.add_node(node)
        >>> graph.add_to_flow_edge_capacity('s','o',3)
        >>> graph.add_to_flow_edge_capacity('s','p',3)
        >>> graph.add_to_flow_edge_capacity('o','p',2)
        >>> graph.add_to_flow_edge_capacity('o','q',3)
        >>> graph.add_to_flow_edge_capacity('p','r',2)
        >>> graph.add_to_flow_edge_capacity('r','t',3)
        >>> graph.add_to_flow_edge_capacity('q','r',4)
        >>> graph.add_to_flow_edge_capacity('q','t',2)
        >>> graph.max_flow_ford_fulkerson('s','t')
        5
        >>> dfs_set = set()
        >>> graph.flow_dfs('s',dfs_set)
        >>> dfs_set
        {'p', 's'}
        """
        dfs_set.add(node)
        for nbr in self.neighbors(node):
            edge = (node, nbr)
            edge_dict = self[node][nbr]
            residual = edge_dict['capacity'] - edge_dict['flow']
            if residual > 0 and nbr not in dfs_set:
                self.flow_dfs(nbr, dfs_set)
