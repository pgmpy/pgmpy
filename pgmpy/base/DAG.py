#!/usr/bin/env python3

import itertools

import networkx as nx

from pgmpy.base import UndirectedGraph


class DAG(nx.DiGraph):
    """
    Base class for all Directed Graphical Models.

    Each node in the graph can represent either a random variable, `Factor`,
    or a cluster of random variables. Edges in the graph represent the
    dependencies between these.

    Parameters
    ----------
    data: input graph
        Data to initialize graph. If data=None (default) an empty graph is
        created. The data can be an edge list or any Networkx graph object.

    Examples
    --------
    Create an empty DAG with no nodes and no edges

    >>> from pgmpy.base import DAG
    >>> G = DAG()

    G can be grown in several ways:

    **Nodes:**

    Add one node at a time:

    >>> G.add_node(node='a')

    Add the nodes from any container (a list, set or tuple or the nodes
    from another graph).

    >>> G.add_nodes_from(nodes=['a', 'b'])

    **Edges:**

    G can also be grown by adding edges.

    Add one edge,

    >>> G.add_edge(u='a', v='b')

    a list of edges,

    >>> G.add_edges_from(ebunch=[('a', 'b'), ('b', 'c')])

    If some edges connect nodes not yet in the model, the nodes
    are added automatically. There are no errors when adding
    nodes or edges that already exist.

    **Shortcuts:**

    Many common graph features allow python syntax for speed reporting.

    >>> 'a' in G     # check if node in graph
    True
    >>> len(G)  # number of nodes in graph
    3
    """

    def __init__(self, ebunch=None):
        super(DAG, self).__init__(ebunch)

    def add_node(self, node, weight=None):
        """
        Adds a single node to the Graph.

        Parameters
        ----------
        node: str, int, or any hashable python object.
            The node to add to the graph.

        weight: int, float
            The weight of the node.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_node(node='A')
        >>> sorted(G.nodes())
        ['A']

        Adding a node with some weight.
        >>> G.add_node(node='B', weight=0.3)

        The weight of these nodes can be accessed as:
        >>> G.node['B']
        {'weight': 0.3}
        >>> G.node['A']
        {'weight': None}
        """

        # Check for networkx 2.0 syntax
        if isinstance(node, tuple) and len(node) == 2 and isinstance(node[1], dict):
            node, attrs = node
            if attrs.get('weight', None) is not None:
                attrs['weight'] = weight
        else:
            attrs = {'weight': weight}

        super(DAG, self).add_node(node, weight=weight)

    def add_nodes_from(self, nodes, weights=None):
        """
        Add multiple nodes to the Graph.

        **The behviour of adding weights is different than in networkx.

        Parameters
        ----------
        nodes: iterable container
            A container of nodes (list, dict, set, or any hashable python
            object).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the variable at index i.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=['A', 'B', 'C'])
        >>> sorted(G.nodes())
        ['A', 'B', 'C']

        Adding nodes with weights:
        >>> G.add_nodes_from(nodes=['D', 'E'], weights=[0.3, 0.6])
        >>> G.node['D']
        {'weight': 0.3}
        >>> G.node['E']
        {'weight': 0.6}
        >>> G.node['A']
        {'weight': None}
        """
        nodes = list(nodes)

        if weights:
            if len(nodes) != len(weights):
                raise ValueError("The number of elements in nodes and weights"
                                 "should be equal.")
            for index in range(len(nodes)):
                self.add_node(node=nodes[index], weight=weights[index])
        else:
            for node in nodes:
                self.add_node(node=node)

    def add_edge(self, u, v, weight=None):
        """
        Add an edge between u and v.

        The nodes u and v will be automatically added if they are
        not already in the graph.

        Parameters
        ----------
        u, v : nodes
            Nodes can be any hashable Python object.

        weight: int, float (default=None)
            The weight of the edge

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edge(u='Alice', v='Bob')
        >>> G.nodes()
        ['Alice', 'Bob', 'Charles']
        >>> G.edges()
        [('Alice', 'Bob')]

        When the node is not already present in the graph:
        >>> G.add_edge(u='Alice', v='Ankur')
        >>> G.nodes()
        ['Alice', 'Ankur', 'Bob', 'Charles']
        >>> G.edges()
        [('Alice', 'Bob'), ('Alice', 'Ankur')]

        Adding edges with weight:
        >>> G.add_edge('Ankur', 'Maria', weight=0.1)
        >>> G.edge['Ankur']['Maria']
        {'weight': 0.1}
        """
        super(DAG, self).add_edge(u, v, weight=weight)

    def add_edges_from(self, ebunch, weights=None):
        """
        Add all the edges in ebunch.

        If nodes referred in the ebunch are not already present, they
        will be automatically added. Node names can be any hashable python
        object.

        **The behavior of adding weights is different than networkx.

        Parameters
        ----------
        ebunch : container of edges
            Each edge given in the container will be added to the graph.
            The edges must be given as 2-tuples (u, v).

        weights: list, tuple (default=None)
            A container of weights (int, float). The weight value at index i
            is associated with the edge at index i.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_nodes_from(nodes=['Alice', 'Bob', 'Charles'])
        >>> G.add_edges_from(ebunch=[('Alice', 'Bob'), ('Bob', 'Charles')])
        >>> G.nodes()
        ['Alice', 'Bob', 'Charles']
        >>> G.edges()
        [('Alice', 'Bob'), ('Bob', 'Charles')]

        When the node is not already in the model:
        >>> G.add_edges_from(ebunch=[('Alice', 'Ankur')])
        >>> G.nodes()
        ['Alice', 'Bob', 'Charles', 'Ankur']
        >>> G.edges()
        [('Alice', 'Bob'), ('Bob', 'Charles'), ('Alice', 'Ankur')]

        Adding edges with weights:
        >>> G.add_edges_from([('Ankur', 'Maria'), ('Maria', 'Mason')],
        ...                  weights=[0.3, 0.5])
        >>> G.edge['Ankur']['Maria']
        {'weight': 0.3}
        >>> G.edge['Maria']['Mason']
        {'weight': 0.5}
        """
        ebunch = list(ebunch)

        if weights:
            if len(ebunch) != len(weights):
                raise ValueError("The number of elements in ebunch and weights"
                                 "should be equal")
            for index in range(len(ebunch)):
                self.add_edge(ebunch[index][0], ebunch[index][1],
                              weight=weights[index])
        else:
            for edge in ebunch:
                self.add_edge(edge[0], edge[1])

    def get_parents(self, node):
        """
        Returns a list of parents of node.

        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose parents would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[('diff', 'grade'), ('intel', 'grade')])
        >>> G.get_parents(node='grade')
        ['diff', 'intel']
        """
        return list(self.predecessors(node))

    def moralize(self):
        """
        Removes all the immoralities in the DAG and creates a moral
        graph (UndirectedGraph).

        A v-structure X->Z<-Y is an immorality if there is no directed edge
        between X and Y.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG(ebunch=[('diff', 'grade'), ('intel', 'grade')])
        >>> moral_graph = G.moralize()
        >>> moral_graph.edges()
        [('intel', 'grade'), ('intel', 'diff'), ('grade', 'diff')]
        """
        moral_graph = UndirectedGraph(self.to_undirected().edges())

        for node in self.nodes():
            moral_graph.add_edges_from(
                itertools.combinations(self.get_parents(node), 2))

        return moral_graph

    def get_leaves(self):
        """
        Returns a list of leaves of the graph.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> graph = DAG([('A', 'B'), ('B', 'C'), ('B', 'D')])
        >>> graph.get_leaves()
        ['C', 'D']
        """
        return [node for node, out_degree in self.out_degree_iter() if
                out_degree == 0]

    def out_degree_iter(self, nbunch=None, weight=None):
        if nx.__version__.startswith('1'):
            return super(DAG, self).out_degree_iter(nbunch, weight)
        else:
            return iter(self.out_degree(nbunch, weight))

    def in_degree_iter(self, nbunch=None, weight=None):
        if nx.__version__.startswith('1'):
            return super(DAG, self).in_degree_iter(nbunch, weight)
        else:
            return iter(self.in_degree(nbunch, weight))

    def get_roots(self):
        """
        Returns a list of roots of the graph.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> graph = DAG([('A', 'B'), ('B', 'C'), ('B', 'D'), ('E', 'B')])
        >>> graph.get_roots()
        ['A', 'E']
        """
        return [node for node, in_degree in dict(self.in_degree()).items() if in_degree == 0]

    def get_children(self, node):
        """
        Returns a list of children of node.
        Throws an error if the node is not present in the graph.

        Parameters
        ----------
        node: string, int or any hashable python object.
            The node whose children would be returned.

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> g = DAG(ebunch=[('A', 'B'), ('C', 'B'), ('B', 'D'),
                                      ('B', 'E'), ('B', 'F'), ('E', 'G')])
        >>> g.get_children(node='B')
        ['D', 'E', 'F']
        """
        return list(self.successors(node))

    def is_iequivalent(self, model):
        """
        Checks whether the given model is I-equivalent

        Two graphs G1 and G2 are said to be I-equivalent if they have same skeleton
        and have same set of immoralities.

        Note: For same skeleton different names of nodes can work but for immoralities
        names of nodes must be same

        Parameters
        ----------
        model : A DAG object, for which you want to check I-equivalence

        Returns
        --------
        boolean : True if both are I-equivalent, False otherwise

        Examples
        --------
        >>> from pgmpy.base import DAG
        >>> G = DAG()
        >>> G.add_edges_from([('V', 'W'), ('W', 'X'),
        ...                   ('X', 'Y'), ('Z', 'Y')])
        >>> G1 = DAG()
        >>> G1.add_edges_from([('W', 'V'), ('X', 'W'),
        ...                    ('X', 'Y'), ('Z', 'Y')])
        >>> G.is_iequivalent(G1)
        True

        """
        if not isinstance(model, DAG):
            raise TypeError('model must be an instance of DAG')
        skeleton = nx.algorithms.isomorphism.GraphMatcher(self.to_undirected(), model.to_undirected())
        if skeleton.is_isomorphic() and self.get_immoralities() == model.get_immoralities():
            return True
        return False
