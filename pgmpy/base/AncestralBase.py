from collections import deque
from typing import Hashable, Iterable, Optional

import networkx as nx
import numpy as np


class AncestralBase(nx.Graph):
    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
    ):
        """
        Ancestral graph base class.
        Internally, each edge is stored with an attribute dictionary
        called ``marks``. The ``marks`` dict maps the two endpoint
        nodes to their respective marks, for example:

        - Directed: ("A", "B", "-", ">") is stored as
          ("A", "B", {"marks": {"A": "-", "B": ">"}})
        - Reverse directed: ("A", "B", ">", "-") is stored as
          ("A", "B", {"marks": {"A": ">", "B": "-"}})
        - Bidirected: ("A", "B", ">", ">") is stored as
          ("A", "B", {"marks": {"A": ">", "B": ">"}})
        - Undirected: ("A", "B", "o", "o") is stored as
          ("A", "B", {"marks": {"A": "o", "B": "o"}})

        Parameters
        ----------
        ebunch : Iterable[tuple], optional
            An iterable of edges of the form (u, v, u_mark, v_mark) used to
            initialize the graph. Each mark must be one of {">", "-", "o"}.
            Default is None, which initializes an empty graph.

        latents : set, optional
            Set of latent (unobserved) variables in the graph. Default is
            an empty set.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> list(graph.edges(data=True))
        [('A', 'B', {'marks': {'A': '-', 'B': '>'}}),
         ('B', 'C', {'marks': {'B': '>', 'C': '-'}})]
        >>> graph.add_edge("C", "D", "o", "o")
        >>> list(graph.edges(data=True))
        [('A', 'B', {'marks': {'A': '-', 'B': '>'}}),
         ('B', 'C', {'marks': {'B': '>', 'C': '-'}}),
         ('C', 'D', {'marks': {'C': 'o', 'D': 'o'}})]
        """
        super().__init__()
        self.valid_marks = {">", "-", "o"}
        if ebunch:
            self.add_edges_from(ebunch)
        self.latents = set(latents)

    @property
    def adjacency_matrix(self):
        """
        Return adjacency matrix with edge marks and node-to-index mapping.

        Returns
        -------
        M : np.ndarray
            A square matrix of shape (n_nodes, n_nodes) where M[i, j]
            is the mark at node j for edge (i, j).

        node_index : dict
            Mapping from node label to row/col index.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("B", "C", ">", "-")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> M, node_index = graph.adjacency_matrix
        >>> print(M)
        [[0 '>' 0]
         ['-' 0 '-']
         [0 '>' 0]]
        >>> print(node_index)
        {'A': 0, 'B': 1, 'C': 2}
        """
        nodes = list(self.nodes)
        n = len(nodes)
        node_index = {node: i for i, node in enumerate(nodes)}

        M = np.full((n, n), 0, dtype=object)

        for u, v, data in self.edges(data=True):
            u_idx, v_idx = node_index[u], node_index[v]
            u_mark = data["marks"][u]
            v_mark = data["marks"][v]

            M[u_idx, v_idx] = v_mark
            M[v_idx, u_idx] = u_mark

        return M, node_index

    @adjacency_matrix.setter
    def adjacency_matrix(self, value):
        """
        Set graph edges from an adjacency matrix with edge marks.

        Parameters
        ----------
        value : np.ndarray
            A square matrix where value[i, j] is the mark at node j
            for edge (i, j). Marks must be one of {">", "-", "o
            or 0 (no edge).

        Returns
        -------
        None

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> M = np.array([[0, ">", 0], ["-", 0, ">"], [0, "-", 0]], dtype=object)
        >>> graph = AncestralBase()
        >>> graph.adjacency_matrix = M
        >>> print(graph.nodes)
        ['X_0', 'X_1', 'X_2']
        >>> print(graph.edges(data=True))
        [('X_0', 'X_1', {'marks': {'X_0': '>', 'X_1': '-'}}), ('X_1', 'X_2', {'marks': {'X_1': '>', 'X_2': '-'}})]
        """
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("Adjacency matrix must be square (n x n).")
        n = value.shape[0]
        variables = [f"X_{i}" for i in range(n)]
        self.clear()
        for i in range(n):
            for j in range(n):
                if i != j:
                    u_mark = value[i, j]
                    v_mark = value[j, i]
                    if u_mark != 0 and v_mark != 0:
                        self.add_edge(variables[i], variables[j], u_mark, v_mark)

    def add_edge(self, u, v, u_mark, v_mark):
        """
        Add an edge with specified marks.

        Parameters
        ----------
        u : Hashable
            One endpoint of the edge.

        v : Hashable
            The other endpoint of the edge.

        u_mark : str
            Mark at node u for edge (u, v). Must be one of {">", "-", "o"}.

        v_mark : str
            Mark at node v for edge (u, v). Must be one of {">",
            "-", "o"}.

        Returns
        -------
        None
            Adds the edge to the graph in-place

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> g = AncestralBase()

        # Directed edge A → B
        >>> g.add_edge("A", "B", "-", ">")
        >>> g["A"]["B"]["marks"]
        {'A': '-', 'B': '>'}

        # Reverse directed edge A ← B
        >>> g.add_edge("A", "C", ">", "-")
        >>> g["A"]["C"]["marks"]
        {'A': '>', 'C': '-'}

        # Bidirected edge A ↔ D
        >>> g.add_edge("A", "D", ">", ">")
        >>> g["A"]["D"]["marks"]
        {'A': '>', 'D': '>'}

        # Undirected edge C — E
        >>> g.add_edge("C", "E", "o", "o")
        >>> g["C"]["E"]["marks"]
        {'C': 'o', 'E': 'o'}
        """
        if u == v:
            raise ValueError("Nodes cannot be the same for an edge.")
        if u_mark not in self.valid_marks or v_mark not in self.valid_marks:
            raise ValueError(f"Marks must be one of {self.valid_marks}.")
        super().add_edge(u, v, marks={u: u_mark, v: v_mark})

    def add_edges_from(self, ebunch):
        """
        Add multiple edges from an iterable of (u, v, marks) tuples.

        Parameters
        ----------
        ebunch : Iterable[tuple]
            Each tuple should be of the form (u, v, u_mark, v_mark).

        Returns
        -------
        None
            Adds the edges to the graph in-place.


        Examples
        --------
        >>> g = AncestralBase()
        >>> edges = [("A", "B", "-", ">"), ("B", "C", ">", "-"), ("C", "D", "o", "o")]
        >>> g.add_edges_from(edges)
        >>> list(g.edges(data=True))
        [('A', 'B', {'marks': {'A': '-', 'B': '>'}}),
         ('B', 'C', {'marks': {'B': '>', 'C': '-'}}),
         ('C', 'D', {'marks': {'C': 'o', 'D': 'o'}})]
        """
        for u, v, u_mark, v_mark in ebunch:
            self.add_edge(u, v, u_mark, v_mark)

    def get_neighbors(self, node, u_type=None, v_type=None):
        """
        Get neighbors of a node with optional edge mark constraints.

        Parameters
        ----------
        node : Hashable
            The node whose neighbors are to be found.

        u_type : Optional[str]
            Required mark at the given node for the edge.

        v_type : Optional[str]
            Required mark at the neighbor node for the edge.

        Returns
        -------
        neighbors : set
            Set of neighboring nodes satisfying the mark constraints.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("B", "C", ">", "-"), ("C", "D", "o", "o")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_neighbors("B"))
        {'A', 'C'}
        >>> print(graph.get_neighbors("B", u_type=">"))
        {'C'}
        >>> print(graph.get_neighbors("B", v_type="-"))
        {'A'}
        >>> print(graph.get_neighbors("B", u_type=">", v_type="-"))
        {'C'}
        """
        if node not in self:
            return set()
        neighbors = set()
        for neighbor in nx.all_neighbors(self, node):

            node_mark, neighbor_mark = (
                self.edges[node, neighbor]["marks"][node],
                self.edges[node, neighbor]["marks"][neighbor],
            )

            if (u_type is None or node_mark == u_type) and (
                v_type is None or neighbor_mark == v_type
            ):
                neighbors.add(neighbor)

        return neighbors

    def get_parents(self, node):
        """
        Get nodes that point to this node with '>'

        Parameters
        ----------
        node : Hashable
            The node whose parents are to be found.

        Returns
        -------
        parents : set
            Set of parent nodes.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("C", "B", "-", ">"), ("B", "D", "-", ">")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_parents("B"))
        {'A', 'C'}
        >>> print(graph.get_parents("D"))
        {'B'}
        >>> print(graph.get_parents("A"))
        set()
        """
        return self.get_neighbors(node, u_type=">", v_type="-")

    def get_children(self, node):
        """
        Get nodes that this node points to with '>'

        Parameters
        ----------
        node : Hashable
            The node whose children are to be found.

        Returns
        -------
        children : set
            Set of child nodes.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", "-", ">"), ("A", "C", "-", ">"), ("B", "D", "-", ">")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_children("A"))
        {'B', 'C'}
        >>> print(graph.get_children("B"))
        {'D'}
        >>> print(graph.get_children("D"))
        set()
        """
        return self.get_neighbors(node, u_type="-", v_type=">")

    def get_spouses(self, node):
        """
        Get nodes connected by bidirectional '>' edges (spouses).

        Parameters
        ----------
        node : Hashable
            The node whose spouses are to be found.

        Returns
        -------
        spouses : set
            Set of spouse nodes.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [("A", "B", ">", ">"), ("A", "C", "-", ">"), ("C", "D", ">", ">")]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_spouses("A"))
        {'B'}
        >>> print(graph.get_spouses("C"))
        {'D'}
        >>> print(graph.get_spouses("B"))
        set()
        """
        return self.get_neighbors(node, u_type=">", v_type=">")

    def get_ancestors(self, node):
        """
        Get all ancestor nodes of the given node.

        Parameters
        ----------
        node : Hashable
            The node whose ancestors are to be found.

        Returns
        -------
        ancestors : set
            Set of ancestor nodes including the starting node.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [
        ...     ("A", "B", "-", ">"),
        ...     ("B", "C", "-", ">"),
        ...     ("C", "D", "-", ">"),
        ...     ("E", "C", "-", ">"),
        ... ]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_ancestors("D"))
        {'A', 'B', 'C', 'D', 'E'}
        >>> print(graph.get_ancestors("C"))
        {'A', 'B', 'C', 'E'}
        >>> print(graph.get_ancestors("A"))
        {'A'}
        """
        ancestors = set()
        visited = set()
        queue = deque(node)

        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                ancestors.add(current)
                queue.extend(self.get_parents(current))
        return ancestors

    def get_descendants(self, node):
        """
        Get all descendant nodes (children, grandchildren, etc.)

        Parameters
        ----------
        node : Hashable
            The starting node.

        Returns
        -------
        descendants : set
            Set of descendant nodes including the starting node.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [
        ...     ("A", "B", "-", ">"),
        ...     ("B", "C", "-", ">"),
        ...     ("C", "D", "-", ">"),
        ...     ("B", "E", "-", ">"),
        ... ]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_descendants("A"))
        {'A', 'B', 'C', 'D', 'E'}
        >>> print(graph.get_descendants("B"))
        {'B', 'C', 'D', 'E'}
        >>> print(graph.get_descendants("D"))
        {'D'}
        """
        descendants = set()
        visited = set()
        queue = deque(node)

        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                descendants.add(current)
                queue.extend(self.get_children(current))
        return descendants

    def get_reachable_nodes(self, node, u_type=None, v_type=None):
        """
        Get all nodes reachable from the given node following edges
        with specified marks.

        Parameters
        ----------
        node : Hashable
            The starting node.

        u_type : Optional[str]
            Required mark at the current node for traversal.

        v_type : Optional[str]
            Required mark at the neighbor node for traversal.

        Returns
        -------
        reachable : set
            Set of reachable nodes including the starting node.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        >>> edges = [
        ...     ("A", "B", "-", ">"),
        ...     ("B", "C", "-", ">"),
        ...     ("A", "D", "o", "o"),
        ...     ("D", "E", "o", "o"),
        ... ]
        >>> graph = AncestralBase(ebunch=edges)
        >>> print(graph.get_reachable_nodes("A", v_type=">"))
        {'A', 'B', 'C'}
        >>> print(graph.get_reachable_nodes("A", u_type="o", v_type="o"))
        {'A', 'D', 'E'}
        """
        reachable = set()
        visited = set()
        queue = deque(node)

        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                reachable.add(current)
                queue.extend(self.get_neighbors(current, u_type=u_type, v_type=v_type))
        return reachable
