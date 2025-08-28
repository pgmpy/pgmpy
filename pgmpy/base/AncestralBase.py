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

        Parameters
        ----------
        ebunch : Iterable[tuple], optional
            An iterable of edges (u, v, u_mark, v_mark) to initialize the graph.
            Each mark must be one of {">", "-", "o"}. Default is None
            which initializes an empty graph.

        latents : set, optional
            Set of latent (unobserved) variables in the graph. Default is
            an empty set.
        """
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.latents = set(latents)
        self.valid_marks = {">", "-", "o"}

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

        Raises
        ------
        ValueError
            If the input matrix is not square or contains invalid marks.
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

        Raises
        ------
        ValueError
            If marks are invalid or nodes are the same.
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
            Each tuple should be of the form (u, v, u_mark, v_mark)."""
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
        """
        return self.get_neighbors(node, u_type=">")

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
        """
        return self.get_neighbors(node, v_type=">")

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
