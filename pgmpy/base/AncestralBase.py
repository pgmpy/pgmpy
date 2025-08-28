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
            u_mark, v_mark = data["marks"]

            M[u_idx, v_idx] = v_mark
            M[v_idx, u_idx] = u_mark

        return M, node_index

    @adjacency_matrix.setter
    def adjacency_matrix(self, value):
        value = np.asarray(value)
        if value.ndim != 2 or value.shape[0] != value.shape[1]:
            raise ValueError("Adjacency matrix must be square (n x n).")
        n = value.shape[0]
        variables = [f"X_{i}" for i in range(n)]
        self.clear()
        for i in variables:
            for j in variables:
                if i != j:
                    u_mark = value[i, j]
                    v_mark = value[j, i]
                    if u_mark != 0 and v_mark != 0:
                        self.add_edge(i, j, u_mark, v_mark)

    def add_edge(self, u, v, u_mark, v_mark):
        if u == v:
            raise ValueError("Nodes cannot be the same for an edge.")
        if u_mark not in self.valid_marks or v_mark not in self.valid_marks:
            raise ValueError(f"Marks must be one of {self.valid_marks}.")
        super().add_edge(u, v, marks={u: u_mark, v: v_mark})

    def add_edges_from(self, ebunch):
        for u, v, marks in ebunch:
            self.add_edge(u, v, marks)

    def get_neighbors(self, node, u_type=None, v_type=None):
        """
        Return neighbors of a node that satisfy edge mark constraints.
        u_type = mark at neighbor's side (when going FROM neighbor TO node)
        v_type = mark at node's side (when going FROM node TO neighbor)
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
        """Get nodes that have '>' pointing TO this node"""
        return self.get_neighbors(node, u_type=">")

    def get_children(self, node):
        """Get nodes that this node has '>' pointing TO"""
        return self.get_neighbors(node, v_type=">")

    def get_spouses(self, node):
        """Get nodes with bidirectional '>' marks"""
        return self.get_neighbors(node, u_type=">", v_type=">")

    def get_ancestors(self, node):
        """Get all ancestor nodes (parents, grandparents, etc.)"""
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
        """Get all descendant nodes (children, grandchildren, etc.)"""
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
        Get all the nodes reachable from the given node
        with a certain type of edge marks.
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
