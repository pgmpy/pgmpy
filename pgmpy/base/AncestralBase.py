from collections import deque
from typing import Hashable, Iterable, Optional

import networkx as nx
import numpy as np


class AncestralBase(nx.DiGraph):
    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
    ):
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)
        self.latents = set(latents)

    def to_adjacency_matrix(self):
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

        M = np.full((n, n), "", dtype=object)

        for u, v, data in self.edges(data=True):
            u_idx, v_idx = node_index[u], node_index[v]
            u_mark, v_mark = data["marks"]

            # Mark from u→v is v_mark (mark at v's end)
            M[u_idx, v_idx] = v_mark
            # Mark from v→u is u_mark (mark at u's end)
            M[v_idx, u_idx] = u_mark

        return M, node_index

    def add_edge(self, u, v, u_mark, v_mark):
        if u == v:
            raise ValueError("Nodes cannot be the same for an edge.")
        if u_mark not in {"-", ">", "o"} or v_mark not in {"-", ">", "o"}:
            raise ValueError("Marks must be one of '-', '>', or 'o'.")
        super().add_edge(u, v, marks=(u_mark, v_mark))

    def add_edges_from(self, ebunch):
        for u, v, u_mark, v_mark in ebunch:
            self.add_edge(u, v, u_mark, v_mark)

    def _get_marks(self, u, v):
        """
        Return (mark_at_u, mark_at_v) for the edge between u and v.
        Works regardless of stored edge direction.
        """
        data = self.get_edge_data(u, v)
        if data is not None:  # edge stored as (u, v)
            return data["marks"]
        data = self.get_edge_data(v, u)
        if data is not None:  # edge stored as (v, u) → reverse marks
            u_mark, v_mark = data["marks"]
            return v_mark, u_mark
        raise ValueError(f"No edge between {u} and {v}")

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
            try:
                # Get marks: node_mark is mark at node, neighbor_mark is mark at neighbor
                node_mark, neighbor_mark = self._get_marks(node, neighbor)

                # u_type constraint: mark at neighbor when going FROM neighbor TO node
                # v_type constraint: mark at node when going FROM node TO neighbor
                if (u_type is None or node_mark == u_type) and (
                    v_type is None or neighbor_mark == v_type
                ):
                    neighbors.add(neighbor)
            except ValueError:
                continue
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
        visited = set([node])
        queue = deque(self.get_parents(node))

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
        visited = set([node])
        queue = deque(self.get_children(node))

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
        visited = set([node])
        queue = deque(self.get_neighbors(node, u_type=u_type, v_type=v_type))

        while queue:
            current = queue.popleft()
            if current not in visited:
                visited.add(current)
                reachable.add(current)
                queue.extend(self.get_neighbors(current, u_type=u_type, v_type=v_type))
        return reachable
