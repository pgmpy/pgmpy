#!/usr/bin/env python

import networkx as nx


class AncestralBase(nx.DiGraph):
    """
    Base class for all ancestral graphical models.

    This class extends nx.DiGraph to handle mixed graphs with
    directed, bidirected, and partially directed edges. Edges are represented
    with an attribute that specifies the marks at each endpoint.

    The valid marks are:
    - 'tail': A tail mark (e.g., in A -> B, A has a tail)
    - 'arrowhead': An arrowhead mark (e.g., in A -> B, B has an arrowhead)
    - 'circle': A circle mark (e.g., in A o-> B, A has a circle)
    """

    def __init__(self, ebunch=None):
        super().__init__()
        if ebunch:
            self.add_edges_from(ebunch)

    def add_edge(self, u, v, u_mark, v_mark):
        if u_mark not in {"tail", "arrowhead", "circle"} or v_mark not in {
            "tail",
            "arrowhead",
            "circle",
        }:
            raise ValueError("Marks must be one of 'tail', 'arrowhead', or 'circle'.")

        super().add_edge(u, v, mark=v_mark)

        if self._is_symmetric_edge(u_mark, v_mark):
            super().add_edge(v, u, mark=u_mark)

    def add_edges_from(self, ebunch):
        for u, v, u_mark, v_mark in ebunch:
            self.add_edge(u, v, u_mark, v_mark)

    def _is_symmetric_edge(self, u_mark, v_mark):
        return (u_mark == "arrowhead" and v_mark == "arrowhead") or (
            u_mark == "circle" and v_mark == "circle"
        )

    def is_directed(self, u, v):
        if not self.has_edge(u, v):
            return False
        if self.has_edge(v, u):
            return (
                self.get_edge_data(u, v).get("mark") == "arrowhead"
                and self.get_edge_data(v, u).get("mark") == "tail"
            )
        return self.get_edge_data(u, v).get("mark") == "arrowhead"

    def is_bidirected(self, u, v):
        return (
            self.has_edge(u, v)
            and self.has_edge(v, u)
            and self.get_edge_data(u, v).get("mark") == "arrowhead"
            and self.get_edge_data(v, u).get("mark") == "arrowhead"
        )

    def has_arrowhead_at(self, u, v):
        return (
            self.has_edge(u, v) and self.get_edge_data(u, v).get("mark") == "arrowhead"
        )

    def has_circle_at(self, u, v):
        return self.has_edge(u, v) and self.get_edge_data(u, v).get("mark") == "circle"

    def has_tail_at(self, u, v):
        return self.has_edge(u, v) and self.get_edge_data(u, v).get("mark") == "tail"

    def get_parents(self, node):
        if node not in self:
            return set()
        parents = set()
        for neighbor in self.predecessors(node):
            if (
                self.has_edge(neighbor, node)
                and self.has_edge(node, neighbor)
                and self.get_edge_data(neighbor, node).get("mark") == "arrowhead"
                and self.get_edge_data(node, neighbor).get("mark") == "tail"
            ):
                parents.add(neighbor)
        return parents

    def get_children(self, node):
        if node not in self:
            return set()
        children = set()
        for neighbor in self.successors(node):
            if (
                self.has_edge(node, neighbor)
                and self.has_edge(neighbor, node)
                and self.get_edge_data(node, neighbor).get("mark") == "arrowhead"
                and self.get_edge_data(neighbor, node).get("mark") == "tail"
            ):
                children.add(neighbor)
        return children

    def get_spouses(self, node):
        if node not in self:
            return set()
        spouses = set()
        for neighbor in self.successors(node):
            if (
                self.has_edge(node, neighbor)
                and self.has_edge(neighbor, node)
                and self.get_edge_data(node, neighbor).get("mark") == "arrowhead"
                and self.get_edge_data(neighbor, node).get("mark") == "arrowhead"
            ):
                spouses.add(neighbor)
        return spouses

    def get_ancestors(self, node):
        if node not in self:
            return set()
        ancestors = set()
        queue = list(self.get_parents(node))
        visited = set(queue)

        while queue:
            current = queue.pop(0)
            ancestors.add(current)
            for parent in self.get_parents(current):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return ancestors

    def get_descendants(self, node):
        if node not in self:
            return set()
        descendants = set()
        queue = list(self.get_children(node))
        visited = set(queue)

        while queue:
            current = queue.pop(0)
            descendants.add(current)
            for child in self.get_children(current):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return descendants
