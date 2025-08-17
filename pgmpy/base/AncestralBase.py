#!/usr/bin/env python

import networkx as nx


class AncestralBase:
    """
    Base class for all ancestral graphical models.

    Wraps a networkx.DiGraph to handle mixed graphs with
    directed, bidirected, and partially directed edges.
    """

    def __init__(self, ebunch=None):
        self.graph = nx.DiGraph()
        if ebunch:
            self.add_edges_from(ebunch)

    def add_edge(self, u, v, u_mark, v_mark):
        if u_mark not in {"tail", "arrowhead", "circle"} or v_mark not in {
            "tail",
            "arrowhead",
            "circle",
        }:
            raise ValueError("Marks must be one of 'tail', 'arrowhead', or 'circle'.")

        self.graph.add_edge(u, v, mark=v_mark)

        if self._is_symmetric_edge(u_mark, v_mark):
            self.graph.add_edge(v, u, mark=u_mark)

    def add_edges_from(self, ebunch):
        for u, v, u_mark, v_mark in ebunch:
            self.add_edge(u, v, u_mark, v_mark)

    def _is_symmetric_edge(self, u_mark, v_mark):
        return (u_mark == "arrowhead" and v_mark == "arrowhead") or (
            u_mark == "circle" and v_mark == "circle"
        )

    def is_directed(self, u, v):
        if not self.graph.has_edge(u, v):
            return False
        if self.graph.has_edge(v, u):
            return (
                self.graph.get_edge_data(u, v).get("mark") == "arrowhead"
                and self.graph.get_edge_data(v, u).get("mark") == "tail"
            )
        return self.graph.get_edge_data(u, v).get("mark") == "arrowhead"

    def is_bidirected(self, u, v):
        return (
            self.graph.has_edge(u, v)
            and self.graph.has_edge(v, u)
            and self.graph.get_edge_data(u, v).get("mark") == "arrowhead"
            and self.graph.get_edge_data(v, u).get("mark") == "arrowhead"
        )

    def has_arrowhead_at(self, u, v):
        return (
            self.graph.has_edge(u, v)
            and self.graph.get_edge_data(u, v).get("mark") == "arrowhead"
        )

    def has_circle_at(self, u, v):
        return (
            self.graph.has_edge(u, v)
            and self.graph.get_edge_data(u, v).get("mark") == "circle"
        )

    def has_tail_at(self, u, v):
        return (
            self.graph.has_edge(u, v)
            and self.graph.get_edge_data(u, v).get("mark") == "tail"
        )

    def get_parents(self, node):
        if node not in self.graph:
            return set()
        parents = set()
        for neighbor in self.graph.predecessors(node):
            if (
                self.graph.has_edge(neighbor, node)
                and self.graph.has_edge(node, neighbor)
                and self.graph.get_edge_data(neighbor, node).get("mark") == "arrowhead"
                and self.graph.get_edge_data(node, neighbor).get("mark") == "tail"
            ):
                parents.add(neighbor)
        return parents

    def get_children(self, node):
        if node not in self.graph:
            return set()
        children = set()
        for neighbor in self.graph.successors(node):
            if (
                self.graph.has_edge(node, neighbor)
                and self.graph.has_edge(neighbor, node)
                and self.graph.get_edge_data(node, neighbor).get("mark") == "arrowhead"
                and self.graph.get_edge_data(neighbor, node).get("mark") == "tail"
            ):
                children.add(neighbor)
        return children

    def get_spouses(self, node):
        if node not in self.graph:
            return set()
        spouses = set()
        for neighbor in self.graph.successors(node):
            if (
                self.graph.has_edge(node, neighbor)
                and self.graph.has_edge(neighbor, node)
                and self.graph.get_edge_data(node, neighbor).get("mark") == "arrowhead"
                and self.graph.get_edge_data(neighbor, node).get("mark") == "arrowhead"
            ):
                spouses.add(neighbor)
        return spouses

    def get_ancestors(self, node):
        if node not in self.graph:
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
        if node not in self.graph:
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
