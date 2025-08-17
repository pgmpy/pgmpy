#!/usr/bin/env python

import networkx as nx


class AncestralBase(nx.DiGraph):
    """
    Base class for all ancestral graphical models.

    This class extends the pgmpy.base.DAG to handle mixed graphs with
    directed, bidirected, and partially directed edges. Edges are represented
    with an attribute that specifies the marks at each endpoint.

    The valid marks are:
    - 'tail': A tail mark (e.g., in A -> B, A has a tail)
    - 'arrowhead': An arrowhead mark (e.g., in A -> B, B has an arrowhead)
    - 'circle': A circle mark (e.g., in A o-> B, A has a circle)
    """

    def __init__(self, ebunch=None):
        """
        Initializes an empty AncestralGraph.

        Parameters
        ----------
        ebunch : list, optional
            A container of edges. Each edge must be a tuple of the form
            (u, v, u_mark, v_mark), where u and v are the nodes, and u_mark
            and v_mark are the marks at the endpoints of u and v respectively.
            Example: [('A', 'B', 'tail', 'arrowhead')] for A -> B.
        """
        super().__init__()
        self.graph = nx.DiGraph()

        if ebunch:
            self.add_edges_from(ebunch)

    def add_edge(self, u, v, u_mark, v_mark):
        """
        Adds an edge between nodes u and v with specified endpoint marks.

        Parameters
        ----------
        u : hashable
            The starting node of the edge.

        v : hashable
            The ending node of the edge.

        u_mark : str
            The mark at the 'u' endpoint. Must be one of 'tail', 'arrowhead', 'circle'.

        v_mark : str
            The mark at the 'v' endpoint. Must be one of 'tail', 'arrowhead', 'circle'.
        """
        if u_mark not in {"tail", "arrowhead", "circle"} or v_mark not in {
            "tail",
            "arrowhead",
            "circle",
        }:
            raise ValueError("Marks must be one of 'tail', 'arrowhead', or 'circle'.")

        self.graph.add_edge(u, v, mark=v_mark)

        if self._is_symmetric_edge(u_mark, v_mark):
            self.graph.add_edge(v, u, mark=u_mark)
        elif self.graph.has_edge(v, u):
            pass

    def add_edges_from(self, ebunch):
        for u, v, u_mark, v_mark in ebunch:
            self.add_edge(u, v, u_mark, v_mark)

    def _is_symmetric_edge(self, u_mark, v_mark):

        return (u_mark == "arrowhead" and v_mark == "arrowhead") or (
            u_mark == "circle" and v_mark == "circle"
        )

    def is_directed(self, u, v):

        if self.graph.has_edge(u, v) and self.graph.has_edge(v, u):
            return (
                self.graph.get_edge_data(u, v).get("mark") == "arrowhead"
                and self.graph.get_edge_data(v, u).get("mark") == "tail"
            )
        else:
            return (
                self.graph.has_edge(u, v)
                and self.graph.get_edge_data(u, v).get("mark") == "arrowhead"
            )

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

        parents = set()
        for neighbor in self.graph.predecessors(node):
            if (
                self.graph.get_edge_data(neighbor, node).get("mark") == "arrowhead"
                and self.graph.get_edge_data(node, neighbor).get("mark") == "tail"
            ):
                parents.add(neighbor)
        return parents

    def get_children(self, node):

        children = set()
        for neighbor in self.graph.adj[node]:
            if (
                self.graph.get_edge_data(node, neighbor).get("mark") == "arrowhead"
                and self.graph.get_edge_data(neighbor, node).get("mark") == "tail"
            ):
                children.add(neighbor)
        return children

    def get_spouses(self, node):

        spouses = set()
        for neighbor in self.graph.adj[node]:
            if (
                self.graph.get_edge_data(node, neighbor).get("mark") == "arrowhead"
                and self.graph.get_edge_data(neighbor, node).get("mark") == "arrowhead"
            ):
                spouses.add(neighbor)
        return spouses

    def get_ancestors(self, node):

        ancestors = set()
        queue = list(self.get_parents(node))
        visited = set(queue)

        while queue:
            current_node = queue.pop(0)
            ancestors.add(current_node)
            for parent in self.get_parents(current_node):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(parent)
        return ancestors

    def get_descendants(self, node):

        descendants = set()
        queue = list(self.get_children(node))
        visited = set(queue)

        while queue:
            current_node = queue.pop(0)
            descendants.add(current_node)
            for child in self.get_children(current_node):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
        return descendants
