#!/usr/bin/env python

from itertools import permutations, combinations, chain
from typing import (
    Callable,
    Collection,
    Dict,
    FrozenSet,
    Hashable,
    Optional,
    Set,
    Tuple,
    Union,
)

import pandas as pd
import networkx as nx
from tqdm import tqdm

from pgmpy.base.DAG import DAG



class AncestralGraph(DAG):
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
        ebunch : list-like, optional
            A container of edges. Each edge must be a tuple of the form
            (u, v, u_mark, v_mark), where u and v are the nodes, and u_mark
            and v_mark are the marks at the endpoints of u and v respectively.
            Example: [('A', 'B', 'tail', 'arrowhead')] for A -> B.
        """
        super().__init__()
        # Use NetworkX for the underlying graph structure.
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
        # A single 'add_edge' call might not be enough for bidirected or undirected edges,
        # as NetworkX's DiGraph treats u->v and v->u as separate edges.
        # We store the mark at the 'v' end for the u->v edge.
        # For symmetric edges (bidirected, undirected, etc.), we need to add two edges.
        if u_mark not in {"tail", "arrowhead", "circle"} or v_mark not in {
            "tail",
            "arrowhead",
            "circle",
        }:
            raise ValueError("Marks must be one of 'tail', 'arrowhead', or 'circle'.")

        self.add_node(u)
        self.add_node(v)

        # Store the mark for the edge u -> v
        self.graph.add_edge(u, v, mark=v_mark)

        # For symmetric edges, we must also add the reverse edge
        # and store its mark.
        if self._is_symmetric_edge(u_mark, v_mark):
            self.graph.add_edge(v, u, mark=u_mark)
        elif self.graph.has_edge(v, u):
            # If a reverse edge already exists, and the new edge is not symmetric,
            # this is an invalid operation for this implementation's logic.
            # A more robust implementation might handle mixed edge types between nodes.
            pass

    def add_edges_from(self, ebunch):
        """
        Adds multiple edges from an iterable container.

        Parameters
        ----------
        ebunch : list-like
            A container of edges, where each edge is a tuple of the form
            (u, v, u_mark, v_mark).
        """
        for u, v, u_mark, v_mark in ebunch:
            self.add_edge(u, v, u_mark, v_mark)

    def _is_symmetric_edge(self, u_mark, v_mark):
        """
        Helper function to determine if an edge type is symmetric.
        This includes bidirected (<->) and undirected (o-o) edges.
        """
        return (u_mark == "arrowhead" and v_mark == "arrowhead") or (
            u_mark == "circle" and v_mark == "circle"
        )

    def is_directed(self, u, v):
        """
        Checks if there is a directed edge from u to v (u -> v).
        """
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
        """
        Checks if there is a bidirected edge between u and v (u <-> v).
        This is a symmetric relationship.
        """
        return (
            self.graph.has_edge(u, v)
            and self.graph.has_edge(v, u)
            and self.graph.get_edge_data(u, v).get("mark") == "arrowhead"
            and self.graph.get_edge_data(v, u).get("mark") == "arrowhead"
        )

    def has_arrowhead_at(self, u, v):
        """
        Checks if the edge between u and v has an arrowhead at the v endpoint.
        """
        return (
            self.graph.has_edge(u, v)
            and self.graph.get_edge_data(u, v).get("mark") == "arrowhead"
        )

    def has_circle_at(self, u, v):
        """
        Checks if the edge between u and v has a circle at the v endpoint.
        """
        return (
            self.graph.has_edge(u, v)
            and self.graph.get_edge_data(u, v).get("mark") == "circle"
        )

    def has_tail_at(self, u, v):
        """
        Checks if the edge between u and v has a tail at the v endpoint.
        """
        return (
            self.graph.has_edge(u, v)
            and self.graph.get_edge_data(u, v).get("mark") == "tail"
        )

    ## get all the relationahips between the nodes ##

    # Only for directed edegs for now
    def get_parents(self, node):
        """
        Returns a set of all parents of the given node.

        A node 'p' is a parent of 'c' if there is a directed edge p -> c.

        Parameters
        ----------
        node : hashable
            The node for which to find parents.

        Returns
        -------
        set
            A set of all parents of `node`.
        """
        parents = set()
        for neighbor in self.graph.predecessors(node):
            if (
                self.graph.get_edge_data(neighbor, node).get("mark") == "arrowhead"
                and self.graph.get_edge_data(node, neighbor).get("mark") == "tail"
            ):
                parents.add(neighbor)
        return parents

    def get_children(self, node):
        """
        Returns a set of all children of the given node.

        A node 'c' is a child of 'p' if there is a directed edge p -> c.

        Parameters
        ----------
        node : hashable
            The node for which to find children.

        Returns
        -------
        set
            A set of all children of `node`.
        """
        children = set()
        for neighbor in self.graph.adj[node]:
            if (
                self.graph.get_edge_data(node, neighbor).get("mark") == "arrowhead"
                and self.graph.get_edge_data(neighbor, node).get("mark") == "tail"
            ):
                children.add(neighbor)
        return children

    # A method only for bidirected edges
    def get_spouses(self, node):
        """
        Returns a set of all spouses of the given node.

        A node 's' is a spouse of 'u' if there is a bidirected edge u <-> s.

        Parameters
        ----------
        node : hashable
            The node for which to find spouses.

        Returns
        -------
        set
            A set of all spouses of `node`.
        """
        spouses = set()
        for neighbor in self.graph.adj[node]:
            if (
                self.graph.get_edge_data(node, neighbor).get("mark") == "arrowhead"
                and self.graph.get_edge_data(neighbor, node).get("mark") == "arrowhead"
            ):
                spouses.add(neighbor)
        return spouses

    # Implementation logic only works for directed edges for now
    def get_ancestors(self, node):
        """
        Returns a set of all ancestors of the given node.

        An ancestor of a node 'n' is any node from which there is a directed
        path ending at 'n'.

        Parameters
        ----------
        node : hashable
            The node for which to find ancestors.

        Returns
        -------
        set
            A set of all ancestors of `node`.
        """

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

    # Implementation logic only works for directed edges for now
    def get_descendants(self, node):
        """
        Returns a set of all descendants of the given node.

        A descendant of a node 'n' is any node to which there is a directed
        path starting from 'n'.

        Parameters
        ----------
        node : hashable
            The node for which to find descendants.

        Returns
        -------
        set
            A set of all descendants of `node`.
        """
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
