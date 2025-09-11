from typing import Hashable, Iterable, Optional

import networkx as nx

from pgmpy.base import AncestralBase


class MAG(AncestralBase):
    """
    Class for representing Maximal Ancestral Graphs (MAGs).

    A MAG is a type of graph used in causal inference to represent conditional
    independence relations when some variables are latent (unobserved). Unlike
    simple directed acyclic graphs (DAGs), MAGs allow for special edge types
    (directed and bidirected) that capture the presence of latent confounding
    and selection bias. Every pair of nodes in a MAG is connected in such a way
    that the graph is "maximal," meaning no additional edges can be added
    without changing the set of implied conditional independence relations.

    References
    ----------
    [1] Zhang, J. (2008). Causal Reasoning with Ancestral Graphs. Journal of Machine Learning Research, 9(7).
    """

    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
        roles=None,
    ):
        """
        Initialize a Maximal Ancestral Graph

        Parameters
        ----------
        ebunch : iterable of tuples, optional
            A list or iterable of edges to add at initialization.

        latents : set, default=set()
            Set of latent (unobserved) variables.

        roles : dict, optional (default: None)
            A dictionary mapping roles to node names.
            The keys are roles, and the values are role names (strings or iterables of str).
            If provided, this will automatically assign roles to the nodes in the graph.
            Passing a key-value pair via ``roles`` is equivalent to calling
            ``with_role(role, variables)`` for each key-value pair in the dictionary.

        Returns
        -------
        MAG
            A new instance of a Maximal Ancestral Graph.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG(
        ...     ebunch=[("L", "A", "-", ">"), ("B", "C", "-", ">")], latents={"L"}
        ... )
        >>> sorted(mag.nodes())
        ['A', 'B', 'C', 'L']

        Roles can be assigned to nodes in the graph at construction or using methods.

        At construction:

        >>> mag = MAG(
        ...     ebunch=[("L", "A", "-", ">"), ("B", "C", "-", ">")],
        ...     latents={"L"},
        ...     roles={"exposure": "A", "outcome": "B"},
        ... )

        Roles can also be assigned after creation using ``with_role`` method.

        >>> mag = mag.with_role("adjustment", {"L", "C"})

        Vertices of a specific role can be retrieved using ``get_role`` method.

        >>> mag.get_role("exposure")
        ["A"]
        >>> mag.get_role("adjustment")
        ["L", "C"]

        """
        if ebunch:
            for _, _, u_mark, v_mark in ebunch:
                if (u_mark, v_mark) not in {
                    ("-", ">"),
                    (">", "-"),
                    (">", ">"),
                    ("-", "-"),
                }:
                    raise ValueError(
                        f"Invalid edge type ({u_mark}, {v_mark}). "
                        "MAGs only allow directed ('-', '>'), reverse directed ('>', '-'), "
                        "bidirected ('>', '>'), and undirected ('-', '-') edges."
                    )
        super().__init__(ebunch=ebunch, latents=latents, roles=roles)

    def _is_collider(self, u, c, v):
        """
        Check if a node is a collider in a path u - c - v.

        A collider is a node with incoming arrowheads on both sides:
        u -> c <- v.

        Parameters
        ----------
        u : Hashable
            The first endpoint in the triplet (u, c, v).

        c : Hashable
            The middle node, candidate collider.

        v : Hashable
            The second endpoint in the triplet.

        Returns
        -------
        bool
            True if `c` is a collider on the path, False otherwise.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "Z", "-", ">")
        >>> mag.add_edge("Y", "Z", "-", ">")
        >>> mag._is_collider("X", "Z", "Y")
        True
        """
        if not (self.has_edge(u, c) and self.has_edge(c, v)):
            return False

        mark_uc_at_c = self.edges[u, c]["marks"][c]
        mark_cv_at_c = self.edges[c, v]["marks"][c]

        return mark_uc_at_c == ">" and mark_cv_at_c == ">"

    def has_inducing_path(self, u, v, W):
        """
        Check if there exists an inducing path between two nodes relative to W.

        An inducing path between u and v is a path such that:
        - The path has length > 2 (at least one intermediate node),
        - Every intermediate node is a collider on the path,
        - Every intermediate node is either:
            * in W, or
            * an ancestor of u or v.

        Parameters
        ----------
        u : Hashable
            Source node.

        v : Hashable
            Target node.

        W : set
            Subset of nodes to check inducing paths through (often latents).

        Returns
        -------
        bool
            True if there exists an inducing path, False otherwise.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "L", "-", ">")
        >>> mag.add_edge("Y", "L", "-", ">")
        >>> mag.latents = {"L"}
        >>> mag.has_inducing_path("X", "Y", mag.latents)
        True
        """

        is_inducing = True
        for path in nx.all_simple_paths(self, source=u, target=v):
            if len(path) <= 2:
                continue

            for i in range(1, len(path) - 1):
                prev_node, curr_node, next_node = path[i - 1], path[i], path[i + 1]

                if not self._is_collider(prev_node, curr_node, next_node):
                    is_inducing = False
                    break

                ancestors_uv_vu = self.get_ancestors(u).union(self.get_ancestors(v))
                if curr_node not in W and curr_node not in ancestors_uv_vu:
                    is_inducing = False
                    break

        return is_inducing

    def is_visible_edge(self, u, v) -> bool:
        """
        Check if a directed edge u -> v is visible in the MAG.

        A directed edge A → B in a MAG is considered visible if there exists a vertex C
        not adjacent to B such that either:
            1. C → A exists, or
            2. There is a collider path from C to A that is into A, and every vertex
            on that path is a parent of B.

        Parameters
        ----------
        u : Hashable
            Source node (tail of the edge).

        v : Hashable
            Target node (head of the edge).

        Returns
        -------
        bool
            True if the edge u -> v is visible, False otherwise.

        Examples
        --------
        >>> edges = [
        ...     ("A", "D", "-", ">"),
        ...     ("B", "C", "-", ">"),
        ...     ("X", "A", "-", ">"),
        ... ]
        >>> mag = MAG(ebunch=edges)
        >>> mag.is_visible_edge("A", "D")
        True
        >>> mag.is_visible_edge("B", "C")
        False
        """
        if not self.has_edge(u, v):
            return False
        marks = self.edges[u, v]["marks"]
        if marks.get(u) != "-" or marks.get(v) != ">":
            return False

        neighbors_v = set(self.neighbors(v))

        for c in self.nodes:
            if c in {u, v} or c in neighbors_v:
                continue

            if self.has_edge(c, u):
                cm = self.edges[c, u]["marks"]
                if cm.get(u) == ">":
                    return True

            for path in nx.all_simple_paths(self, source=c, target=u):
                if len(path) < 3:
                    continue

                last = path[-2]
                if not self.has_edge(last, u):
                    continue
                if self.edges[last, u]["marks"][u] != ">":
                    continue

                valid = True
                for i in range(1, len(path) - 1):
                    prev_node, curr_node, next_node = path[i - 1], path[i], path[i + 1]

                    if not self._is_collider(prev_node, curr_node, next_node):
                        valid = False
                        break

                    if not (
                        self.has_edge(curr_node, v)
                        and self.edges[curr_node, v]["marks"].get(curr_node) == "-"
                        and self.edges[curr_node, v]["marks"].get(v) == ">"
                    ):
                        valid = False
                        break

                if valid:
                    return True

        return False

    def lower_manipulation(self, X, inplace=False):
        """
        Performs lower manipulation.

        Removes all edges that are visible and originate from nodes in X.
        For edges from X that are invisible, adds bidirected edges from the other
        endpoint to its neighbors outside X to preserve conditional independencies.
        All other edges remain unchanged.

        Parameters
        ----------
        X : set
            Set of nodes to perform manipulation on.

        inplace : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        MAG
            A new MAG with outgoing edges from X removed.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("A", "B", "-", ">")
        >>> mag.add_edge("A", "C", "-", ">")
        >>> mag.add_edge("C", "B", "-", ">")
        >>> mag.add_edge("B", "C", ">", ">")
        >>> new_mag = mag.lower_manipulation({"A"})
        >>> list(new_mag.edges(data=True))
        [('B', 'C', {'marks': {'B': '>', 'C': '>'}})]
        """
        if not inplace:
            new_mag = self.copy()
        else:
            new_mag = self

        edges_to_remove = []
        edges_to_change = []

        for u in X:
            neighbors = self.get_neighbors(u, u_type="-", v_type=">")
            for v in neighbors:
                if self.is_visible_edge(u, v):
                    edges_to_remove.append((u, v))
                else:
                    edges_to_change.append((u, v))

        new_mag.remove_edges_from(edges_to_change + edges_to_remove)
        for u, v in edges_to_change:
            other = v if u in X else u
            for neighbor in self.neighbors(v):
                if neighbor != other and neighbor not in X:
                    new_mag.add_edge(other, neighbor, ">", ">")

        return new_mag

    def upper_manipulation(self, X, inplace=False):
        """
        Performs upper manipulation.

        Deletes all edges (directed or bidirected) that have an arrowhead
        pointing to any variable in X. The rest of the graph remains unchanged.

        Parameters
        ----------
        X : set
            Set of nodes to perform manipulation on.

        inplace : bool, optional
            If True, modifies the current graph in place. Defaults to False.

        Returns
        -------
        MAG
            A new MAG with incoming edges to X removed.

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "Y", ">", "-")
        >>> mag.add_edge("Z", "X", ">", "-")
        >>> mag.add_edge("A", "X", "-", ">")
        >>> new_mag = mag.upper_manipulation({"X"})
        >>> new_mag.has_edge("Z", "X")
        True
        >>> new_mag.has_edge("A", "X")
        False
        >>> new_mag.has_edge("X", "Y")
        False
        """
        if not inplace:
            new_mag = self.copy()
        else:
            new_mag = self
        edges_to_remove = []

        for u in X:
            neighbors = self.get_neighbors(u, u_type=">", v_type=None)
            for v in neighbors:
                edges_to_remove.append((u, v))

        new_mag.remove_edges_from(edges_to_remove)
        return new_mag
