from collections import deque
from typing import Hashable, Iterable, Optional

from pgmpy.base import MAG, AncestralBase


class PAG(AncestralBase, MAG):
    """
    Partial Ancestral Graph (PAG).

    A PAG represents an equivalence class of MAGs. It allows for circle endpoints ('o')
    to represent uncertainty about whether an endpoint is an arrow ('>') or a tail ('-').
    """

    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
        roles=None,
    ):
        """Initialize Partial Ancestral Graphs.

        Unlike MAGs, PAGs allow circle marks ('o') at edge endpoints to represent
        uncertainty about the true edge mark in the underlying MAG.

        Parameters
        ----------
        ebunch : Iterable[tuple], optional
            An iterable of edges of the form (u, v, u_mark, v_mark) used to
            initialize the graph. Each mark must be one of {">", "-", "o"}.
            Default is None, which initializes an empty graph.

        latents : set, optional
            Set of latent (unobserved) variables in the graph. Default is
            an empty set.

        roles : dict, optional (default: None)
            The keys are roles, and the values are role names (strings or iterables of str).
            If provided, this will automatically assign roles to the nodes in the graph.
            Passing a key-value pair via ``roles`` is equivalent to calling
            ``with_role(role, variables)`` for each key-value pair in the dictionary.

        Returns
        -------
        PAG
            A new instance of a Partial Ancestral Graph.

        Examples
        --------
        >>> from pgmpy.base import AncestralBase
        [('A', 'B', {'marks': {'A': '-', 'B': '>'}}),
         ('B', 'C', {'marks': {'B': '>', 'C': '-'}}),
         ('C', 'D', {'marks': {'C': 'o', 'D': 'o'}})]

        Roles can be assigned to nodes in the graph at construction or using methods.

        At construction:
        >>> g = AncestralBase(
        ...     ebunch=[("L", "A", "-", ">"), ("B", "C", "-", ">")],
        ...     latents={"L"},
        ...     roles={"exposure": "A", "outcome": "B"},
        ... )

        Roles can also be assigned after creation using ``with_role`` method.

        >>> g = g.with_role("adjustment", {"L", "C"})

        Vertices of a specific role can be retrieved using ``get_role`` method.

        >>> g.get_role("exposure")
        ["A"]
        >>> g.get_role("adjustment")
        ["L", "C"]
        """

        super().__init__(ebunch=ebunch, latents=latents, roles=roles)

    def is_definite_non_collider(self, vertex, adj_u, adj_v):
        """
        Determine if a vertex on a path is a definite non-collider.

        A vertex is a definite non-collider if:
        - Either incident edge has a tail at the vertex, or
        - Both incident edges have circle marks at vertex and the two adjacent vertices are not adjacent.

        Parameters
        ----------
        vertex : Hashable
            The vertex to check on the path.

        adj_u : Hashable
            The node preceding `vertex` on the path.

        adj_v : Hashable
            The node following `vertex` on the path.

        Returns
        -------
        bool
            True if the vertex is a definite non-collider, False otherwise.
        """
        u_mark, _ = (
            self.edges[adj_u, vertex]["marks"][adj_u],
            self.edges[adj_u, vertex]["marks"][vertex],
        )
        _, v_mark = (
            self.edges[vertex, adj_v]["marks"][vertex],
            self.edges[vertex, adj_v]["marks"][adj_v],
        )
        if u_mark == "-" or v_mark == "-":
            return True
        if u_mark == "o" and v_mark == "o" and not self.has_edge(adj_u, adj_v):
            return True
        return False

    def has_possibly_directed_path(self, u, v):
        """
        Check if there exists a possibly directed path from node u to node v.

        A path is possibly directed if no edge points into the preceding node along the path.

        Parameters
        ----------
        u : Hashable
            The starting node of the path.

        v : Hashable
            The target node of the path.

        Returns
        -------
        bool
            True if a possibly directed path exists from u to v, False otherwise.
        """
        visited = set()
        queue = deque([u])
        while queue:
            current = queue.popleft()
            if current == v:
                return True
            visited.add(current)
            for neighbor in self.get_neighbors(current):
                mark = self.edges[current, neighbor]["marks"][current]
                if mark != ">" and neighbor not in visited:
                    queue.append(neighbor)
        return False

    def get_possible_ancestors(self, node):
        """
        Return the set of possible ancestors of a given node.

        A node X is a possible ancestor of Y if there exists a possibly directed path from X to Y.

        Parameters
        ----------
        node : Hashable
            The node whose possible ancestors are being queried.

        Returns
        -------
        set
            Set of possible ancestor nodes including the node itself.
        """
        possible_ancestors = set()
        for other in self.nodes:
            if self.has_possibly_directed_path(other, node):
                possible_ancestors.add(other)
        return possible_ancestors

    def is_definitely_visible(self, u, v):
        """
        Determine if an edge u -> v is definitely visible in the PAG.

        An edge is definitely visible if it satisfies the visibility conditions
        for all MAGs represented by the PAG.

        Parameters
        ----------
        u : Hashable
            The source node of the edge.

        v : Hashable
            The target node of the edge.

        Returns
        -------
        bool
            True if the edge is definitely visible, False otherwise.
        """
        if self.edges[u, v]["marks"][u] != "-" or self.edges[u, v]["marks"][v] != ">":
            return False
        for parent in self.get_parents(u):
            if parent not in self.get_neighbors(v):
                return True
        for spouse in self.get_spouses(u):
            if spouse in self.get_parents(v):
                return True
        return False

    def is_definite_m_connecting_path(self, path, Z):
        """
        Check if a path is a definite m-connecting path relative to a conditioning set Z.

        A path is definite m-connecting if:
        - Every non-endpoint vertex is either a definite non-collider or a collider.
        - Every definite non-collider is not in Z.
        - Every collider is a possible ancestor of some node in Z.

        Parameters
        ----------
        path : list
            List of nodes representing the path.

        Z : set
            Conditioning set of nodes.

        Returns
        -------
        bool
            True if the path is a definite m-connecting path relative to Z, False otherwise.
        """
        for i in range(1, len(path) - 1):
            u, v, w = path[i - 1], path[i], path[i + 1]
            if self.is_definite_non_collider(path, v, u, w):
                if v in Z:
                    return False
            else:
                if not (self.get_possible_ancestors(v) & Z):
                    return False
        return True

    def is_possibly_m_connecting_path(self, path, Z):
        """
        Check if a path is a possibly m-connecting path relative to a conditioning set Z.

        Parameters
        ----------
        path : list
            List of nodes representing the path.

        Z : set
            Conditioning set of nodes.

        Returns
        -------
        bool
            True if the path is possibly m-connecting relative to Z, False otherwise.
        """
        for i in range(1, len(path) - 1):
            u, v, w = path[i - 1], path[i], path[i + 1]
            if self.is_definite_non_collider(path, v, u, w):
                if v in Z:
                    return False
            else:
                if not (self.get_possible_ancestors(v) & Z):
                    return False
        return True

    def is_definitely_m_separated(self, X, Y, Z):
        """
        Determine if sets X and Y are definitely m-separated by a conditioning set Z in the PAG.

        X and Y are definitely m-separated if no possibly m-connecting path exists between them given Z.

        Parameters
        ----------
        X : set
            Set of nodes representing the first set.

        Y : set
            Set of nodes representing the second set.

        Z : set
            Conditioning set of nodes.

        Returns
        -------
        bool
            True if X and Y are definitely m-separated by Z, False otherwise.
        """
        X = set(X)
        Y = set(Y)
        Z = set(Z)

        def explore(node, coming_from=None, visited=None):
            if visited is None:
                visited = set()
            visited.add(node)
            for neighbor in self.neighbors(node):
                if neighbor in visited:
                    continue
                if coming_from:
                    if self.is_definite_non_collider(
                        [coming_from, node, neighbor], node, coming_from, neighbor
                    ):
                        if node in Z:
                            continue
                    else:
                        if not (self.get_possible_ancestors(node) & Z):
                            continue
                yield neighbor
                yield from explore(neighbor, coming_from=node, visited=visited.copy())

        for x in X:
            reachable = set(explore(x))
            if reachable & Y:
                return False
        return True

    def YX_manipulation(self, Y, X, inplace=False):
        """
        Apply composite YX-manipulation: lower-priority lower manipulation
        followed by lower-priority upper manipulation.

        Parameters
        ----------
        Y : set
            Set of nodes for lower-priority manipulation.

        X : set
            Set of nodes for upper-priority manipulation.

        inplace : bool
            Manipulates the graph in place if True, False otherwise.

        Returns
        -------
        None
            Manipulates the graph in-place.
        """
        if not inplace:
            new_pag = self.copy()
        else:
            new_pag = self
        new_pag.lower_manipulation(Y)
        new_pag.upper_manipulation(X)
