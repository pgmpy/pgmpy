from typing import Hashable, Iterable, Optional

import networkx as nx

from pgmpy.base.AncestralBase import AncestralBase


class MAG(AncestralBase):
    """
    Class for representing Maximal Ancestral Graphs (MAGs).
    MAGs are mixed graphs that can contain directed (->), bidirected (<->),
    and undirected (-) edges, and are closed under marginalization and conditioning.
    """

    def __init__(
        self,
        ebunch: Optional[Iterable[tuple[Hashable, Hashable]]] = None,
        latents: set[Hashable] = set(),
    ):
        """
        Initialize a Maximal Ancestral Graph.

        Parameters
        ----------
        ebunch : iterable of tuples, optional
            A list or iterable of edges to add at initialization.
        latents : set, default=set()
            Set of latent (unobserved) variables.

        Returns
        -------
        MAG
            A new instance of a Maximal Ancestral Graph.

        Examples
        --------
        >>> mag = MAG(ebunch=[("A", "B"), ("B", "C")], latents={"L"})
        >>> mag.nodes()
        ['A', 'B', 'C']
        """
        super().__init__(ebunch=ebunch, latents=latents)

    def _is_collider(self, u_node, c_node, v_node):
        """
        Check if a node is a collider in a path u - c - v.

        A collider is a node with incoming arrowheads on both sides:
        u -> c <- v.

        Parameters
        ----------
        u_node : Hashable
            The first endpoint in the triple (u, c, v).
        c_node : Hashable
            The middle node, candidate collider.
        v_node : Hashable
            The second endpoint in the triple.

        Returns
        -------
        bool
            True if `c_node` is a collider on the path, False otherwise.

        Examples
        --------
        >>> from pgmpy.base.MAG import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "Z", "-", ">")  # X -> Z
        >>> mag.add_edge("Y", "Z", "-", ">")  # Y -> Z
        >>> mag._is_collider("X", "Z", "Y")
        True
        """
        mark_uc_at_c = self.edges[u_node, c_node]["marks"][c_node]
        mark_cv_at_c = self.edges[c_node, v_node]["marks"][c_node]

        return mark_uc_at_c == ">" and mark_cv_at_c == ">"

    def has_inducing_path(self, u, v, W):
        """
        Check if there exists an inducing path between two nodes relative to W.

        An inducing path between u and v is a path such that:
        - All intermediate nodes are in W,
        - Each intermediate node is a collider,
        - Each intermediate node is an ancestor of u or v.

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
        >>> from pgmpy.base.MAG import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "L", "-", ">")  # X -> L
        >>> mag.add_edge("Y", "L", "-", ">")  # Y -> L
        >>> mag.latents = {"L"}
        >>> mag.has_inducing_path("X", "Y", mag.latents)
        True
        """
        for path in nx.all_simple_paths(self, source=u, target=v):
            if len(path) <= 2:
                continue

            intermediate_nodes = set(path[1:-1])

            if not intermediate_nodes.issubset(W):
                continue

            ancestors_uv = self.get_ancestors(u).union(self.get_ancestors(v))
            is_inducing = True

            for i in range(1, len(path) - 1):
                prev_node, current_node, next_node = path[i - 1], path[i], path[i + 1]

                if not self._is_collider(prev_node, current_node, next_node):
                    is_inducing = False
                    break

                if current_node not in ancestors_uv:
                    is_inducing = False
                    break

            if is_inducing:
                return True

        return False

    def is_visible_edge(self, u, v) -> bool:
        """
        Check if an edge is visible.

        An edge is visible if it exists and is not shielded by an inducing path
        through latent variables.

        Parameters
        ----------
        u : Hashable
            First node.

        v : Hashable
            Second node.

        Returns
        -------
        bool
            True if the edge is visible, False otherwise.

        Examples
        --------
        >>> from pgmpy.base.MAG import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "Y", "-", ">")  # X -> Y
        >>> mag.is_visible_edge("X", "Y")
        True
        """
        if not self.has_edge(u, v):
            return False
        return not self.has_inducing_path(u, v, self.latents)

    def is_invisible_edge(self, u, v):
        """
        Check if an edge is invisible.

        An edge is invisible if it exists but is shielded by an inducing path
        through latent variables.

        Parameters
        ----------
        u : Hashable
            First node.

        v : Hashable
            Second node.

        Returns
        -------
        bool
            True if the edge is invisible, False otherwise.

        Examples
        --------
        >>> from pgmpy.base.MAG import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "L", "-", ">")
        >>> mag.add_edge("Y", "L", "-", ">")
        >>> mag.latents = {"L"}
        >>> mag.is_invisible_edge("X", "Y")
        True
        """
        if not self.has_edge(u, v):
            return False
        return self.has_inducing_path(u, v, self.latents)

    def lower_manipulation(self, X):
        """
        Perform lower manipulation (marginalization).

        Removes variables in `X` from the MAG while preserving independence
        structure implied by marginalization. Invisible edges are replaced with
        bidirected edges. Directed edges into marginalized nodes are removed.

        Parameters
        ----------
        X : set
            Set of nodes to marginalize (remove).

        Returns
        -------
        MAG
            A new MAG with nodes in X marginalized out.

        Examples
        --------
        >>> from pgmpy.base.MAG import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "L", "-", ">")
        >>> mag.add_edge("Y", "L", "-", ">")
        >>> mag.latents = {"L"}
        >>> new_mag = mag.lower_manipulation({"L"})
        >>> new_mag.edges()
        [('X', 'Y')]
        """
        new_mag = self.copy()

        for u, v in list(self.edges()):
            if u not in X and v not in X:
                continue

            if self.is_invisible_edge(u, v):
                new_mag.add_edge(u, v, ">", ">")

            elif self.is_visible_edge(u, v):
                marks = self.edges[u, v]["marks"]
                if v in X and marks.get(u) == "-" and marks.get(v) == ">":
                    new_mag.remove_edge(u, v)
                elif u in X and marks.get(v) == "-" and marks.get(u) == ">":
                    new_mag.remove_edge(u, v)

        new_mag.remove_nodes_from(X)
        return new_mag

    def upper_manipulation(self, X):
        """
        Perform upper manipulation (conditioning).

        Removes directed edges outgoing from nodes in `X`, representing
        conditioning on those variables.

        Parameters
        ----------
        X : set
            Set of nodes to condition on.

        Returns
        -------
        MAG
            A new MAG with outgoing edges from X removed.

        Examples
        --------
        >>> from pgmpy.base.MAG import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "Y", "-", ">")
        >>> new_mag = mag.upper_manipulation({"X"})
        >>> new_mag.has_edge("X", "Y")
        False
        """
        new_mag = self.copy()
        edges_to_remove = []

        for u, v in self.edges():
            marks = self.edges[u, v]["marks"]
            if u in X and marks.get(u) == "-" and marks.get(v) == ">":
                edges_to_remove.append((u, v))
            elif v in X and marks.get(v) == "-" and marks.get(u) == ">":
                edges_to_remove.append((u, v))

        new_mag.remove_edges_from(edges_to_remove)
        return new_mag
