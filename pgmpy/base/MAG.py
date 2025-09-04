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
        super().__init__(ebunch=ebunch, latents=latents)

        if roles is None:
            roles = {}
        elif not isinstance(roles, dict):
            raise TypeError("Roles must be provided as dictionary")

        for role, vars in roles.items():
            self.with_role(role=role, variables=vars, inplace=True)

    def _is_collider(self, u, c, v):
        """
        Check if a node is a collider in a path u - c - v.

        A collider is a node with incoming arrowheads on both sides:
        u -> c <- v.

        Parameters
        ----------
        u : Hashable
            The first endpoint in the triple (u, c, v).
        c : Hashable
            The middle node, candidate collider.
        v : Hashable
            The second endpoint in the triple.

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
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "Y", "-", ">")
        >>> mag.is_visible_edge("X", "Y")
        True
        """
        if not self.has_edge(u, v):
            return False

        graph_without_edge = self.copy()
        graph_without_edge.remove_edge(u, v)
        return not graph_without_edge.has_inducing_path(u, v, self.latents)

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
        >>> from pgmpy.base import MAG
        >>> mag = MAG()
        >>> mag.add_edge("X", "L", "-", ">")
        >>> mag.add_edge("Y", "L", "-", ">")
        >>> mag.latents = {"L"}
        >>> new_mag = mag.lower_manipulation({"L"})
        >>> list(new_mag.edges())
        [('X', 'Y')]
        """
        new_mag = self.copy()

        for node in X:
            neighbors = list(self.neighbors(node))

            for i in range(len(neighbors)):
                for j in range(i + 1, len(neighbors)):
                    u, v = neighbors[i], neighbors[j]

                    if self._is_collider(u, node, v):
                        new_mag.add_edge(u, v, ">", ">")

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
        >>> from pgmpy.base import MAG
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

    def __eq__(self, other):
        """
        Checks if two MAGs are equal. Two MAGs are equal if they have the same
        nodes, edges(including marks), latent variables, and variable roles

        Parameters
        ----------
        other: MAG object
            The other MAG to compare with

        Returns
        -------
        bool
            True if the MAGs are equal, False otherwise

        Examples
        --------
        >>> from pgmpy.base import MAG
        >>> mag1 = MAG(
        ...     ebunch=[("X", "Y", "-", ">"), ("Y", "Z", "-", ">")],
        ...     latents={"L"},
        ...     roles={"exposure": "X"},
        ... )
        >>> mag2 = MAG(
        ...     ebunch=[("X", "Y", "-", ">"), ("Y", "Z", "-", ">")],
        ...     latents={"L"},
        ...     roles={"exposure": "X"},
        ... )
        >>> mag1 == mag2
        True

        >>> mag3 = MAG(
        ...     ebunch=[("X", "Y", "-", ">")], latents={"L"}, roles={"exposure": "X"}
        ... )
        >>> mag1 == mag3
        False
        """
        if not isinstance(other, MAG):
            return False

        self_edges = {
            (u, v, frozenset(data["marks"].items()))
            for u, v, data in self.edges(data=True)
        }
        other_edges = {
            (u, v, frozenset(data["marks"].items()))
            for u, v, data in other.edges(data=True)
        }

        return (
            set(self.nodes()) == set(other.nodes())
            and self_edges == other_edges
            and self.latents == other.latents
            and self.get_role_dict() == other.get_role_dict()
        )
