from collections.abc import Hashable

import networkx as nx

from pgmpy.base._base import _CoreGraph


class PAG(_CoreGraph):
    """Representation of a Partial Ancestral Graph (PAG).

    A PAG represents a Markov equivalence class of maximal ancestral graphs.
    Besides tails (``"-"``) and arrowheads (``">"``), it permits circle
    endpoint marks (``"o"``). A circle indicates that the mark at that
    endpoint has not yet been determined.

    The edge representation follows :class:`~pgmpy.base._base._CoreGraph`:
    ``"o>"`` represents ``u o-> v``, ``"<o"`` represents ``u <-o v``,
    and ``"oo"`` represents ``u o-o v``.

    Parameters
    ----------
    edge_list : iterable of tuple, optional
        Edges as ``(u, v, edge_type)``. Valid types are ``"--"``, ``"-o"``,
        ``"o-"``, ``"->"``, ``"<-"``, ``"o>"``, ``"<o"``, ``"<>"``,
        and ``"oo"``.
    exposures, outcomes, latents : set, optional
        Node roles used by causal-analysis algorithms.
    roles : dict, optional
        Additional node roles.

    References
    ----------
    - :footcite:t:`zhang_2008`
    """

    SUPPORTED_EDGE_TYPES = _CoreGraph.SUPPORTED_EDGE_TYPES

    def add_edge(self, u: Hashable, v: Hashable, edge_type: str) -> None:
        """Add a PAG edge, rejecting parallel edges between the same nodes.

        A PAG assigns one pair of endpoint marks to each adjacent node pair;
        parallel edges would not have an unambiguous PAG interpretation.
        """
        if self.has_edge(u, v):
            raise ValueError(f"A PAG already has an edge between {u!r} and {v!r}.")
        super().add_edge(u, v, edge_type)

    def get_edge_marks(self, u: Hashable, v: Hashable) -> dict[Hashable, str]:
        """Return the endpoint marks for the unique edge between ``u`` and ``v``.

        Raises
        ------
        ValueError
            If no edge exists or the nodes are connected by parallel edges.
        """
        if not self.has_edge(u, v):
            raise ValueError(f"No edge exists between {u!r} and {v!r}.")

        edge_data = self.get_edge_data(u, v)
        if edge_data is None or len(edge_data) != 1:
            raise ValueError(
                f"Expected one edge between {u!r} and {v!r}; use get_edge_type "
                "for parallel edges."
            )

        return next(iter(edge_data.values())).copy()

    def modify_edge(
        self,
        u: Hashable,
        v: Hashable,
        mark_u: str | None = None,
        mark_v: str | None = None,
    ) -> None:
        """Update one or both endpoint marks of an existing PAG edge.

        ``mark_u`` and ``mark_v`` must each be a tail (``"-"``), arrowhead
        (``">"``), or circle (``"o"``). A value of ``None`` preserves that
        endpoint's current mark.
        """
        if mark_u is not None:
            self.set_marker(v, u, mark_u)
        if mark_v is not None:
            self.set_marker(u, v, mark_v)

    def is_definite_non_collider(
        self, previous: Hashable, node: Hashable, next_node: Hashable
    ) -> bool:
        """Return whether ``node`` is a definite non-collider on a path.

        A node is a definite non-collider when one incident edge has a tail at
        the node, or when both incident endpoints are circles and the adjacent
        path nodes are not adjacent.
        """
        previous_marks = self.get_edge_marks(previous, node)
        next_marks = self.get_edge_marks(node, next_node)

        if previous_marks[node] == "-" or next_marks[node] == "-":
            return True

        return (
            previous_marks[node] == "o"
            and next_marks[node] == "o"
            and not self.has_edge(previous, next_node)
        )

    def is_uncovered(self, path: list[Hashable]) -> bool:
        """Return whether every consecutive triple in ``path`` is unshielded."""
        if len(path) < 2:
            raise ValueError("A path must contain at least two nodes.")

        if any(not self.has_edge(left, right) for left, right in zip(path, path[1:])):
            raise ValueError("Every consecutive pair in path must be adjacent.")

        return all(
            not self.has_edge(path[index - 1], path[index + 1])
            for index in range(1, len(path) - 1)
        )

    def get_potentially_directed_paths(
        self, start: Hashable, end: Hashable
    ) -> list[list[Hashable]]:
        """Return simple paths that can be directed from ``start`` to ``end``.

        A path is potentially directed when no edge has an arrowhead into its
        preceding node or a tail at its following node.
        """
        if start == end:
            raise ValueError("Start and end nodes must differ.")
        if start not in self or end not in self:
            raise ValueError("Start and end nodes must both be present in the graph.")

        potentially_directed = []
        for path in nx.all_simple_paths(self, source=start, target=end):
            if all(
                self.get_edge_marks(left, right)[left] != ">"
                and self.get_edge_marks(left, right)[right] != "-"
                for left, right in zip(path, path[1:])
            ):
                potentially_directed.append(path)
        return potentially_directed

    def get_possible_ancestors(self, node: Hashable) -> set[Hashable]:
        """Return nodes that have a potentially directed path into ``node``."""
        if node not in self:
            raise ValueError(f"Node {node!r} is not in the graph.")

        return {
            candidate
            for candidate in self.nodes
            if candidate == node or self.get_potentially_directed_paths(candidate, node)
        }
