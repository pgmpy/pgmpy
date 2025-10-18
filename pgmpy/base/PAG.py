from itertools import combinations, product
from typing import Hashable, Iterable, Optional

import networkx as nx

from pgmpy.base import AncestralBase


class PAG(AncestralBase):
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
        - Both incident edges have circle marks at vertex and the two adjacent vertices are not adjacent to each other.

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
        u_mark = self.edges[adj_u, vertex]["marks"].get(vertex)
        v_mark = self.edges[vertex, adj_v]["marks"].get(vertex)
        if u_mark == "-" or v_mark == "-":
            return True
        if u_mark == "o" and v_mark == "o" and not self.has_edge(adj_u, adj_v):
            return True
        return False

    def get_possible_ancestors(self, node):
        """
        Return the set of possible ancestors of a given node.

        A node X is a possible ancestor of Y if X=Y or if there exists a possibly directed path from X to Y.

        Parameters
        ----------
        node : Hashable
            The node whose possible ancestors are being queried.

        Returns
        -------
        set
            Set of possible ancestor nodes including the node itself.
        """
        possible_ancestors = set([node])
        for other in self.nodes:
            if other == node:
                continue
            pd_paths = self.get_potentially_directed(start=other, end=node)
            if pd_paths:
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
        if not self.has_edge(u, v):
            return False
        if self.edges[u, v]["marks"][u] != "-" or self.edges[u, v]["marks"][v] != ">":
            return False

        for neighbor in self.get_neighbors(u, v_type=">"):
            if neighbor not in self.get_neighbors(v):
                return True

        stack = [u]
        visited = set()

        while stack:
            current = stack.pop()

            for pred in self.get_neighbors(current, u_type=None, v_type=">"):
                if pred in visited or pred == u:
                    continue
                visited.add(pred)

                if pred not in self.get_neighbors(v, u_type=None, v_type=None):
                    return True

                if pred in self.get_neighbors(v, u_type="-", v_type=">"):
                    stack.append(pred)

        return False

    def is_uncovered(self, path):
        """
        Check whether a path is uncovered.

        A path is uncovered if no node in the path has a shortcut connection
        to a non-consecutive node (i.e., no edges between nodes at distance 2).

        Parameters
        ----------
        path : list
            Sequence of nodes representing the path.

        Returns
        -------
        bool
            True if the path is uncovered, False otherwise.
        """
        if len(path) < 3:
            return True

        for i in range(1, len(path) - 1):
            x = path[i - 1]
            z = path[i + 1]
            if self.has_edge(x, z):
                return False

        return True

    def get_potentially_directed(self, start, end):
        """
        Find all potentially directed paths between two nodes.

        A potentially directed path is one where no edge has an arrowhead
        pointing backward toward the source.

        Parameters
        ----------
        start : Hashable
            The starting node.

        end : Hashable
            The target node.

        Returns
        -------
        list[list[Hashable]]
            List of paths, each represented as a list of nodes.
        """
        all_pd = []
        all_paths = nx.all_simple_paths(self, source=start, target=end)

        for path in all_paths:
            is_pd = True
            for i in range(len(path) - 1):
                x = path[i]
                y = path[i + 1]
                if self.edges[x, y]["marks"][x] == ">":
                    is_pd = False
                    break

            if is_pd:
                all_pd.append(path)

        return all_pd

    def is_valid_fork_configuration(self, u, w, forks):
        """
        Check whether there exist two forks v, x -> w such that
        there are uncovered potentially directed paths u -> ... -> v
        and u -> ... -> x, with distinct first neighbors mu, omega
        that are not adjacent.

        Parameters
        ----------
        u : Hashable
            Source node with edge u o-> w.

        w : Hashable
            Target node of edge u o-> w.

        forks : list
            List of nodes v such that v -> w.

        Returns
        -------
        bool
            True if the fork configuration satisfies R10 condition,
            False otherwise.
        """
        for (
            v,
            x,
        ) in combinations(forks, 2):
            pd_uv = [
                p for p in self.get_potentially_directed(u, v) if self.is_uncovered(p)
            ]
            if not pd_uv:
                continue

            pd_ux = [
                p for p in self.get_potentially_directed(u, x) if self.is_uncovered(p)
            ]
            if not pd_ux:
                continue

            for p1 in pd_uv:
                mu = p1[1] if len(p1) > 1 else v
                for p2 in pd_ux:
                    omega = p2[1] if len(p2) > 1 else x
                    if mu != omega and not self.has_edge(mu, omega):
                        return True
        return False

    def get_paths_with_marks(self, u, v, u_type=None, v_type=None):
        """
        Find all simple paths between u and v where each traversed edge
        satisfies the specified mark constraints.

        Parameters
        ----------
        u, v : Hashable
            Start and end nodes (must be distinct).

        u_type : str or None, default=None
            Required mark at the current node for each traversed edge.
            If None, allow any mark.

        v_type : str or None, default=None
            Required mark at the neighbor node for each traversed edge.
            If None, allow any mark.

        Returns
        -------
        list[list[Hashable]]
            All valid paths from u to v.

        Raises
        ------
        ValueError
            If u == v.
        """
        if u == v:
            raise ValueError("Start and end nodes must differ (path length >= 2).")

        valid_paths = []
        for path in nx.all_simple_paths(self, source=u, target=v):
            ok = True
            for a, b in zip(path, path[1:]):
                mark_a = self.edges[a, b]["marks"].get(a)
                mark_b = self.edges[a, b]["marks"].get(b)
                if u_type is not None and mark_a != u_type:
                    ok = False
                    break
                if v_type is not None and mark_b != v_type:
                    ok = False
                    break
            if ok:
                valid_paths.append(path)

        return valid_paths

    def modify_edge(self, u, v, mark_u=None, mark_v=None):
        """
        Modify the marks on an existing edge between two nodes.

        Parameters
        ----------
        u : Hashable
            First node.

        v : Hashable
            Second node.

        mark_u : str
            New mark at node u (one of '-', '>', 'o').

        mark_v : str
            New mark at node v (one of '-', '>', 'o').

        Raises
        ------
        ValueError
            If there is no edge between u and v.
        """
        if not self.has_edge(u, v):
            raise ValueError(f"No edge between {u} and {v}")
        if mark_u is not None:
            self.edges[u, v]["marks"][u] = mark_u
        if mark_v is not None:
            self.edges[u, v]["marks"][v] = mark_v

    def has_discriminating_path(self, x, y, v):
        # node is a non end point and adjacent to y
        # x and y are not adjacent

        if x == y:
            raise ValueError("`x` and `y` cannot be the same nodes.")
        if self.get_edge_marks(y, v) != {y: ">", v: ">"}:
            return False

        discriminating_paths = []

        for path in nx.all_simple_edge_paths(self, x, v):
            if self.has_edge(x, y):
                continue

            for x1, x2 in path[1:]:

                # should be a collider
                if self.get_edge_marks(x1, x2) != {x1: ">", x2: ">"}:
                    continue

                # must be parent of y
                if self.get_edge_marks(x1, y) != {x1: "-", y: ">"}:
                    continue

            discriminating_paths.append(path)

        return discriminating_paths if discriminating_paths else False

    def rule_1(self, inplace=False, **kwargs):
        """
        If we have a triple u *-> v o-* w such that:
        - u and w are non-adjacent,
        then orient them as u *-> v -> w.

        Parameters
        ----------
        inplace : bool, default=False
            If True, modifies the graph in place.
            If False, works on and returns a copy.

        Returns
        -------
        PAG or None
            A new graph with orientations applied if inplace=False,
            otherwise None.
        """
        pag = self if inplace else self.copy()

        for node in pag.nodes:
            u_candidates = pag.get_neighbors(node, u_type=">", v_type=None)

            w_candidates = pag.get_neighbors(node, u_type="o", v_type=None)

            for u, w in product(u_candidates, w_candidates):
                if not pag.has_edge(u, w):
                    pag.modify_edge(node, w, mark_u="-", mark_v=">")

        if not inplace:
            return pag

    def rule_2(self, inplace=False, **kwargs):
        """
        If u -> v *-> w or u *-> v -> w and u *-o w, then orient u *-> w .

        Parameters
        ----------
        inplace : bool, default=False
            If True, modifies the graph in place.
            If False, works on and returns a copy.

        Returns
        -------
        PAG or None
            A new graph with    ations applied if inplace=False,
            otherwise None.
        """
        pag = self if inplace else self.copy()

        for node in pag.nodes:
            u_candidates = pag.get_neighbors(node, u_type=">", v_type="-")
            w_candidates = pag.get_neighbors(node, u_type=None, v_type=">")

            for u, w in product(u_candidates, w_candidates):
                if self.has_edge(u, w):
                    if self.get_edge_marks(u, w)[w] == "o":
                        pag.modify_edge(u, w, mark_u=None, mark_v=">")

        for node in pag.nodes:
            u_candidates = pag.get_neighbors(node, u_type=">", v_type=None)
            w_candidates = pag.get_neighbors(node, u_type="-", v_type=">")

            for u, w in product(u_candidates, w_candidates):
                if self.has_edge(u, w):
                    if self.get_edge_marks(u, w)[w] == "o":
                        pag.modify_edge(u, w, mark_u=None, mark_v=">")

        if not inplace:
            return pag

    def rule_3(self, inplace=False, **kwargs):
        """
        If u *-> v <-* w,
        u *-o z o-* w,
        u and w are not adjacent,
        and z *-o v,
        then orient the edge z *-o v as z*-> v.

        Parameters
        ----------
        inplace : bool, default=False
            If True, modifies the graph in place.
            If False, works on and returns a copy.

        Returns
        -------
        PAG or None
            A new graph with orientations applied if inplace=False,
            otherwise None.
        """

        pag = self if inplace else self.copy()

        for v in pag.nodes:
            potential_uw_cond1 = pag.get_neighbors(v, u_type=">", v_type=None)

            if len(potential_uw_cond1) < 2:
                break

            potential_z = pag.get_neighbors(v, u_type="o", v_type=None)
            for z in potential_z:
                potential_uw_cond2 = pag.get_neighbors(z, u_type="o", v_type=None)

                common_uw = potential_uw_cond2.intersection(potential_uw_cond1)
                for u, w in combinations(common_uw, 2):
                    if not pag.has_edge(u, w):
                        pag.modify_edge(z, v, mark_u=None, mark_v=">")

        if not inplace:
            return pag

    def rule_4(self, inplace=False, **kwargs):
        """
        If u = (θ, . . . , α , β, γ) is a discriminating path between θ and γ
        for β , and β◦−−∗ γ ; then if β ∈ Sepset(θ, γ ), orient
        β◦−−∗ γ as β → γ ; otherwise orient the triple (α , β, γ ) as α ↔ β ↔ γ .
        """
        pag = self if inplace else self.copy()
        if "separating_sets" not in kwargs:
            raise ValueError("Separating Sets is not passed")

        if not inplace:
            return pag

    def rule_5(self, inplace=False, **kwargs):
        """
        R5: Uncovered circle path between two nodes (u, v) connected by an o-o edge.
        """
        pag = self if inplace else self.copy()

        # TO DO the implementation of R5

        if not inplace:
            return pag

    def rule_6(self, inplace=False, **kwargs):
        """
        If u -- v o–-* w and u and w may or may not be adjacent, then orient v o--* w as v -–* w.

        Parameters
        ----------
        inplace : bool, default=False
            If True, modifies the graph in place.
            If False, works on and returns a copy.

        Returns
        -------
        PAG or None
            A new graph with orientations applied if inplace=False,
            otherwise None.
        """
        pag = self if inplace else self.copy()

        for v in pag.nodes:
            u_candidates = pag.get_neighbors(v, u_type="-", v_type=">")

            w_candidates = pag.get_neighbors(v, u_type="o", v_type=None)

            for _, w in product(u_candidates, w_candidates):
                pag.modify_edge(v, w, mark_u=">", mark_v=None)

        if not inplace:
            return pag

    def rule_7(self, inplace=False, **kwargs):
        """
        If u --o v o--* w, u and w are non-adjacent, then orient v o--* w as v -–* w.

        Parameters
        ----------
        inplace : bool, default=False
            If True, modifies the graph in place.
            If False, works on and returns a copy.

        Returns
        -------
        PAG or None
            A new graph with orientations applied if inplace=False,
            otherwise None.
        """
        pag = self if inplace else self.copy()

        for v in pag.nodes:
            u_candidates = pag.get_neighbors(v, u_type="-", v_type="-")
            w_candidates = pag.get_neighbors(v, u_type="o", v_type=None)
            for u, w in product(u_candidates, w_candidates):
                if u == w or pag.has_edge(u, w):
                    continue

                self.modify_edge(v, w, mark_u="-", mark_v=None)

        if not inplace:
            return pag

    def rule_8(self, inplace=False, **kwargs):
        """
        If u o-> v and there exists a directed path v -> … -> w such that
        u and w are adjacent by u o– w, then orient u -> w.

        Parameters
        ----------
        inplace : bool, default=False
            If True, modifies the graph in place.
            If False, works on and returns a copy.

        Returns
        -------
        PAG or None
            A new graph with orientations applied if inplace=False,
            otherwise None.
        """
        pag = self if inplace else self.copy()
        for v in pag.nodes:
            u_candidates = (pag.get_neighbors(v, u_type=">", v_type="-")) or (
                pag.get_neighbors(v, u_type="o", v_type="-")
            )

            w_candidates = pag.get_neighbors(v, u_type="-", v_type=">")

            for u, w in product(u_candidates, w_candidates):
                if not pag.has_edge(u, w):
                    continue

                if (
                    self.get_edge_marks(u, w)[u] == "o"
                    and self.get_edge_marks(u, v)[w] == ">"
                ):
                    pag.modify_edge(u, w, mark_u="-", mark_v=">")

        if not inplace:
            return pag

    def rule_9(self, inplace=False, **kwargs):
        """
        If u o-> w and there exists an uncovered potentially directed path
        ⟨u, v, …, w⟩ with w and v non-adjacent, then orient u -> w.

        Parameters
        ----------
        inplace : bool, default=False
            If True, modifies the graph in place.
            If False, works on and returns a copy.

        Returns
        -------
        PAG or None
            A new graph with orientations applied if inplace=False,
            otherwise None.
        """
        pag = self if inplace else self.copy()

        for u, w in list(pag.edges):
            if not (
                pag.has_edge(u, w)
                and pag.edges[u, w]["marks"].get(u) == "o"
                and pag.edges[u, w]["marks"].get(w) == ">"
            ):
                continue

            pd_paths = pag.get_potentially_directed(start=u, end=w)

            for path in pd_paths:
                if not pag.is_uncovered(path=path):
                    continue

                v = path[1]

                if not pag.has_edge(v, w):
                    pag.modify_edge(u, w, mark_u="-", mark_v=">")
                    break

        if not inplace:
            return pag

    def rule_10(self, inplace=False, **kwargs):
        """
        If u o-> w, and there exist two nodes v -> w <- x such that:
        - there is an uncovered potentially directed path from u to v,
        - there is an uncovered potentially directed path from u to x,
        - the first neighbors μ and ω on those paths (after u) are distinct,
        - and μ, ω are not adjacent,
        then orient u -> w.

        Parameters
        ----------
        inplace : bool, default=False
            If True, modifies the graph in place.
            If False, works on and returns a copy.

        Returns
        -------
        PAG or None
            A new graph with orientations applied if inplace=False,
            otherwise None.
        """
        pag = self if inplace else self.copy()

        for u, w in list(pag.edges):
            if not (
                pag.has_edge(u, w)
                and self.get_edge_marks(u, w)[u] == "o"
                and self.get_edge_marks(u, w)[w] == ">"
            ):
                continue

            forks = pag.get_neighbors(w, u_type=">", v_type="-")

            if len(forks) < 2:
                continue

            if pag.is_valid_fork_configuration(u=u, w=w, forks=forks):
                pag.modify_edge(u, w, mark_u="-", mark_v=">")

        if not inplace:
            return pag

    def apply_orientation_rules(self, rules=None, inplace=False, separating_sets=None):
        """
        Apply all orientation rules (R1 to R10) until no more changes occur.

        The rules are applied repeatedly in sequence, propagating orientations
        until the graph stabilizes.

        Parameters
        ----------
        inplace : bool, default=False
            If True, modifies the graph in place.
            If False, returns a new graph.

        Returns
        -------
        PAG
            The graph with orientation rules applied.
        """
        pag = self if inplace else self.copy()

        rules_map = {
            "R1": pag.rule_1,
            "R2": pag.rule_2,
            "R3": pag.rule_3,
            # R4 ; TO DO
            "R5": pag.rule_5,
            "R6": pag.rule_6,
            "R7": pag.rule_7,
            "R8": pag.rule_8,
            "R9": pag.rule_9,
            "R10": pag.rule_10,
        }

        rules_to_apply = rules or list(rules_map.keys())

        missing = set(rules_to_apply) - set(rules)
        if missing:
            raise ValueError(f"Unknown Rule(s) Requested:  {missing}")

        for r in rules_to_apply:
            func = rules_map[r]
            if inplace:
                func(pag, separating_sets=separating_sets, inplace=inplace)
            else:
                pag = func(pag, separating_sets=separating_sets, inplace=inplace)

        return pag
