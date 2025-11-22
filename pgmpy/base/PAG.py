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
        r"""
        Check whether a path is uncovered.

        a path :math:`p = (V_0 , \cdots , V_n)` is said to be uncovered if
        for every :math:`1 \le i \le n-1, V_{i-1} `and` V_{i+1}` are not adjacent, i.e.,
        if every consecutive triple on the path is unshielded.

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

    def get_potentially_directed_paths(self, start, end):
        r"""
        Return all potentially directed paths between two nodes.

        A path :math: `p = (V_0 , \cdots , V_n)` is said to be potentially directed (abbreviated as p.d.)
        from :math:`V_0 \text{to} V_n \text{if for every} 0 \le i \le n-1, \text{the edge between}
        V_{i} \text{and} V_{i+1} \text{is not into} V_{i} \text{or out of} V_{i+1}`.

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
                edge_marks = self.get_edge_marks(x, y)
                if not edge_marks[x] == ">" or not edge_marks[y] == "-":
                    is_pd = False
                    break

            if is_pd:
                all_pd.append(path)

        return all_pd

    def is_valid_fork_configuration(self, u, w, forks):
        """
        Check whether a pair of forks supports the R10 orientation condition.

        This verifies that there exist nodes `v` and `x` such that both point into `w`
        (i.e., `v --> w` and `x --> w`), and there are uncovered, potentially directed
        paths from `u` to each of them. The first neighbors after `u` on these paths
        (`mu` and `omega`) must be different and must not be adjacent.

        Parameters
        ----------
        u : Hashable
            The source node with an edge `u o--> w`.

        w : Hashable
            The endpoint of the edge `u o--> w`.

        forks : list
            Nodes that have edges pointing into `w` (i.e., all `v` such that `v --> w`).

        Returns
        -------
        bool
            True if the configuration satisfies the R10 condition, otherwise False.
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
        Find all simple paths between two nodes that satisfy given edge-mark constraints.

        A valid path is one where every traversed edge matches the required mark
        on the current node (`u_type`) and on the neighbor (`v_type`). If either
        constraint is None, that side of the edge is unrestricted.

        Parameters
        ----------
        u, v : Hashable
            The start and end nodes. They must be different.

        u_type : str or None, default None
            The required mark on the current node for each step along the path.
            If None, any mark is allowed.

        v_type : str or None, default None
            The required mark on the neighboring node for each step.
            If None, any mark is allowed.

        Returns
        -------
        list[list[Hashable]]
            All simple paths from `u` to `v` that satisfy the mark conditions.

        Raises
        ------
        ValueError
            If `u` and `v` are the same node.
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

        This updates the marks on the edge `u--v`. Any mark set to None is left
        unchanged, so only the marks you explicitly provide will be updated.

        Parameters
        ----------
        u : Hashable
            One endpoint of the edge.

        v : Hashable
            The other endpoint of the edge.

        mark_u : str, default None
            The new mark at node `u` (allowed values: '-', '>', 'o').
            If None, the existing mark at `u` is preserved.

        mark_v : str, default None
            The new mark at node `v` (allowed values: '-', '>', 'o').
            If None, the existing mark at `v` is preserved.

        Raises
        ------
        ValueError
            If no edge exists between `u` and `v`.
        """
        if not self.has_edge(u, v):
            raise ValueError(f"No edge between {u} and {v}")
        if mark_u is not None:
            self.edges[u, v]["marks"][u] = mark_u
        if mark_v is not None:
            self.edges[u, v]["marks"][v] = mark_v

    def get_discriminating_path(self, x, y, v):
        r"""
        Check whether there exists a discriminating path for node `v` between `x` and `y`.

        A path \( p = (X, \ldots, W, V, Y) \) in a MAG is a *discriminating path*
        for `V` if it meets the following conditions:

        - The path has at least three edges.
        - `V` is an internal (non-endpoint) node on the path and is adjacent to `Y`
        along that path.
        - `X` is not adjacent to `Y`.
        - Every node between `X` and `V` is a collider on the path and is also
        a parent of `Y`.

        The function returns all such discriminating paths, if any exist.

        Returns
        -------
        list[list[Hashable]]
            All discriminating paths for `v` between `x` and `y`.
            Returns an empty list if none exist.
        """

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

        return discriminating_paths

    def rule_1(self, inplace=False, **kwargs):
        """
        Orient a triple of nodes when the middle node forms a specific mixed pattern.

        If the graph contains a configuration of the form
        `u *--> v o--* w`
        and `u` and `w` are not adjacent, then the edges are oriented as
        `u *--> v --> w`.

        Parameters
        ----------
        inplace : bool, default False
            If True, apply the orientation directly to the graph.
            If False, return a modified copy and leave the original graph unchanged.

        Returns
        -------
        PAG or None
            The updated graph if `inplace=False`. Returns None when applying
            changes in place.
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
        If u --> v *--> w or u *--> v --> w and u *--o w, then orient u *--> w .

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
        Orient the edge `z *--o v` when it is supported by the surrounding structure.

        This rule applies when the following configuration is present:

        - `u *--> v <--* w`
        - `u *--o z o--* w`
        - `u` and `w` are not adjacent
        - `z *--o v` is an existing edge

        When all these conditions hold, the edge `z *--o v` is oriented as `z *-> v`.

        Parameters
        ----------
        inplace : bool, default False
            If True, apply the orientation directly to the current graph.
            If False, operate on and return a modified copy.

        Returns
        -------
        PAG or None
            The updated graph if `inplace=False`. Returns None when changes
            are applied in place.
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
        Orient edges using discriminating paths.

        This rule examines each node `v` and looks for nodes `y` connected to `v`
        with an `o` mark at `v`. For each such pair `(v, y)`, it checks every
        other node `x` to determine whether a discriminating path exists for
        the triple `(x, y, v)`.

        For each discriminating path found:

        - If `v` is in the separating set for the pair `(x, y)`,
        the edge between `v` and `y` is oriented as `v > y`.

        - If `v` is *not* in the separating set for that pair,
        the edge connecting the predecessor of `v` on the path (i.e., `path[-3]`)
        is oriented as `path[-3] --> v`, and the edge `v`--`y` is oriented as `v --> y`.

        A dictionary of separating sets must be provided via the `separating_sets`
        keyword argument. Keys should be `(x, y)` tuples, and values should be
        sets of conditioning nodes.

        Parameters
        ----------
        inplace : bool, default False
            If True, update the current graph directly.
            If False, return a modified copy and leave the original graph unchanged.

        **kwargs
            separating_sets : dict
                A mapping from `(x, y)` node pairs to the set of nodes that
                separate them. This argument is required.

        Returns
        -------
        PAG or None
            The updated graph when `inplace=False`. Returns None when changes
            are applied in place.

        Raises
        ------
        ValueError
            If `separating_sets` is not provided.

        """
        pag = self if inplace else self.copy()
        if "separating_sets" not in kwargs:
            raise ValueError("Separating Sets is not passed")

        for v in pag.nodes:
            potential_y = pag.get_neighbors(v, u_type="o", v_type=None)
            for y in potential_y:
                for x in pag.nodes:

                    if x == v or x == y:
                        continue
                    discriminating_paths = pag.get_discriminating_path(x, y, v)
                    for path in discriminating_paths:
                        if v in kwargs["separating_sets"].get((x, y), set()):
                            pag.modify_edge(v, y, mark_u="-", mark_v=">")
                        else:
                            pag.modify_edge(path[-3], v, mark_u=">", mark_v=">")
                            pag.modify_edge(v, y, mark_u=">", mark_v=">")

        if not inplace:
            return pag

    def rule_5(self, inplace=False, **kwargs):
        r"""
        Orient edges along an uncovered circle path.

        This rule is triggered when two nodes `u` and `v` are connected by an `o--o`
        edge, and there exists an uncovered circle path between them. Specifically:

        - `u` and `v` share an `o--o` edge.
        - There is a path ⟨u, …, v⟩ of length at least 4 made entirely of `o--o` edges.
        - The path is uncovered.
        - The second node and the second-to-last node on the path are not adjacent
        to the opposite endpoints (i.e., no edge between `path[0]` and `path[-2]`,
        and none between `path[1]` and `path[-1]`).

        When these conditions are met, the `o–o` edge between `u` and `v` is oriented,
        and all edges along the uncovered path are oriented as well.

        Parameters
        ----------
        inplace : bool, default False
            If True, update the current graph directly.
            If False, return a modified copy and leave the original untouched.

        Returns
        -------
        PAG or None
            The updated graph when `inplace=False`. Returns None when changes
            are applied in place.
        """
        pag = self if inplace else self.copy()

        for u, v in pag.edges:
            edge_marks = pag.get_edge_marks(u, v)
            if edge_marks[u] == "o" and edge_marks[v] == "o":
                paths = pag.get_paths(u, v, {("o", "o")})
                for path in paths:
                    if len(path) >= 4 and pag.is_uncovered(path):
                        if not pag.has_edge(path[0], path[-2]) and not pag.has_edge(
                            path[1], path[-1]
                        ):
                            pag.modify_edge(u, v, mark_u="-", mark_v="-")

                            for i in range(1, len(path) - 1):
                                pag.modify_edge(
                                    path[i], path[i + 1], mark_u="-", mark_v="-"
                                )

        if not inplace:
            return pag

    def rule_6(self, inplace=False, **kwargs):
        r"""
        If u -- v o--* w and u and w may or may not be adjacent, then orient v o--* w as v --* w.

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

            if len(u_candidates) > 0:
                for w in w_candidates:
                    pag.modify_edge(v, w, mark_u="-", mark_v=None)

        if not inplace:
            return pag

    def rule_7(self, inplace=False, **kwargs):
        r"""
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
            u_candidates = pag.get_neighbors(v, u_type="o", v_type="-")
            w_candidates = pag.get_neighbors(v, u_type="o", v_type=None)
            for u, w in product(u_candidates, w_candidates):
                if u == w or pag.has_edge(u, w):
                    continue

                self.modify_edge(v, w, mark_u="-", mark_v=None)

        if not inplace:
            return pag

    def rule_8(self, inplace=False, **kwargs):
        r"""
        if u --> v --> w or u --o v --> w, and u o--> w, then orient u o--> w as u --> w.

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
            u_candidates = pag.get_neighbors(
                v, u_type=">", v_type="-"
            ) or pag.get_neighbors(v, u_type="o", v_type="-")
            w_candidates = pag.get_neighbors(v, u_type="-", v_type=">")

            for u, w in product(u_candidates, w_candidates):
                if pag.has_edge(u, w):
                    marks_uw = pag.get_edge_marks(u, w)
                    if marks_uw[u] == "o" and marks_uw[w] == ">":
                        pag.modify_edge(u, w, mark_u="-", mark_v=">")

        if not inplace:
            return pag

    def rule_9(self, inplace=False, **kwargs):
        r"""
        Orient the edge `u → w` when an uncovered, potentially directed path supports it.

        This rule applies when:

        - The edge between `u` and `w` is `u o --> w` (circle at `u`, arrow at `w`), and
        - There exists an uncovered, potentially directed path
        ⟨u, v, …, w⟩ such that `v` and `w` are not adjacent.

        When these conditions are satisfied, the edge is oriented as `u --> w`.

        Parameters
        ----------
        inplace : bool, default False
            If True, apply the orientation directly to the existing graph.
            If False, operate on a copy and return the updated graph.

        Returns
        -------
        PAG or None
            The updated graph if `inplace=False`. Returns None when applying
            changes in place.
        """
        pag = self if inplace else self.copy()

        for u, w in list(pag.edges):
            marks = pag.get_edge_marks(u, w)
            if marks[u] == "o" and marks[w] == ">":
                pd_paths = pag.get_potentially_directed_paths(start=u, end=w)

                for path in pd_paths:
                    if len(path) >= 4 and pag.is_uncovered(path=path):
                        v = path[1]

                        if not pag.has_edge(v, w):
                            pag.modify_edge(u, w, mark_u="-", mark_v=">")

        if not inplace:
            return pag

    def rule_10(self, inplace=False, **kwargs):
        r"""
        Orient the edge `u -> w` based on uncovered, potentially directed paths.

        This rule applies when there are two nodes `v` and `x` such that
        `v → w ← x`, and the following conditions hold:

        - There is an uncovered, potentially directed path from `u` to `v`.
        - There is an uncovered, potentially directed path from `u` to `x`.
        - The first neighbors after `u` on these paths (call them `mu` and `omega`)
        are different.
        - `mu` and `omega` are not adjacent.

        When these conditions are met, the edge `u --> w` is oriented accordingly.

        Parameters
        ----------
        inplace : bool, default False
            If True, apply the orientation directly to the existing graph.
            If False, operate on a copy and return the updated graph.

        Returns
        -------
        PAG or None
            The updated graph if `inplace=False`. Returns None when changes
            are applied in place.
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
            "R4": pag.rule_4,
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
