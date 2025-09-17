from collections import deque
from itertools import product
from typing import Hashable, Iterable, Optional

from pgmpy.base import MAG


class PAG(MAG):
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

    def find_uncovered_pd_path(self, start, end, pag):
        """
        Find an uncovered potentially directed path from start to end.

        - Potentially directed: For every edge X—Y, the mark at X is not '>'.
        - Uncovered: For every triple <X, Y, Z> on the path, X and Z are not adjacent.
        """
        stack = [(start, [start])]

        while stack:
            current, path = stack.pop()

            if current == end:
                return path

            for neighbor in pag.get_neighbors(current):
                if neighbor in path:
                    continue  # avoid trivial cycles

                # Check potentially directed condition
                if pag.edges[current, neighbor]["marks"].get(current) == ">":
                    continue

                # Check uncovered condition for last triple
                if len(path) >= 2:
                    prev = path[-2]
                    if pag.has_edge(prev, neighbor):
                        continue

                # Extend path
                new_path = path + [neighbor]
                stack.append((neighbor, new_path))

        return None

    def rule_1(self, pag):
        """
        R1: If u *→ v ◦−◦ w and u and w are not adjacent, orient v ◦−◦ w as v → w.
        We require arrowhead at v on (u,v) and o--o (v,w).
        """
        changed = False

        for v in pag.nodes:
            u_candidates = [
                u
                for u in pag.get_neighbors(v)
                if pag.has_edge(u, v) and pag.edges[u, v]["marks"].get(v) == ">"
            ]

            w_candidates = [
                w
                for w in pag.get_neighbors(v)
                if pag.has_edge(v, w)
                and pag.edges[v, w]["marks"].get(v) == "o"
                and pag.edges[v, w]["marks"].get(w) == "o"
            ]

            for u, w in product(u_candidates, w_candidates):
                if pag.has_edge(u, w):  # skip if u and w are adjacent
                    continue

                if (
                    pag.edges[v, w]["marks"].get(v) == "o"
                    and pag.edges[v, w]["marks"].get(w) == "o"
                ):

                    pag.edges[v, w]["marks"][v] = "-"
                    pag.edges[v, w]["marks"][w] = ">"
                    changed = True
        return changed

    def rule_2(self, pag):
        """
        R2: If u → v → w or u *→ v → w, and u ◦−◦ w, then orient u → w.
        - Requires: arrowhead at v from u, and directed v → w.
        """
        changed = False

        for v in pag.nodes:
            # Step 1: u candidates: u *→ v (arrowhead at v)
            u_candidates = [
                u
                for u in pag.get_neighbors(v)
                if pag.has_edge(u, v) and pag.edges[u, v]["marks"].get(v) == ">"
            ]

            # Step 2: w candidates: v → w
            w_candidates = [
                w
                for w in pag.get_neighbors(v)
                if pag.has_edge(v, w)
                and pag.edges[v, w]["marks"].get(v) == "-"
                and pag.edges[v, w]["marks"].get(w) == ">"
            ]

            # Step 3: For each (u, w), check if u ◦−◦ w
            for u, w in product(u_candidates, w_candidates):
                if not pag.has_edge(u, w):
                    continue

                if (
                    pag.edges[u, w]["marks"].get(u) == "o"
                    and pag.edges[u, w]["marks"].get(w) == "o"
                ):

                    # Orient u ◦−◦ w into u → w
                    pag.edges[u, w]["marks"][u] = "-"
                    pag.edges[u, w]["marks"][w] = ">"
                    changed = True

        return changed

    def rule_3(self, pag):
        """
        R3: Fork Rule.
        If u *→ v ←* w, and u −◦ x ◦− w with u and w not adjacent,
        and x −◦ v, then orient x → v.
        """
        changed = False

        for v in pag.nodes:
            # Step 1: Find u, w such that u *→ v and w *→ v
            in_candidates = [
                u
                for u in pag.get_neighbors(v)
                if pag.has_edge(u, v) and pag.edges[u, v]["marks"].get(v) == ">"
            ]

            for u, w in product(in_candidates, in_candidates):
                if u == w:
                    continue

                # Skip if u and w are adjacent
                if pag.has_edge(u, w):
                    continue

                # Step 2: Find x such that u −◦ x ◦− w
                for x in pag.nodes:
                    if x in (u, v, w):
                        continue

                    if not (pag.has_edge(u, x) and pag.has_edge(w, x)):
                        continue

                    if not (
                        pag.edges[u, x]["marks"].get(u) == "-"
                        and pag.edges[u, x]["marks"].get(x) == "o"
                        and pag.edges[w, x]["marks"].get(w) == "-"
                        and pag.edges[w, x]["marks"].get(x) == "o"
                    ):
                        continue

                    # Step 3: Check if x −◦ v
                    if not pag.has_edge(x, v):
                        continue

                    if not (
                        pag.edges[x, v]["marks"].get(x) == "-"
                        and pag.edges[x, v]["marks"].get(v) == "o"
                    ):
                        continue

                    # Step 4: Orient x −◦ v into x → v
                    pag.edges[x, v]["marks"][v] = ">"
                    changed = True

        return changed

    def rule_5(self, pag):
        """
        R5: Uncovered Circle Path.
        If there is a path of ◦−◦ edges between u and v where:
        - every triple is unshielded,
        - u is not adjacent to the second-last node,
        - v is not adjacent to the second node,
        then orient all edges on that path as undirected.
        """
        changed = False

        # Step 1: consider only circle–circle edges
        circle_edges = [
            (u, v)
            for u, v in pag.edges
            if pag.has_edge(u, v)
            and pag.edges[u, v]["marks"].get(u) == "o"
            and pag.edges[u, v]["marks"].get(v) == "o"
        ]

        for u, v in circle_edges:
            stack = [(u, [u])]

            while stack:
                current, path = stack.pop()

                for nbr in pag.get_neighbors(current):
                    # Step 2: must be a circle–circle edge
                    if not pag.has_edge(current, nbr):
                        continue
                    if not (
                        pag.edges[current, nbr]["marks"].get(current) == "o"
                        and pag.edges[current, nbr]["marks"].get(nbr) == "o"
                    ):
                        continue
                    if nbr in path:
                        continue

                    new_path = path + [nbr]

                    # Step 3: reached v via a path of length ≥ 3
                    if nbr == v and len(new_path) >= 3:
                        # Check uncovered triples: no shortcuts between nodes at distance 2
                        if any(
                            pag.has_edge(new_path[i], new_path[i + 2])
                            for i in range(len(new_path) - 2)
                        ):
                            continue

                        # Check endpoint adjacency conditions
                        x, y = new_path[1], new_path[-2]
                        if pag.has_edge(u, y) or pag.has_edge(v, x):
                            continue

                        # Step 4: orient entire path as undirected
                        for a, b in zip(new_path, new_path[1:]):
                            pag.edges[a, b]["marks"][a] = "-"
                            pag.edges[a, b]["marks"][b] = "-"
                        pag.edges[u, v]["marks"][u] = "-"
                        pag.edges[u, v]["marks"][v] = "-"
                        changed = True

                        stack.clear()
                        break

                    # Step 5: continue DFS if still valid
                    if len(new_path) < 3 or not pag.has_edge(new_path[-3], nbr):
                        stack.append((nbr, new_path))

        return changed

    from itertools import product

    def rule_6(self, pag):
        """
        R6: If u − v ◦−◦ w, then orient v − w.
        """
        changed = False

        for v in pag.nodes:
            # Step 1: u candidates with undirected edge u − v
            u_candidates = [
                u
                for u in pag.get_neighbors(v)
                if pag.has_edge(u, v)
                and pag.edges[u, v]["marks"].get(u) == "-"
                and pag.edges[u, v]["marks"].get(v) == "-"
            ]

            # Step 2: w candidates with v ◦−◦ w
            w_candidates = [
                w
                for w in pag.get_neighbors(v)
                if pag.has_edge(v, w)
                and pag.edges[v, w]["marks"].get(v) == "o"
                and pag.edges[v, w]["marks"].get(w) == "o"
            ]

            # Step 3: For each u − v ◦−◦ w, orient v − w
            for u, w in product(u_candidates, w_candidates):
                pag.edges[v, w]["marks"][v] = "-"
                pag.edges[v, w]["marks"][w] = "-"
                changed = True

        return changed

    def rule_7(self, pag):
        """
        R7: Tail Propagation with Non-Adjacency.
        If u −◦ v ◦−◦ w and u,w not adjacent, then orient v − w.
        """
        changed = False

        for v in pag.nodes:
            # Step 1: u candidates with u −◦ v
            u_candidates = [
                u
                for u in pag.get_neighbors(v)
                if pag.has_edge(u, v)
                and pag.edges[u, v]["marks"].get(u) == "-"
                and pag.edges[u, v]["marks"].get(v) == "o"
            ]

            # Step 2: w candidates with v ◦−◦ w
            w_candidates = [
                w
                for w in pag.get_neighbors(v)
                if pag.has_edge(v, w)
                and pag.edges[v, w]["marks"].get(v) == "o"
                and pag.edges[v, w]["marks"].get(w) == "o"
            ]

            # Step 3: For each (u, w), check non-adjacency, then orient v − w
            for u, w in product(u_candidates, w_candidates):
                if pag.has_edge(u, w):  # skip if u and w adjacent
                    continue

                # Orient v ◦−◦ w into v − w
                pag.edges[v, w]["marks"][v] = "-"
                pag.edges[v, w]["marks"][w] = "-"
                changed = True

        return changed

    def rule_8(self, pag):
        """
        R8: Partial Arrow Completion.
        If (u → v → w) or (u −◦ v → w), and u ◦→ w, then orient u → w.
        """
        changed = False

        for v in pag.nodes:
            # Step 1: u candidates (u → v OR u −◦ v)
            u_candidates = [
                u
                for u in pag.get_neighbors(v)
                if pag.has_edge(u, v)
                and (
                    # u → v
                    (
                        pag.edges[u, v]["marks"].get(u) == "-"
                        and pag.edges[u, v]["marks"].get(v) == ">"
                    )
                    or (
                        pag.edges[u, v]["marks"].get(u) == "-"
                        and pag.edges[u, v]["marks"].get(v) == "o"
                    )
                )
            ]

            # Step 2: w candidates (v → w)
            w_candidates = [
                w
                for w in pag.get_neighbors(v)
                if pag.has_edge(v, w)
                and pag.edges[v, w]["marks"].get(v) == "-"
                and pag.edges[v, w]["marks"].get(w) == ">"
            ]

            # Step 3: For each (u, w), check if u ◦→ w
            for u, w in product(u_candidates, w_candidates):
                if not pag.has_edge(u, w):
                    continue

                if (
                    pag.edges[u, w]["marks"].get(u) == "o"
                    and pag.edges[u, w]["marks"].get(w) == ">"
                ):
                    # Orient u ◦→ w into u → w
                    pag.edges[u, w]["marks"][u] = "-"
                    changed = True

        return changed

    def rule_9(self, pag):
        """
        R9: Potentially Directed Path.
        If u ◦→ w and there is an uncovered potentially directed path <u, v, …, w>
        with w and v not adjacent, then orient u → w.
        """
        changed = False

        for u, w in pag.edges:
            # Step 1: Look for u ◦→ w
            if not (
                pag.has_edge(u, w)
                and pag.edges[u, w]["marks"].get(u) == "o"
                and pag.edges[u, w]["marks"].get(w) == ">"
            ):
                continue

            # BFS search for uncovered potentially directed paths
            queue = deque([(u, [u])])

            while queue:
                current, path = queue.popleft()

                # Step 2: Reached w with a valid path of length ≥ 3
                if current == w and len(path) >= 3:
                    is_valid = True

                    for i in range(len(path) - 1):
                        a, b = path[i], path[i + 1]

                        # Edge must be potentially directed a → b
                        if pag.edges[a, b]["marks"].get(a) == ">":
                            is_valid = False
                            break

                        # Check uncovered triples: no edge between (a, c)
                        if i >= 1:
                            prev = path[i - 1]
                            if pag.has_edge(prev, b):
                                is_valid = False
                                break

                    if is_valid:
                        v = path[1]
                        # Ensure w and v are not adjacent
                        if not pag.has_edge(v, w):
                            # Orient u ◦→ w → u → w
                            pag.edges[u, w]["marks"][u] = "-"
                            changed = True
                            break

                # Step 3: Continue exploring potentially directed edges
                for nbr in pag.get_neighbors(current):
                    if nbr in path:
                        continue
                    if pag.edges[current, nbr]["marks"].get(current) == ">":
                        continue
                    queue.append((nbr, path + [nbr]))

        return changed

    def rule_10(self, pag):
        """
        R10: Two Forks Rule.
        If u ◦→ w and there are two forks v → w ← x,
        and uncovered potentially directed paths from u to v and u to x
        with different first neighbors, then orient u → w.
        """
        changed = False

        for u, w in pag.edges:
            # Step 1: Look for u ◦→ w
            if not (
                pag.has_edge(u, w)
                and pag.edges[u, w]["marks"].get(u) == "o"
                and pag.edges[u, w]["marks"].get(w) == ">"
            ):
                continue

            # Step 2: Find forks into w (v → w and x → w)
            forks = [
                v
                for v in pag.get_neighbors(w)
                if pag.has_edge(v, w)
                and pag.edges[v, w]["marks"].get(v) == "-"
                and pag.edges[v, w]["marks"].get(w) == ">"
            ]

            # Need at least two distinct forks
            if len(forks) < 2:
                continue

            # Step 3: Check pairs of forks (v, x)
            for i in range(len(forks)):
                for j in range(i + 1, len(forks)):
                    v, x = forks[i], forks[j]

                    # Find uncovered potentially directed paths u → … → v and u → … → x
                    p1 = self.find_uncovered_pd_path(u, v, pag)
                    p2 = self.find_uncovered_pd_path(u, x, pag)

                    if not (p1 and p2):
                        continue

                    # Extract first neighbors after u
                    mu = p1[1] if len(p1) > 1 else v
                    omega = p2[1] if len(p2) > 1 else x

                    # Step 4: Must have different first neighbors
                    if mu == omega:
                        continue

                    # Step 5: Ensure those first neighbors are not adjacent
                    if pag.has_edge(mu, omega):
                        continue

                    # Step 6: Orient u ◦→ w into u → w
                    pag.edges[u, w]["marks"][u] = "-"
                    changed = True
                    break  # no need to keep searching once oriented

        return changed

    def apply_orientation_rules(self, rules=None, inplace=False, sepsets=None):
        """
        Apply a set of orientation rules (R1–R10) to the PAG.

        Parameters
        ----------
        rules : list[str], optional
            List of rule names to apply. If None, all rules R0–R10 are applied.

        inplace : bool, default False
            If True, modify the current PAG in place. Otherwise, work on a copy.

        sepsets : dict, optional
            Required for rules that depend on separating sets (R0, R4).

        Returns
        -------
        PAG
            The graph after applying the orientation rules.
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

        # Normalize and validate rule list
        if rules:
            rules_to_apply = [r.upper() for r in rules]
            unknown = set(rules_to_apply) - set(rules_map.keys())
            if unknown:
                raise ValueError(f"Unknown rules: {unknown}")
        else:
            rules_to_apply = list(rules_map.keys())

        # Iteratively apply rules until convergence
        changed = True
        while changed:
            changed = False
            for rule_name in rules_to_apply:
                rule_func = rules_map[rule_name]
                if rule_name in {"R0", "R4"}:
                    if sepsets is None:
                        raise ValueError(
                            f"Rule {rule_name} requires sepsets to be provided."
                        )
                    if rule_func(pag, sepsets):
                        changed = True
                else:
                    if rule_func(pag):
                        changed = True

        return pag
